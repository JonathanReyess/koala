#!/usr/bin/env python3
"""Train PoseCNN_LSTM_Attn, CLI-driven, on either the legacy notebook data
or the manifest/features/mask output of extract_landmarks.py.

Examples
--------
Reproduce the notebook's own split/training (random split, legacy pkl):
    python train.py --pkl data/KSL77_joint_stream_47pt.pkl \
        --split-mode random --out-dir runs/random_legacy

Signer-grouped 5-fold, on freshly extracted features:
    python train.py --features features_v2/features.npy \
        --mask features_v2/mask.npy --manifest features_v2/manifest.csv \
        --split-mode signer_kfold --out-dir runs/signer_kfold_v2

Final deployment model, trained on all signers (see RESULTS.md's "config f"
and --stop-epoch's docstring below for how --stop-epoch is chosen):
    python train.py --features features_v2/features.npy \
        --mask features_v2/mask.npy --manifest features_v2/manifest.csv \
        --split-mode full --stop-epoch 34 --mirror-aug --normalize-body \
        --out-dir runs/full_model

Each run writes, per fold (fold_0/ for "random" and "full", fold_0/..fold_4/
for "signer_kfold"), under --out-dir:
    model.pt        best-val-accuracy checkpoint ("full" mode: final-epoch weights, no validation set to select a "best" one)
    label_map.json  {"label_map": {...}, "reverse_label_map": {...}}
    config.json     hyperparameters + split metadata (+ best_epoch; "full" mode also + stop_epoch/scheduler_t_max), read back by evaluate.py/export_onnx.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transforms
from dataset import (
    PoseDataset,
    build_datasets,
    build_label_maps,
    load_legacy_pkl,
    load_manifest_dataset,
    random_split,
    signer_kfold_splits,
)
from models import (
    DEFAULT_CNN_HIDDEN,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_LR,
    DEFAULT_LSTM_HIDDEN,
    PoseCNN_LSTM_Attn,
)


def set_seed(seed: int) -> None:
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


def train_one_fold(
    features: np.ndarray,
    labels: np.ndarray,
    split,
    args: argparse.Namespace,
    fold_dir: Path,
    device: torch.device,
    mask: np.ndarray | None = None,
) -> dict:
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds = build_datasets(
        features,
        labels,
        split,
        mask=mask,
        augment_train=True,
        mirror_aug=args.mirror_aug,
        mirror_p=0.5,
    )

    # drop_last=True: BatchNorm1d in the classifier head raises if a training
    # batch has exactly 1 sample (can happen on the last batch of an epoch
    # for small/uneven folds, e.g. signer_kfold). Eval-mode BatchNorm uses
    # running stats instead, so val/test loaders don't need this.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    if len(train_loader) == 0:
        raise ValueError(
            f"Training set for this fold has only {len(train_ds)} samples, smaller than "
            f"--batch-size {args.batch_size} (drop_last=True leaves 0 batches). Use a smaller "
            "--batch-size for this fold."
        )

    model = PoseCNN_LSTM_Attn(
        num_classes=len(split.label_map),
        cnn_hidden=args.cnn_hidden,
        lstm_hidden=args.lstm_hidden,
        dropout_rate=args.dropout,
    ).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_state = None
    best_epoch = None
    epochs_since_best = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        print(
            f"  epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            f"{'  (new best)' if is_best else ''}"
        )

        if args.patience and epochs_since_best >= args.patience:
            print(f"  early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
            break

    assert best_state is not None
    torch.save(best_state, fold_dir / "model.pt")

    with open(fold_dir / "label_map.json", "w") as f:
        json.dump(
            {
                "label_map": {str(k): v for k, v in split.label_map.items()},
                "reverse_label_map": {str(k): v for k, v in split.reverse_label_map.items()},
            },
            f,
            indent=2,
        )

    config = {
        "fold": split.fold,
        "cnn_hidden": args.cnn_hidden,
        "lstm_hidden": args.lstm_hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "epochs_run": epoch + 1,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split_mode": args.split_mode,
        "min_hand_frac": args.min_hand_frac,
        "mirror_aug": args.mirror_aug,
        "normalize_body": args.normalize_body,
        "trim_idle": args.trim_idle,
        "trim_idle_motion_threshold": args.trim_idle_motion_threshold,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_signers": sorted(split.train_signers) if split.train_signers else None,
        "val_signers": sorted(split.val_signers) if split.val_signers else None,
        "test_signers": sorted(split.test_signers) if split.test_signers else None,
        "n_train": len(split.train_idx),
        "n_val": len(split.val_idx),
        "n_test": len(split.test_idx),
    }
    with open(fold_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  saved best checkpoint (val_acc={best_val_acc:.4f}) -> {fold_dir}")
    return config


def train_full(
    features: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    mask: np.ndarray | None = None,
    signer_ids: np.ndarray | None = None,
) -> dict:
    """Trains the final deployment checkpoint on ALL samples -- no
    validation split, no test set, no early stopping (there's nothing held
    out to early-stop or select a "best" epoch against).

    The cosine scheduler is still built with T_max = --epochs (the same
    value the CV runs used), but the loop only runs --stop-epoch epochs, so
    the LR this model sees at epoch e matches what a CV fold's model saw at
    that same epoch e -- --stop-epoch picks a point on the *same* schedule,
    it doesn't change the schedule's shape.
    """
    assert args.stop_epoch is not None and args.stop_epoch > 0, (
        "--split-mode full requires --stop-epoch N (N > 0)"
    )

    fold_dir = out_dir / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)

    label_map, reverse_label_map = build_label_maps(labels)
    y_dense = np.array([label_map[int(l)] for l in labels], dtype=np.int64)

    train_ds = PoseDataset(
        features, y_dense, mask=mask, augment=True, mirror_aug=args.mirror_aug, mirror_p=0.5
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if len(train_loader) == 0:
        raise ValueError(
            f"Training set has only {len(train_ds)} samples, smaller than --batch-size "
            f"{args.batch_size} (drop_last=True leaves 0 batches). Use a smaller --batch-size."
        )

    model = PoseCNN_LSTM_Attn(
        num_classes=len(label_map),
        cnn_hidden=args.cnn_hidden,
        lstm_hidden=args.lstm_hidden,
        dropout_rate=args.dropout,
    ).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.stop_epoch):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        print(f"  epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

    torch.save(model.state_dict(), fold_dir / "model.pt")

    with open(fold_dir / "label_map.json", "w") as f:
        json.dump(
            {
                "label_map": {str(k): v for k, v in label_map.items()},
                "reverse_label_map": {str(k): v for k, v in reverse_label_map.items()},
            },
            f,
            indent=2,
        )

    config = {
        "fold": None,
        "cnn_hidden": args.cnn_hidden,
        "lstm_hidden": args.lstm_hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "epochs_run": args.stop_epoch,
        "stop_epoch": args.stop_epoch,
        "scheduler_t_max": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split_mode": "full",
        "min_hand_frac": args.min_hand_frac,
        "mirror_aug": args.mirror_aug,
        "normalize_body": args.normalize_body,
        "trim_idle": args.trim_idle,
        "trim_idle_motion_threshold": args.trim_idle_motion_threshold,
        # No validation set in "full" mode, so there's no val-based "best
        # epoch" to select -- the final epoch's weights are what's saved.
        "best_val_acc": None,
        "best_epoch": None,
        "train_signers": sorted(set(signer_ids.tolist())) if signer_ids is not None else None,
        "val_signers": None,
        "test_signers": None,
        "n_train": len(features),
        "n_val": 0,
        "n_test": 0,
    }
    with open(fold_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  saved final checkpoint (trained {args.stop_epoch} epochs, no validation) -> {fold_dir}")
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = p.add_argument_group("data")
    data.add_argument("--pkl", type=str, default=None, help="Legacy data/KSL77_joint_stream_47pt.pkl (random split only)")
    data.add_argument("--features", type=str, default=None, help="features.npy from extract_landmarks.py")
    data.add_argument("--mask", type=str, default=None, help="mask.npy from extract_landmarks.py (used for --mirror-aug/--normalize-body/--trim-idle)")
    data.add_argument("--manifest", type=str, default=None, help="manifest.csv from extract_landmarks.py")
    data.add_argument(
        "--min-hand-frac",
        type=float,
        default=0.0,
        help="Manifest-mode only: drop samples with frac_frames_with_any_hand below this "
        "(in addition to always dropping n_sampled_ok==0 / frac_frames_with_pose==0 rows). "
        "Default 0.0 disables this extra filter.",
    )

    p.add_argument(
        "--split-mode",
        choices=["random", "signer_kfold", "full"],
        default="random",
        help="'full' trains on all signers with no validation split and no early stopping -- "
        "the final deployment model. Requires --stop-epoch.",
    )
    p.add_argument("--n-splits", type=int, default=5, help="signer_kfold only")
    p.add_argument(
        "--stop-epoch",
        type=int,
        default=None,
        help="--split-mode full only: train exactly this many epochs (required for full mode). "
        "The cosine scheduler is still built with T_max=--epochs, so the LR schedule up to "
        "--stop-epoch matches what the CV runs saw at those same epochs -- this only changes "
        "when training stops, not the shape of the schedule.",
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")

    hp = p.add_argument_group("hyperparameters (defaults = notebook's Optuna best_params)")
    hp.add_argument("--cnn-hidden", type=int, default=DEFAULT_CNN_HIDDEN)
    hp.add_argument("--lstm-hidden", type=int, default=DEFAULT_LSTM_HIDDEN)
    hp.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT_RATE)
    hp.add_argument("--lr", type=float, default=DEFAULT_LR)
    hp.add_argument("--optimizer", choices=["adam", "rmsprop"], default="adam")
    hp.add_argument("--epochs", type=int, default=60)
    hp.add_argument("--batch-size", type=int, default=32)
    hp.add_argument("--patience", type=int, default=15, help="early stop after N epochs with no val improvement; 0 disables")

    tf = p.add_argument_group("feature transforms (see transforms.py; all off by default)")
    tf.add_argument(
        "--mirror-aug",
        action="store_true",
        help="Train only. With p=0.5 per sample per epoch, left/right-mirror the clip "
        "(transforms.mirror_clip) as a training-time augmentation. Pair with --tta-mirror "
        "in evaluate.py for test-time averaging (not required -- independent flags).",
    )
    tf.add_argument(
        "--normalize-body",
        action="store_true",
        help="Deterministic, applied to every sample (train/val/test) identically: subtract "
        "the clip's mean mid-shoulder point and divide by mean shoulder width "
        "(transforms.normalize_body). evaluate.py must use the same setting -- it reads this "
        "back from config.json automatically.",
    )
    tf.add_argument(
        "--trim-idle",
        action="store_true",
        help="Deterministic, applied to every sample (train/val/test) identically: crop to the "
        "first/last frame where either hand is detected and moving, then resample back to the "
        "original frame count (transforms.trim_idle). evaluate.py reads this back from "
        "config.json automatically.",
    )
    tf.add_argument(
        "--trim-idle-motion-threshold",
        type=float,
        default=transforms.DEFAULT_TRIM_MOTION_THRESHOLD,
        help="Only used with --trim-idle: minimum frame-to-frame wrist displacement "
        "(normalized units) to count as 'moving'.",
    )
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pkl:
        features, labels = load_legacy_pkl(args.pkl)
        signer_ids = None
        mask = None  # legacy .pkl has no mask; transforms fall back to inferring one
    else:
        if not (args.features and args.mask and args.manifest):
            print("Provide either --pkl, or all of --features/--mask/--manifest", file=sys.stderr)
            return 2
        features, mask, manifest = load_manifest_dataset(
            args.features, args.mask, args.manifest, min_hand_frac=args.min_hand_frac
        )
        labels = manifest["class_id"].to_numpy()
        signer_ids = manifest["signer_id"].to_numpy()

    if args.normalize_body:
        print("Applying --normalize-body to all samples...")
        features = transforms.normalize_body_bulk(features, mask)
    if args.trim_idle:
        print(f"Applying --trim-idle (motion threshold={args.trim_idle_motion_threshold}) to all samples...")
        features, mask = transforms.trim_idle_bulk(
            features, mask, motion_threshold=args.trim_idle_motion_threshold
        )

    t0 = time.time()

    if args.split_mode == "full":
        if args.stop_epoch is None:
            print("--split-mode full requires --stop-epoch N", file=sys.stderr)
            return 2
        config = train_full(features, labels, args, out_dir, device, mask=mask, signer_ids=signer_ids)
        with open(out_dir / "run_config.json", "w") as f:
            json.dump({"folds": [config], "elapsed_sec": time.time() - t0}, f, indent=2)
        print(f"Done in {time.time() - t0:.1f}s -> {out_dir}")
        return 0

    if args.split_mode == "random":
        splits = [random_split(labels, seed=args.seed)]
    else:
        if signer_ids is None:
            print("--split-mode signer_kfold requires --manifest (signer_id column)", file=sys.stderr)
            return 2
        splits = signer_kfold_splits(labels, signer_ids, n_splits=args.n_splits, seed=args.seed)

    fold_configs = []
    for split in splits:
        fold_idx = split.fold if split.fold is not None else 0
        fold_dir = out_dir / f"fold_{fold_idx}"
        print(f"=== fold {fold_idx} ({len(splits)} total) ===")
        fold_configs.append(train_one_fold(features, labels, split, args, fold_dir, device, mask=mask))

    with open(out_dir / "run_config.json", "w") as f:
        json.dump({"folds": fold_configs, "elapsed_sec": time.time() - t0}, f, indent=2)

    print(f"Done in {time.time() - t0:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
