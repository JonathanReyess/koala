#!/usr/bin/env python3
"""Evaluate checkpoint(s) written by train.py.

Reads back each fold's config.json (written by train.py) to reproduce the
*exact* same split it was trained on -- you don't need to re-pass
--split-mode/--seed unless you want to force a mismatch check.

For --split-mode signer_kfold, also re-derives all folds' signer partitions
and re-runs check_signer_partition() independently of train.py, before
reporting anything (see dataset.check_signer_partition).

Examples
--------
    python evaluate.py --pkl data/KSL77_joint_stream_47pt.pkl \
        --checkpoint-dir runs/random_legacy --out runs/random_legacy/eval.json

    python evaluate.py --features features_v2/features.npy \
        --mask features_v2/mask.npy --manifest features_v2/manifest.csv \
        --checkpoint-dir runs/signer_kfold_v2 --out runs/signer_kfold_v2/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transforms
from dataset import (
    Split,
    check_signer_partition,
    load_legacy_pkl,
    load_manifest_dataset,
    random_split,
    signer_kfold_splits,
)
from models import PoseCNN_LSTM_Attn


def load_fold(fold_dir: Path) -> tuple[dict, dict, torch.nn.Module]:
    with open(fold_dir / "config.json") as f:
        config = json.load(f)
    with open(fold_dir / "label_map.json") as f:
        label_map_json = json.load(f)
    label_map = {int(k): v for k, v in label_map_json["label_map"].items()}
    reverse_label_map = {int(k): v for k, v in label_map_json["reverse_label_map"].items()}

    model = PoseCNN_LSTM_Attn(
        num_classes=len(label_map),
        cnn_hidden=config["cnn_hidden"],
        lstm_hidden=config["lstm_hidden"],
        dropout_rate=config["dropout"],
    )
    state = torch.load(fold_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return config, {"label_map": label_map, "reverse_label_map": reverse_label_map}, model


def _top_k_accuracy_from_probs(probs: torch.Tensor, y: torch.Tensor, k: int) -> float:
    topk = probs.topk(k, dim=1).indices
    hit = (topk == y.unsqueeze(1)).any(dim=1)
    return hit.float().mean().item()


@torch.no_grad()
def predict_probs(
    model,
    features: np.ndarray,
    idx: np.ndarray,
    mask: np.ndarray | None = None,
    tta_mirror: bool = False,
) -> torch.Tensor:
    """Softmax probabilities (len(idx), num_classes) for features[idx].

    With tta_mirror, averages the softmax of each clip and of its mirror
    (transforms.mirror_clip) -- the same rule the browser uses. `features` must
    already be preprocessed (e.g. normalize_body_bulk) the way the checkpoint
    was trained.
    """
    X = torch.tensor(features[idx], dtype=torch.float32)
    probs = F.softmax(model(X), dim=1)

    if tta_mirror:
        sel_mask = mask[idx] if mask is not None else None
        mirrored = np.empty_like(features[idx])
        for i, feat in enumerate(features[idx]):
            coords = transforms.chw_to_tjc(feat)
            m = (
                sel_mask[i].astype(bool)
                if sel_mask is not None
                else transforms.infer_mask_from_coords(coords)
            )
            mirrored_coords, _ = transforms.mirror_clip(coords, m)
            mirrored[i] = transforms.tjc_to_chw(mirrored_coords)
        X_mirrored = torch.tensor(mirrored, dtype=torch.float32)
        probs_mirrored = F.softmax(model(X_mirrored), dim=1)
        probs = (probs + probs_mirrored) / 2.0
    return probs


@torch.no_grad()
def evaluate_split(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    split: Split,
    signer_ids: np.ndarray | None,
    mask: np.ndarray | None = None,
    tta_mirror: bool = False,
) -> dict:
    y_dense = np.array([split.label_map[int(l)] for l in labels], dtype=np.int64)
    y_test = torch.tensor(y_dense[split.test_idx], dtype=torch.long)

    probs = predict_probs(model, features, split.test_idx, mask=mask, tta_mirror=tta_mirror)

    preds = probs.argmax(1)

    top1 = (preds == y_test).float().mean().item()
    top3 = _top_k_accuracy_from_probs(probs, y_test, k=min(3, probs.shape[1]))
    macro_f1 = f1_score(y_test.numpy(), preds.numpy(), average="macro", zero_division=0)
    report = classification_report(
        y_test.numpy(), preds.numpy(), output_dict=True, zero_division=0
    )

    per_signer = None
    if signer_ids is not None:
        test_signers = signer_ids[split.test_idx]
        correct = (preds == y_test).numpy()
        per_signer = {}
        for signer in sorted(set(test_signers.tolist())):
            m = test_signers == signer
            per_signer[str(signer)] = {
                "n": int(m.sum()),
                "accuracy": float(correct[m].mean()) if m.sum() else None,
            }

    return {
        "top1": top1,
        "top3": top3,
        "macro_f1": macro_f1,
        "per_class_f1": {
            k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "per_signer_accuracy": per_signer,
        "n_test": len(split.test_idx),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", type=str, default=None)
    p.add_argument("--features", type=str, default=None)
    p.add_argument("--mask", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument(
        "--min-hand-frac",
        type=float,
        default=None,
        help="Manifest-mode only: must match the value train.py used for this checkpoint "
        "(saved in config.json) -- by default this is read back from config.json "
        "automatically so the split reproduces exactly. Passing a different value here "
        "raises an error, since it would silently evaluate against a different filtered "
        "dataset than the one the split (and thus the checkpoint's test set) was built from.",
    )
    p.add_argument(
        "--normalize-body",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Must match train.py's setting for this checkpoint (read back from config.json "
        "automatically by default; passing a mismatched value here refuses to run).",
    )
    p.add_argument(
        "--trim-idle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Must match train.py's setting for this checkpoint (read back from config.json "
        "automatically by default; passing a mismatched value here refuses to run).",
    )
    p.add_argument(
        "--trim-idle-motion-threshold",
        type=float,
        default=None,
        help="Must match train.py's setting for this checkpoint (read back from config.json "
        "automatically by default; passing a mismatched value here refuses to run).",
    )
    p.add_argument(
        "--tta-mirror",
        action="store_true",
        help="Eval-only test-time augmentation: average softmax over each test clip and its "
        "mirror (transforms.mirror_clip). Independent of training -- works with any "
        "checkpoint, doesn't need to match how it was trained.",
    )
    p.add_argument("--checkpoint-dir", type=str, required=True, help="Directory train.py wrote (contains fold_*/ subdirs)")
    p.add_argument("--out", type=str, default=None, help="Optional path to write results as JSON")
    return p


def _resolve_against_config(cli_value, config, key: str, default, fold_dir: Path):
    """Returns config[key] (falling back to `default` for older checkpoints
    that predate this key), or exits the process if `cli_value` was
    explicitly passed and disagrees with it -- using a different
    preprocessing setting than training used would silently evaluate
    against a different dataset than the one the split was built from.
    """
    config_value = config.get(key, default)
    if cli_value is not None and cli_value != config_value:
        flag = "--" + key.replace("_", "-")
        print(
            f"{flag}={cli_value} was passed, but this checkpoint was trained with "
            f"{key}={config_value} (from {fold_dir}/config.json). Using a different value here "
            "would evaluate against a differently-preprocessed dataset than the one the split "
            "was built from -- refusing.",
            file=sys.stderr,
        )
        sys.exit(2)
    return config_value


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    checkpoint_dir = Path(args.checkpoint_dir)
    fold_dirs = sorted(checkpoint_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        print(f"No fold_* subdirectories found under {checkpoint_dir}", file=sys.stderr)
        return 1

    first_config, _, _ = load_fold(fold_dirs[0])
    split_mode = first_config["split_mode"]
    seed = first_config["seed"]
    # Older checkpoints (trained before these flags existed) fall back to
    # train.py's own defaults for each.
    min_hand_frac = _resolve_against_config(args.min_hand_frac, first_config, "min_hand_frac", 0.0, fold_dirs[0])
    normalize_body = _resolve_against_config(args.normalize_body, first_config, "normalize_body", False, fold_dirs[0])
    trim_idle = _resolve_against_config(args.trim_idle, first_config, "trim_idle", False, fold_dirs[0])
    trim_idle_motion_threshold = _resolve_against_config(
        args.trim_idle_motion_threshold,
        first_config,
        "trim_idle_motion_threshold",
        transforms.DEFAULT_TRIM_MOTION_THRESHOLD,
        fold_dirs[0],
    )

    if args.pkl:
        features, labels = load_legacy_pkl(args.pkl)
        signer_ids = None
        mask = None
    else:
        if not (args.features and args.mask and args.manifest):
            print("Provide either --pkl, or all of --features/--mask/--manifest", file=sys.stderr)
            return 2
        features, mask, manifest = load_manifest_dataset(
            args.features, args.mask, args.manifest, min_hand_frac=min_hand_frac
        )
        labels = manifest["class_id"].to_numpy()
        signer_ids = manifest["signer_id"].to_numpy()

    if normalize_body:
        features = transforms.normalize_body_bulk(features, mask)
    if trim_idle:
        features, mask = transforms.trim_idle_bulk(
            features, mask, motion_threshold=trim_idle_motion_threshold
        )

    if split_mode == "random":
        splits = [random_split(labels, seed=seed)]
    else:
        assert signer_ids is not None, "signer_kfold checkpoints require --manifest"
        splits = signer_kfold_splits(labels, signer_ids, n_splits=len(fold_dirs), seed=seed)
        check_signer_partition(splits, signer_ids)  # independent re-check, per PR2 spec

    per_fold_results = []
    for fold_dir, split in zip(fold_dirs, splits):
        config, maps, model = load_fold(fold_dir)
        assert config["split_mode"] == split_mode, f"{fold_dir}: split_mode mismatch"
        # Sanity: the label map this checkpoint was trained with must match
        # the one we just rebuilt for the same seed/data, or the split isn't
        # actually reproducible.
        assert maps["label_map"] == split.label_map, (
            f"{fold_dir}: label_map from training doesn't match a freshly rebuilt split "
            "-- data or seed may have changed since training."
        )
        result = evaluate_split(
            model, features, labels, split, signer_ids, mask=mask, tta_mirror=args.tta_mirror
        )
        result["fold"] = split.fold
        per_fold_results.append(result)
        signer_note = f", n_test_signers={len(split.test_signers)}" if split.test_signers else ""
        print(
            f"fold {split.fold}: top1={result['top1']:.4f} top3={result['top3']:.4f} "
            f"macro_f1={result['macro_f1']:.4f} n_test={result['n_test']}{signer_note}"
        )

    summary: dict = {"split_mode": split_mode, "per_fold": per_fold_results}
    if len(per_fold_results) > 1:
        for metric in ("top1", "top3", "macro_f1"):
            values = np.array([r[metric] for r in per_fold_results])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
            print(f"{metric}: mean={values.mean():.4f} std={values.std():.4f} (n={len(values)} folds)")

        # Every signer is tested in exactly one fold, so we can safely combine
        # per-signer accuracy across folds into one out-of-fold table.
        combined_per_signer: dict = {}
        for r in per_fold_results:
            if r["per_signer_accuracy"]:
                for signer, stats in r["per_signer_accuracy"].items():
                    combined_per_signer[signer] = {**stats, "fold": r["fold"]}
        summary["per_signer_accuracy_combined"] = combined_per_signer or None

        if combined_per_signer:
            print("\nper-signer accuracy (fold = which test fold this signer was held out in):")
            print(f"  {'signer':<8}{'fold':<6}{'n':<6}{'accuracy':<10}")
            for signer in sorted(combined_per_signer):
                stats = combined_per_signer[signer]
                acc_str = f"{stats['accuracy']:.4f}" if stats["accuracy"] is not None else "n/a"
                print(f"  {signer:<8}{stats['fold']:<6}{stats['n']:<6}{acc_str:<10}")
    else:
        summary["top1"] = per_fold_results[0]["top1"]
        summary["top3"] = per_fold_results[0]["top3"]
        summary["macro_f1"] = per_fold_results[0]["macro_f1"]

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
