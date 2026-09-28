#!/usr/bin/env python3
"""Generates golden test vectors for the TypeScript/browser port (PR 3) to
verify against -- see docs/PREPROCESSING.md §6 and preprocessing_spec.json.

For each of a few real clips, writes: the raw landmarks + mask (the input
a browser port starts from), the normalized landmarks (expected output of
normalize_body, preprocessing_spec.json's "normalize_body" section), and
the expected model logits for both the original and mirrored input plus
the final TTA-averaged probabilities (preprocessing_spec.json's
"mirror_tta" section) -- computed with the *actual trained model*, so a
correct browser port should reproduce these numbers (within floating-point
noise) end to end.

Requires a trained checkpoint (--checkpoint-dir); this only makes sense to
run once the production ("config f") model has been trained -- see
RESULTS.md and the Colab commands at the end of AUDIT.md/this repo's PR
description for how that checkpoint gets produced.

Usage:
    python scripts/generate_golden_vectors.py \
        --checkpoint-dir runs/full_model --out golden_vectors.json
    # Uses data/KSL77_joint_stream_47pt.pkl by default (mask inferred from
    # all-zero joints, since the legacy pkl has none). To use real
    # extract_landmarks.py output with a real mask instead:
    python scripts/generate_golden_vectors.py --checkpoint-dir runs/full_model \
        --features features_v2/features.npy --mask features_v2/mask.npy \
        --manifest features_v2/manifest.csv --out golden_vectors.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transforms  # noqa: E402
from dataset import load_legacy_pkl, load_manifest_dataset  # noqa: E402
from models import PoseCNN_LSTM_Attn  # noqa: E402


def resolve_fold_dir(checkpoint_dir: Path) -> Path:
    if (checkpoint_dir / "model.pt").exists():
        return checkpoint_dir
    fold_dirs = sorted(checkpoint_dir.glob("fold_*"))
    if len(fold_dirs) == 1:
        return fold_dirs[0]
    if len(fold_dirs) > 1:
        raise ValueError(
            f"{checkpoint_dir} has {len(fold_dirs)} fold_* subdirs -- point --checkpoint-dir "
            "at one specific fold directory."
        )
    raise FileNotFoundError(f"No model.pt found at {checkpoint_dir} or in any fold_* subdir of it.")


def load_checkpoint(fold_dir: Path):
    with open(fold_dir / "config.json") as f:
        config = json.load(f)
    with open(fold_dir / "label_map.json") as f:
        label_map_json = json.load(f)
    label_map = {int(k): v for k, v in label_map_json["label_map"].items()}

    model = PoseCNN_LSTM_Attn(
        num_classes=len(label_map),
        cnn_hidden=config["cnn_hidden"],
        lstm_hidden=config["lstm_hidden"],
        dropout_rate=config["dropout"],
    )
    state = torch.load(fold_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, config, label_map_json


def generate_vector(model, features_chw: np.ndarray, mask_tj: np.ndarray | None, label_map_json: dict) -> dict:
    """features_chw: (3, T, J) raw (unnormalized) landmarks for one clip."""
    raw_coords = transforms.chw_to_tjc(features_chw)  # (T, J, 3)
    raw_mask = (
        mask_tj.astype(np.uint8)
        if mask_tj is not None
        else transforms.infer_mask_from_coords(raw_coords).astype(np.uint8)
    )

    normalized_coords = transforms.normalize_body(raw_coords, raw_mask)
    mirrored_coords, _mirrored_mask = transforms.mirror_clip(normalized_coords, raw_mask)

    x_original = torch.tensor(
        transforms.tjc_to_chw(normalized_coords), dtype=torch.float32
    ).unsqueeze(0)
    x_mirrored = torch.tensor(
        transforms.tjc_to_chw(mirrored_coords), dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        logits_original = model(x_original)[0]
        logits_mirrored = model(x_mirrored)[0]
        probs = (F.softmax(logits_original, dim=0) + F.softmax(logits_mirrored, dim=0)) / 2.0

    predicted_dense_idx = int(probs.argmax().item())
    reverse_label_map = label_map_json["reverse_label_map"]

    return {
        "raw_landmarks": raw_coords.astype(np.float64).tolist(),
        "raw_mask": raw_mask.astype(int).tolist(),
        "normalized_landmarks": normalized_coords.astype(np.float64).tolist(),
        "model_input_shape": [1, 3, raw_coords.shape[0], raw_coords.shape[1]],
        "logits_original": logits_original.numpy().astype(np.float64).tolist(),
        "logits_mirrored": logits_mirrored.numpy().astype(np.float64).tolist(),
        "probs_tta_averaged": probs.numpy().astype(np.float64).tolist(),
        "predicted_class_dense_index": predicted_dense_idx,
        "predicted_class_original_id": reverse_label_map.get(str(predicted_dense_idx)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--pkl", type=str, default="data/KSL77_joint_stream_47pt.pkl")
    p.add_argument("--features", type=str, default=None, help="Use extract_landmarks.py output instead of --pkl")
    p.add_argument("--mask", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--n", type=int, default=3)
    p.add_argument(
        "--indices", type=int, nargs="+", default=None, help="Explicit sample indices (overrides --n)"
    )
    p.add_argument("--out", type=str, default="golden_vectors.json")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    checkpoint_dir = Path(args.checkpoint_dir)
    try:
        fold_dir = resolve_fold_dir(checkpoint_dir)
    except (ValueError, FileNotFoundError) as e:
        print(str(e), file=sys.stderr)
        return 2

    model, config, label_map_json = load_checkpoint(fold_dir)
    print(f"Loaded checkpoint from {fold_dir} (split_mode={config['split_mode']})")

    if args.features:
        if not (args.mask and args.manifest):
            print("--features requires --mask and --manifest too", file=sys.stderr)
            return 2
        features, mask, manifest = load_manifest_dataset(args.features, args.mask, args.manifest)
        source = f"{args.features} (mask={args.mask}, manifest={args.manifest})"
        video_paths = manifest["video_path"].tolist()
    else:
        features, _labels = load_legacy_pkl(args.pkl)
        mask = None
        source = args.pkl
        video_paths = None

    n = len(features)
    if args.indices:
        indices = args.indices
    else:
        indices = np.linspace(0, n - 1, args.n, dtype=int).tolist()

    vectors = []
    for idx in indices:
        vec = generate_vector(model, features[idx], mask[idx] if mask is not None else None, label_map_json)
        vec["sample_index"] = int(idx)
        vec["source"] = source
        if video_paths is not None:
            vec["video_path"] = video_paths[idx]
        vectors.append(vec)
        print(
            f"  sample {idx}: predicted_class_dense_index={vec['predicted_class_dense_index']} "
            f"(original_id={vec['predicted_class_original_id']})"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "spec_version": "1.0.0",
                "checkpoint_dir": str(fold_dir),
                "vectors": vectors,
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(vectors)} golden vector(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
