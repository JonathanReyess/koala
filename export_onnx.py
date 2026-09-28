#!/usr/bin/env python3
"""Exports a trained PoseCNN_LSTM_Attn checkpoint (from train.py) to ONNX,
for the browser/PR 3 side (or any other ONNX Runtime consumer).

Input:  "input",  float32, (N, 3, 32, 47), N (batch) dynamic.
Output: "logits", float32, (N, num_classes) -- raw logits, not softmax
        (softmax/top-k/TTA-mirror averaging are inference-time concerns,
        not part of the exported graph).

Always runs a PyTorch-vs-onnxruntime parity check after exporting (on a
random-noise input and a "real-shaped" input in the actual normalized
coordinate range, both with more than one batch element to also exercise
the dynamic batch axis) and refuses to leave a mismatched model.onnx in
place if the check fails.

Usage:
    python export_onnx.py --checkpoint-dir runs/full_model --out runs/full_model/model.onnx
    # --checkpoint-dir may point directly at a fold dir (.../fold_0) or at
    # its parent (if it contains exactly one fold_*/ subdir, e.g. a
    # --split-mode full run, which always has just fold_0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import PoseCNN_LSTM_Attn, NUM_JOINTS

OPSET = 17
SEQUENCE_LENGTH = 32
MAX_ABS_DIFF_TOLERANCE = 1e-4


def resolve_fold_dir(checkpoint_dir: Path) -> Path:
    if (checkpoint_dir / "model.pt").exists():
        return checkpoint_dir
    fold_dirs = sorted(checkpoint_dir.glob("fold_*"))
    if len(fold_dirs) == 1:
        return fold_dirs[0]
    if len(fold_dirs) > 1:
        raise ValueError(
            f"{checkpoint_dir} has {len(fold_dirs)} fold_* subdirs ({[d.name for d in fold_dirs]}) "
            "-- point --checkpoint-dir at one specific fold directory to export just that checkpoint."
        )
    raise FileNotFoundError(f"No model.pt found at {checkpoint_dir} or in any fold_* subdir of it.")


def load_checkpoint(fold_dir: Path) -> tuple[torch.nn.Module, dict, dict]:
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


def export(model: torch.nn.Module, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, SEQUENCE_LENGTH, NUM_JOINTS, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
    )


def parity_check(model: torch.nn.Module, onnx_path: Path) -> bool:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    all_ok = True

    test_inputs = {
        # Pure noise, batch size 2 -- generic sanity check + dynamic-batch exercise.
        "random (batch=2)": torch.randn(2, 3, SEQUENCE_LENGTH, NUM_JOINTS, dtype=torch.float32),
        # Real-shaped: values in [0, 1], the actual normalized-coordinate
        # range landmarks are extracted in (see extract_landmarks.py) --
        # batch size 4 to further exercise the dynamic batch axis.
        "real-shaped (batch=4, [0,1] uniform)": torch.rand(
            4, 3, SEQUENCE_LENGTH, NUM_JOINTS, dtype=torch.float32
        ),
        # Single-sample batch -- the shape actual inference requests use.
        "real-shaped (batch=1)": torch.rand(1, 3, SEQUENCE_LENGTH, NUM_JOINTS, dtype=torch.float32),
    }

    with torch.no_grad():
        for name, x in test_inputs.items():
            torch_out = model(x).numpy()
            onnx_out = session.run(["logits"], {"input": x.numpy()})[0]
            max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
            ok = max_abs_diff < MAX_ABS_DIFF_TOLERANCE
            all_ok = all_ok and ok
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {name}: max abs diff = {max_abs_diff:.2e} (tolerance {MAX_ABS_DIFF_TOLERANCE:.0e})")

    return all_ok


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--out", type=str, default=None, help="Default: <checkpoint-dir>/model.onnx")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    checkpoint_dir = Path(args.checkpoint_dir)
    try:
        fold_dir = resolve_fold_dir(checkpoint_dir)
    except (ValueError, FileNotFoundError) as e:
        print(str(e), file=sys.stderr)
        return 2
    out_path = Path(args.out) if args.out else fold_dir / "model.onnx"

    model, config, _label_map_json = load_checkpoint(fold_dir)
    print(f"Loaded checkpoint from {fold_dir} (split_mode={config['split_mode']}, "
          f"cnn_hidden={config['cnn_hidden']}, lstm_hidden={config['lstm_hidden']})")

    export(model, out_path)
    print(f"Exported ONNX (opset {OPSET}) -> {out_path}")

    print("Running PyTorch-vs-onnxruntime parity check...")
    ok = parity_check(model, out_path)

    if not ok:
        print(
            f"PARITY CHECK FAILED: one or more inputs exceeded the {MAX_ABS_DIFF_TOLERANCE:.0e} "
            f"tolerance. Removing {out_path} -- do not use this ONNX file.",
            file=sys.stderr,
        )
        out_path.unlink(missing_ok=True)
        return 1

    print(f"PARITY CHECK PASSED. {out_path} matches the PyTorch model within {MAX_ABS_DIFF_TOLERANCE:.0e}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
