#!/usr/bin/env python3
"""Compares frame-level hand/pose detection rates between the legacy
notebook extraction (data/KSL77_joint_stream_47pt.pkl, MediaPipe legacy
solutions.Holistic, silent zero-fill) and a new extract_landmarks.py run
(MediaPipe Tasks HolisticLandmarker, explicit mask.npy) -- to answer "does
the new Tasks-API extraction find hands less often than the old legacy one?"

numpy-only (plus stdlib pickle/argparse) so this can run anywhere the .pkl
and mask.npy are, without needing torch/pandas/mediapipe installed.

47-point layout (both old and new): left hand 0-20, right hand 21-41,
pose 42-46.

Old features: a hand is "missing" in a frame iff every (x, y, z) in that
hand's 21-joint block is exactly 0 -- the legacy pipeline's silent
zero-fill convention (see AUDIT.md §3). This can't distinguish "MediaPipe
found nothing" from "MediaPipe found the hand exactly at the origin", but
that's an inherent limitation of data that was never given a real mask, not
something this script can fix after the fact.

Usage:
    python scripts/compare_hand_detection.py \
        --old-pkl data/KSL77_joint_stream_47pt.pkl \
        --new-mask features_v2/mask.npy
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

LEFT_HAND = slice(0, 21)
RIGHT_HAND = slice(21, 42)
POSE = slice(42, 47)


def rates_from_old_pkl(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        features, _labels = pickle.load(f)
    features = np.asarray(features, dtype=np.float32)  # (N, 3, T, J)
    # -> (N, T, J, 3) so "all-zero across x,y,z" is a per-(sample,frame,joint) check
    coords = np.transpose(features, (0, 2, 3, 1))
    joint_present = np.any(coords != 0, axis=-1)  # (N, T, J)
    return _rates_from_joint_present(joint_present)


def rates_from_new_mask(mask_path: str) -> dict:
    mask = np.load(mask_path).astype(bool)  # (N, T, J)
    return _rates_from_joint_present(mask)


def _rates_from_joint_present(joint_present: np.ndarray) -> dict:
    """joint_present: (N, T, J) bool. Returns frame-level detection rates:
    fraction of all (sample, frame) pairs where left/right/any hand, and
    pose, are present. A hand block counts as "present" in a frame if any
    of its 21 joints are marked present (matches how both pipelines write
    a whole hand's mask/zero-fill together, not joint-by-joint)."""
    n_samples, n_frames = joint_present.shape[0], joint_present.shape[1]
    total = n_samples * n_frames

    left_present = joint_present[:, :, LEFT_HAND].any(axis=-1)
    right_present = joint_present[:, :, RIGHT_HAND].any(axis=-1)
    any_hand_present = left_present | right_present
    pose_present = joint_present[:, :, POSE].any(axis=-1)

    return {
        "n_samples": int(n_samples),
        "n_frames_per_sample": int(n_frames),
        "n_total_frames": int(total),
        "left_hand_rate": float(left_present.sum() / total),
        "right_hand_rate": float(right_present.sum() / total),
        "any_hand_rate": float(any_hand_present.sum() / total),
        "pose_rate": float(pose_present.sum() / total),
    }


def print_rates(name: str, rates: dict) -> None:
    print(f"{name}: {rates['n_samples']} samples x {rates['n_frames_per_sample']} frames "
          f"= {rates['n_total_frames']} total frames")
    print(f"  left hand detected:  {rates['left_hand_rate']:.4f}")
    print(f"  right hand detected: {rates['right_hand_rate']:.4f}")
    print(f"  any hand detected:   {rates['any_hand_rate']:.4f}")
    print(f"  pose detected:       {rates['pose_rate']:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-pkl", type=str, default="data/KSL77_joint_stream_47pt.pkl")
    p.add_argument("--new-mask", type=str, required=True, help="Path to mask.npy from extract_landmarks.py")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    old_rates = rates_from_old_pkl(args.old_pkl)
    new_rates = rates_from_new_mask(args.new_mask)

    print("=== OLD (legacy solutions.Holistic, notebook/src/backend) ===")
    print_rates(args.old_pkl, old_rates)
    print()
    print("=== NEW (Tasks HolisticLandmarker, extract_landmarks.py) ===")
    print_rates(args.new_mask, new_rates)
    print()

    print("=== DELTA (new - old) ===")
    for key in ("left_hand_rate", "right_hand_rate", "any_hand_rate", "pose_rate"):
        delta = new_rates[key] - old_rates[key]
        direction = "higher" if delta > 0 else ("lower" if delta < 0 else "same")
        print(f"  {key}: {delta:+.4f} ({direction} in the new Tasks extraction)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
