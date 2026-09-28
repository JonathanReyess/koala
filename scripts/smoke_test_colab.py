#!/usr/bin/env python3
"""One-shot smoke test for extract_landmarks.py, meant to be run on Colab
(real Linux MediaPipe, not the macOS build this repo was otherwise written
against) before kicking off a full extraction over the KSL-77 corpus.

It downloads the .task model(s), runs the detector on a handful of real
videos, prints:
  - the raw per-frame result object's field names (so you can eyeball them
    against extract_landmarks.result_to_47pt() if mediapipe's exact result
    shape ever differs from what this repo was written against)
  - each video's frame width x height, and separate left-hand/right-hand/pose
    detection fractions (not just the combined "any hand" fraction)
  - the final features/mask array shapes
and asserts there are no NaNs and the shapes are exactly (3, seq_len, 47) /
(seq_len, 47) per video.

Non-square videos + --save-overlay
-----------------------------------
MediaPipe Tasks vision graphs can log:
    "Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the
    square ROI."
This is a known, usually-benign warning from an internal ROI-cropping
calculator, but "usually" isn't good enough before running this over the
whole KSL-77 corpus -- if it actually indicates x/y distortion on non-square
frames, every downstream landmark would be subtly wrong. Use --save-overlay
DIR to render our own (3, 32, 47) feature array (not the raw MediaPipe
result -- so this also checks our own coords_seq/mask_seq indexing) back
onto the original frames at slots 0/10/20/31, so you can eyeball whether the
47 points actually land on the hands/shoulders or are offset.

Usage (Colab, after `git clone` + `pip install -r requirements.txt`):
    python scripts/smoke_test_colab.py \
        --videos-dir /content/drive/MyDrive/KSL_Project/KSL_Raw_Videos/77 \
        --n 3 --save-overlay /content/overlays
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import extract_landmarks as el  # noqa: E402

# BGR (cv2 drawing convention), one color per 47-point group.
LEFT_HAND_COLOR = (0, 0, 255)  # red
RIGHT_HAND_COLOR = (255, 144, 30)  # blue
POSE_COLOR = (0, 215, 255)  # orange
OVERLAY_GROUPS = [
    (range(0, 21), LEFT_HAND_COLOR, "left hand (0-20)"),
    (range(21, 42), RIGHT_HAND_COLOR, "right hand (21-41)"),
    (range(42, 47), POSE_COLOR, "pose (42-46)"),
]


def find_videos(videos_dir: Path, n: int) -> list[Path]:
    candidates = sorted(
        p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
    )
    return candidates[:n]


def read_sampled_frame(video_path: Path, slot: int, sequence_length: int) -> Optional[np.ndarray]:
    """Re-seeks to the same frame extract_single_video sampled into `slot`,
    for overlay rendering. (extract_single_video itself only keeps
    coords/mask, not raw pixels, to keep memory down during a full-corpus
    run -- this re-read is smoke-test-only.)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = el.sample_frame_indices(total_frames, sequence_length)
    if slot >= len(indices):
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[slot]))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def draw_overlay(frame_bgr: np.ndarray, coords_47x3: np.ndarray, mask_47: np.ndarray) -> np.ndarray:
    """coords_47x3/mask_47 are for ONE sampled frame slot, drawn straight from
    our own (3, 32, 47) features / (32, 47) mask arrays (transposed back to
    (47, 3) for this one slot) -- deliberately not from the raw MediaPipe
    result, so this also exercises our own left/right/pose index layout.
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    for idx_range, color, _label in OVERLAY_GROUPS:
        for j in idx_range:
            if not mask_47[j]:
                continue
            x, y = coords_47x3[j, 0], coords_47x3[j, 1]
            px, py = int(round(float(x) * w)), int(round(float(y) * h))
            cv2.circle(img, (px, py), 4, color, -1, lineType=cv2.LINE_AA)

    legend_y = 20
    for _idx_range, color, label in OVERLAY_GROUPS:
        cv2.rectangle(img, (10, legend_y - 12), (26, legend_y + 2), color, -1)
        cv2.putText(
            img, label, (32, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        legend_y += 22
    return img


def save_overlays_for_video(
    video_path: Path, features: np.ndarray, mask: np.ndarray, sequence_length: int, out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slots = sorted(set(np.linspace(0, sequence_length - 1, 4, dtype=int).tolist()))
    written = []
    for slot in slots:
        frame = read_sampled_frame(video_path, slot, sequence_length)
        if frame is None:
            print(f"  WARNING: could not re-read frame for slot {slot} (overlay skipped)")
            continue
        coords_47x3 = features[:, slot, :].T  # (3,T,J) -> this slot's (J,3)
        mask_47 = mask[slot]  # (T,J) -> this slot's (J,)
        annotated = draw_overlay(frame, coords_47x3, mask_47)
        dest = out_dir / f"{video_path.stem}_frame{slot:02d}.png"
        cv2.imwrite(str(dest), annotated)
        written.append(dest)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", type=str, required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--backend", choices=["holistic", "hand_pose"], default="holistic")
    p.add_argument("--sequence-length", type=int, default=el.SEQUENCE_LENGTH)
    p.add_argument("--model-dir", type=str, default="./mediapipe_models")
    p.add_argument("--holistic-model", type=str, default=None)
    p.add_argument("--hand-model", type=str, default=None)
    p.add_argument("--pose-model", type=str, default=None)
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument(
        "--save-overlay",
        type=str,
        default=None,
        metavar="DIR",
        help="Save PNGs of frames 0/10/20/31 (of the sampled 32) with our 47 "
        "points drawn on them, so you can visually confirm the points land "
        "on the hands/shoulders and aren't offset (see NORM_RECT note above).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    el.print_device_check()
    print()

    videos_dir = Path(args.videos_dir)
    videos = find_videos(videos_dir, args.n)
    if not videos:
        print(f"No .mp4 files found under {videos_dir}", file=sys.stderr)
        return 1
    print(f"Smoke-testing {len(videos)} video(s) from {videos_dir}:")
    for v in videos:
        print(f"  - {v.name}")
    print()

    overlay_dir = Path(args.save_overlay) if args.save_overlay else None

    all_ok = True
    with el.Detector(args.backend, args) as detector:
        for video_path in videos:
            print(f"--- {video_path.name} ---")

            # Peek at the raw result's field names on the first readable frame,
            # so a mediapipe version mismatch is obvious before trusting the
            # aggregated stats below.
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_result = detector.detect(frame_rgb)
                if args.backend == "holistic":
                    fields = [a for a in dir(raw_result) if not a.startswith("_")]
                    print(f"  HolisticLandmarkerResult fields: {fields}")
                else:
                    hand_result, pose_result = raw_result
                    print(f"  HandLandmarkerResult fields: {[a for a in dir(hand_result) if not a.startswith('_')]}")
                    print(f"  PoseLandmarkerResult fields: {[a for a in dir(pose_result) if not a.startswith('_')]}")
            else:
                print("  WARNING: could not read first frame for field-name peek")

            features, mask, stats = el.extract_single_video(
                str(video_path), detector, args.sequence_length
            )

            width, height = stats["frame_width"], stats["frame_height"]
            print(f"  frame size: {width}x{height}" + (" (NON-SQUARE)" if width != height else " (square)"))
            print(
                f"  left-hand detected: {stats['frac_frames_with_left_hand']:.2f}  "
                f"right-hand detected: {stats['frac_frames_with_right_hand']:.2f}  "
                f"pose detected: {stats['frac_frames_with_pose']:.2f}"
            )
            print(f"  features shape: {features.shape}  mask shape: {mask.shape}")

            if width != height:
                print(
                    "  NOTE: non-square frame. If you saw the MediaPipe warning "
                    "'Using NORM_RECT without IMAGE_DIMENSIONS is only supported "
                    "for the square ROI', this is the video shape it's about. "
                    "mp.Image(...) is built directly from this frame's own "
                    "(height, width) via its array shape, so the *top-level* "
                    "image already carries its real (non-square) dimensions -- "
                    "the warning is coming from an internal ROI-cropping "
                    "calculator, not from us handing MediaPipe a mis-shaped "
                    "image. Whether that internal calculator's default "
                    "full-frame ROI still ends up correct for a non-square "
                    "frame is exactly what --save-overlay is for: check the "
                    "saved PNGs below for offset before trusting this on the "
                    "full corpus."
                )

            if overlay_dir is not None:
                written = save_overlays_for_video(
                    video_path, features, mask, args.sequence_length, overlay_dir
                )
                print(f"  wrote {len(written)} overlay PNG(s) to {overlay_dir}")

            ok = True
            if features.shape != (3, args.sequence_length, el.NUM_JOINTS):
                print(f"  FAIL: unexpected features shape {features.shape}")
                ok = False
            if mask.shape != (args.sequence_length, el.NUM_JOINTS):
                print(f"  FAIL: unexpected mask shape {mask.shape}")
                ok = False
            if np.isnan(features).any():
                print("  FAIL: NaNs in features")
                ok = False
            print("  PASS" if ok else "  FAIL")
            all_ok = all_ok and ok
            print()

    print("SMOKE TEST: " + ("PASS" if all_ok else "FAIL"))
    if overlay_dir is not None:
        print(f"-> Inspect the overlay PNGs in {overlay_dir} before running the full extraction.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
