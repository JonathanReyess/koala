#!/usr/bin/env python3
"""Extract 47-point joint sequences from raw KSL-77 videos using the MediaPipe
Tasks API (HolisticLandmarker, or HandLandmarker+PoseLandmarker as a fallback).

Replaces the MediaPipe-extraction cells of notebook/KSL.ipynb. Unlike the
notebook (and src/backend/app.py), missing landmarks are recorded in an
explicit per-joint mask instead of being silently zero-filled.

Expected input layout (KSL-77 raw videos, per PR2 spec):
    <src>/<class_id>/<NN>_<class_id>.MP4        (NN = 00-19, the signer id)

Output (written to <out>/):
    features.npy   float32 (N, 3, 32, 47)  -- same layout src/backend/app.py uses
    mask.npy        uint8  (N, 32, 47)      -- 1 = joint detected, 0 = missing
    manifest.csv    idx, video_path, class_id, signer_id, n_frames,
                     n_sampled_ok, frac_frames_with_pose,
                     frac_frames_with_left_hand, frac_frames_with_right_hand,
                     frac_frames_with_any_hand
                     (n_sampled_ok == 0 means the video failed to decode at
                     all, e.g. a truncated/corrupted file -- "moov atom not
                     found" and similar cv2/ffmpeg errors leave every sampled
                     frame unread, so frac_frames_with_pose is also 0 for
                     these. dataset.py drops such rows before splitting;
                     see its --min-hand-frac / drop_undetected_samples.)
    mismatches.csv  video_path, folder_class_id, filename_class_id
                     (written whenever a filename's embedded class id disagrees
                     with the folder it's found in; see --strict)
    cache/          one .npz per video, so re-running after a Colab disconnect
                     skips everything already extracted

Usage (Colab):
    python extract_landmarks.py --src /content/drive/MyDrive/KSL_Project/KSL_Raw_Videos \
        --out /content/drive/MyDrive/KSL_Project/features_v2

Usage (local device check, no video processing):
    python extract_landmarks.py --device-check
"""
from __future__ import annotations

import argparse
import csv
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - only hit if opencv isn't installed
    cv2 = None

NUM_JOINTS = 47
SEQUENCE_LENGTH = 32
# Nose, left shoulder, right shoulder, left elbow, right elbow -- same indices
# as src/backend/app.py / notebook/KSL.ipynb, into the 33-point BlazePose topology.
POSE_INDICES = [0, 11, 12, 13, 14]
# BlazePose wrist indices, used to disambiguate which detected hand is the
# subject's left/right hand in the `hand_pose` fallback backend (see
# `assign_hands_by_wrist` below).
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16

# <NN>_<class_id>.mp4, case-insensitive extension (raw files are shipped as .MP4).
FILENAME_RE = re.compile(r"^(\d{2})_(\d+)\.mp4$", re.IGNORECASE)

DEFAULT_MODEL_URLS = {
    "holistic": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


# ---------------------------------------------------------------------------
# Device / environment check
# ---------------------------------------------------------------------------


def print_device_check() -> None:
    system = platform.system()
    print(f"platform.system(): {system}")
    print(f"platform.platform(): {platform.platform()}")
    try:
        import mediapipe as mp

        print(f"mediapipe version: {getattr(mp, '__version__', 'unknown')}")
    except ImportError:
        print("mediapipe: NOT INSTALLED")
    if system == "Darwin":
        print(
            "\nWARNING: MediaPipe Tasks vision graphs (HolisticLandmarker, "
            "HandLandmarker, PoseLandmarker) are known to abort on macOS in "
            "some builds with 'Check failed: service_ Service is unavailable' "
            "coming from a Metal-backed calculator (DrishtiMetalHelper), "
            "regardless of BaseOptions.Delegate.CPU. This is an environment "
            "issue, not a bug in this script. Run extraction on Colab/Linux "
            "instead; use --device-check there to confirm before a full run."
        )


# ---------------------------------------------------------------------------
# Model asset download/cache
# ---------------------------------------------------------------------------


def ensure_model(path_or_none: Optional[str], url: str, model_dir: Path) -> str:
    if path_or_none:
        return path_or_none
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / Path(url).name
    if not dest.exists():
        print(f"Downloading {url} -> {dest}")
        urlretrieve(url, dest)
    return str(dest)


# ---------------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------------


@dataclass
class VideoItem:
    idx: int
    video_path: str
    class_id: int
    signer_id: str  # keep as zero-padded string ("00".."19")


def discover_videos(src: Path, strict: bool, mismatches_path: Path) -> list[VideoItem]:
    items: list[VideoItem] = []
    mismatches: list[tuple[str, str, str]] = []

    class_dirs = sorted((p for p in src.iterdir() if p.is_dir()), key=lambda p: p.name)
    idx = 0
    for class_dir in class_dirs:
        if not class_dir.name.isdigit():
            continue
        folder_class_id = class_dir.name
        for video_path in sorted(class_dir.iterdir()):
            if not video_path.is_file():
                continue
            m = FILENAME_RE.match(video_path.name)
            if not m:
                continue
            signer_id, filename_class_id = m.group(1), m.group(2)
            if filename_class_id != folder_class_id:
                msg = (
                    f"class id mismatch: folder={folder_class_id!r} "
                    f"filename={filename_class_id!r} ({video_path})"
                )
                if strict:
                    raise AssertionError(msg)
                print(f"WARNING: {msg}", file=sys.stderr)
                mismatches.append((str(video_path), folder_class_id, filename_class_id))
                continue
            items.append(
                VideoItem(
                    idx=idx,
                    video_path=str(video_path),
                    class_id=int(folder_class_id),
                    signer_id=signer_id,
                )
            )
            idx += 1

    if mismatches:
        mismatches_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mismatches_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_path", "folder_class_id", "filename_class_id"])
            w.writerows(mismatches)
        print(f"Logged {len(mismatches)} mismatched filename(s) to {mismatches_path}")

    return items


# ---------------------------------------------------------------------------
# Result -> 47-point coords + mask
#
# Kept as one small function per field-access site so a version mismatch in
# HolisticLandmarkerResult / HandLandmarkerResult / PoseLandmarkerResult only
# needs a fix here.
# ---------------------------------------------------------------------------


def _unwrap_landmark_list(value):
    """Handles both `[Landmark, ...]` and `[[Landmark, ...]]` result shapes.

    Some MediaPipe Tasks result fields are a flat list of landmarks (single
    subject), others are a list of per-detected-instance lists. Holistic only
    ever has one subject, so if we see a list of lists, take the first.
    """
    if not value:
        return None
    first = value[0]
    if isinstance(first, (list, tuple)):
        return first if first else None
    return value


def _landmarks_to_xyz(landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def assign_hands_by_wrist(
    hand_landmarks_list: list,
    handedness_list: list,
    pose_landmarks: Optional[list],
) -> tuple[Optional[list], Optional[list]]:
    """Decide which detected hand (0, 1, or 2 of them) is the subject's left
    hand vs right hand.

    Preferred method: match each detected hand's landmark centroid to the
    nearer of the pose's left/right wrist (BlazePose indices 15/16). This is
    robust to whether the source video is mirrored and matches how the old
    legacy `solutions.Holistic` pipeline resolved left/right (by anatomical
    position, not by MediaPipe's standalone hand-detector handedness label).

    Fallback (only used when no pose was detected in this frame): trust
    HandLandmarker's own `handedness` classification. NOTE: MediaPipe's Hands
    handedness is documented as computed assuming a mirrored (selfie-style)
    input image; if the KSL-77 raw videos are not mirrored, this fallback's
    left/right may be swapped relative to the wrist-matching method above.
    This only affects frames where pose detection failed, which are already
    partially unreliable.
    """
    if not hand_landmarks_list:
        return None, None

    if pose_landmarks is not None:
        left_wrist = np.array(
            [pose_landmarks[LEFT_WRIST_IDX].x, pose_landmarks[LEFT_WRIST_IDX].y]
        )
        right_wrist = np.array(
            [pose_landmarks[RIGHT_WRIST_IDX].x, pose_landmarks[RIGHT_WRIST_IDX].y]
        )
        left_hand, right_hand = None, None
        for hand in hand_landmarks_list:
            centroid = np.mean([[lm.x, lm.y] for lm in hand], axis=0)
            d_left = np.linalg.norm(centroid - left_wrist)
            d_right = np.linalg.norm(centroid - right_wrist)
            if d_left <= d_right:
                if left_hand is None or d_left < left_hand[0]:
                    left_hand = (d_left, hand)
            else:
                if right_hand is None or d_right < right_hand[0]:
                    right_hand = (d_right, hand)
        return (
            left_hand[1] if left_hand else None,
            right_hand[1] if right_hand else None,
        )

    # Fallback: standalone handedness label (mirrored-image assumption).
    left_hand, right_hand = None, None
    for hand, handedness in zip(hand_landmarks_list, handedness_list):
        label = handedness[0].category_name if handedness else None
        if label == "Left":
            left_hand = hand
        elif label == "Right":
            right_hand = hand
    return left_hand, right_hand


def result_to_47pt(result, backend: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert one frame's detector result into (coords(47,3), mask(47,)).

    `result` is:
      - a single HolisticLandmarkerResult, when backend == "holistic"
      - a (HandLandmarkerResult, PoseLandmarkerResult) tuple, when
        backend == "hand_pose"
    """
    coords = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    mask = np.zeros((NUM_JOINTS,), dtype=np.uint8)

    if backend == "holistic":
        pose_landmarks = _unwrap_landmark_list(getattr(result, "pose_landmarks", None))
        left_hand = _unwrap_landmark_list(getattr(result, "left_hand_landmarks", None))
        right_hand = _unwrap_landmark_list(getattr(result, "right_hand_landmarks", None))
    elif backend == "hand_pose":
        hand_result, pose_result = result
        pose_landmarks = _unwrap_landmark_list(
            getattr(pose_result, "pose_landmarks", None)
        )
        hand_landmarks_list = getattr(hand_result, "hand_landmarks", None) or []
        handedness_list = getattr(hand_result, "handedness", None) or []
        left_hand, right_hand = assign_hands_by_wrist(
            hand_landmarks_list, handedness_list, pose_landmarks
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if left_hand is not None:
        coords[0:21] = _landmarks_to_xyz(left_hand)
        mask[0:21] = 1
    if right_hand is not None:
        coords[21:42] = _landmarks_to_xyz(right_hand)
        mask[21:42] = 1
    if pose_landmarks is not None:
        for i, pose_index in enumerate(POSE_INDICES):
            lm = pose_landmarks[pose_index]
            coords[42 + i] = [lm.x, lm.y, lm.z]
            mask[42 + i] = 1

    return coords, mask


# ---------------------------------------------------------------------------
# Per-video extraction
# ---------------------------------------------------------------------------


def sample_frame_indices(total_frames: int, sequence_length: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    return np.linspace(0, total_frames - 1, sequence_length, dtype=int)


class Detector:
    """Wraps either a single HolisticLandmarker or a HandLandmarker+PoseLandmarker
    pair behind one `.detect(frame_rgb) -> raw_result` call, and remembers which
    backend it is so `result_to_47pt` knows how to read the result.
    """

    def __init__(self, backend: str, args: argparse.Namespace):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        from mediapipe.tasks.python.core.base_options import BaseOptions as BO

        self.mp = mp
        self.backend = backend
        model_dir = Path(args.model_dir)

        if backend == "holistic":
            model_path = ensure_model(
                args.holistic_model, DEFAULT_MODEL_URLS["holistic"], model_dir
            )
            opts = vision.HolisticLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=model_path, delegate=BO.Delegate.CPU
                ),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=args.min_detection_confidence,
                min_hand_landmarks_confidence=args.min_detection_confidence,
            )
            self.holistic = vision.HolisticLandmarker.create_from_options(opts)
        elif backend == "hand_pose":
            hand_model_path = ensure_model(
                args.hand_model, DEFAULT_MODEL_URLS["hand"], model_dir
            )
            pose_model_path = ensure_model(
                args.pose_model, DEFAULT_MODEL_URLS["pose"], model_dir
            )
            hand_opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=hand_model_path, delegate=BO.Delegate.CPU
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=args.min_detection_confidence,
            )
            pose_opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=pose_model_path, delegate=BO.Delegate.CPU
                ),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=args.min_detection_confidence,
            )
            self.hand = vision.HandLandmarker.create_from_options(hand_opts)
            self.pose = vision.PoseLandmarker.create_from_options(pose_opts)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    def detect(self, frame_rgb: np.ndarray):
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        if self.backend == "holistic":
            return self.holistic.detect(mp_image)
        hand_result = self.hand.detect(mp_image)
        pose_result = self.pose.detect(mp_image)
        return hand_result, pose_result

    def close(self) -> None:
        """Releases the underlying MediaPipe Tasks graph(s).

        Without this, letting the Detector (and its HolisticLandmarker /
        HandLandmarker / PoseLandmarker) get garbage-collected instead of
        explicitly closed raises a TypeError out of their __del__ at
        interpreter shutdown (a known MediaPipe Tasks Python behavior).
        Always call this (or use `with Detector(...) as d:`) instead of
        just letting it go out of scope.
        """
        if self.backend == "holistic":
            self.holistic.close()
        else:
            self.hand.close()
            self.pose.close()

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def extract_single_video(
    video_path: str, detector: Detector, sequence_length: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (features(3, seq_len, 47), mask(seq_len, 47), stats dict)."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required to read video files.")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = sample_frame_indices(total_frames, sequence_length)

    coords_seq = np.zeros((sequence_length, NUM_JOINTS, 3), dtype=np.float32)
    mask_seq = np.zeros((sequence_length, NUM_JOINTS), dtype=np.uint8)

    n_read = 0
    for slot, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        n_read += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(frame_rgb)
        coords, joint_mask = result_to_47pt(result, detector.backend)
        coords_seq[slot] = coords
        mask_seq[slot] = joint_mask
    cap.release()

    pose_cols = slice(42, 47)
    left_cols = slice(0, 21)
    right_cols = slice(21, 42)
    frac_pose = float(mask_seq[:, pose_cols].any(axis=1).mean()) if sequence_length else 0.0
    frac_left = float(mask_seq[:, left_cols].any(axis=1).mean()) if sequence_length else 0.0
    frac_right = float(mask_seq[:, right_cols].any(axis=1).mean()) if sequence_length else 0.0
    frac_hand = (
        float((mask_seq[:, left_cols].any(axis=1) | mask_seq[:, right_cols].any(axis=1)).mean())
        if sequence_length
        else 0.0
    )

    features = coords_seq.transpose(2, 0, 1)  # (3, seq_len, 47)
    stats = {
        "n_frames": total_frames,
        "n_sampled_ok": n_read,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frac_frames_with_pose": frac_pose,
        "frac_frames_with_left_hand": frac_left,
        "frac_frames_with_right_hand": frac_right,
        "frac_frames_with_any_hand": frac_hand,
    }
    return features, mask_seq, stats


# ---------------------------------------------------------------------------
# Cache / resumability
# ---------------------------------------------------------------------------


def cache_path_for(item: VideoItem, cache_dir: Path) -> Path:
    stem = Path(item.video_path).stem
    return cache_dir / f"{item.class_id:03d}_{item.signer_id}_{stem}.npz"


def load_or_extract(
    item: VideoItem, detector: Optional[Detector], cache_dir: Path, sequence_length: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    cache_file = cache_path_for(item, cache_dir)
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=True) as data:
            stats = data["stats"].item()
            return data["features"], data["mask"], stats

    assert detector is not None, "detector required when cache miss occurs"
    features, mask, stats = extract_single_video(item.video_path, detector, sequence_length)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file, features=features, mask=mask, stats=np.array(stats, dtype=object)
    )
    return features, mask, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=str, help="Root dir: <src>/<class_id>/<NN>_<class_id>.mp4")
    p.add_argument("--out", type=str, help="Output dir for features.npy/mask.npy/manifest.csv")
    p.add_argument("--backend", choices=["holistic", "hand_pose"], default="holistic")
    p.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--model-dir", type=str, default="./mediapipe_models")
    p.add_argument("--holistic-model", type=str, default=None, help="Path to holistic_landmarker.task; downloaded if omitted")
    p.add_argument("--hand-model", type=str, default=None, help="Path to hand_landmarker.task; downloaded if omitted")
    p.add_argument("--pose-model", type=str, default=None, help="Path to pose_landmarker.task; downloaded if omitted")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N discovered videos (debugging)")
    p.add_argument("--strict", action="store_true", help="Raise instead of logging+skipping on class-id/filename mismatches")
    p.add_argument("--no-resume", action="store_true", help="Ignore cache/ and re-extract every video")
    p.add_argument("--device-check", action="store_true", help="Print platform/mediapipe info and exit (no processing)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.device_check:
        print_device_check()
        return 0

    if not args.src or not args.out:
        print("--src and --out are required (unless using --device-check)", file=sys.stderr)
        return 2

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"

    items = discover_videos(src, strict=args.strict, mismatches_path=out / "mismatches.csv")
    if args.limit:
        items = items[: args.limit]
    print(f"Discovered {len(items)} videos under {src}")

    detector: Optional[Detector] = None

    all_features, all_masks, rows = [], [], []
    n_cached, n_extracted = 0, 0
    try:
        for item in items:
            cache_file = cache_path_for(item, cache_dir)
            if args.no_resume and cache_file.exists():
                cache_file.unlink()

            if not cache_file.exists() and detector is None:
                detector = Detector(args.backend, args)

            was_cached = cache_file.exists()
            features, mask, stats = load_or_extract(item, detector, cache_dir, args.sequence_length)
            n_cached += int(was_cached)
            n_extracted += int(not was_cached)

            all_features.append(features)
            all_masks.append(mask)
            rows.append(
                {
                    "idx": item.idx,
                    "video_path": item.video_path,
                    "class_id": item.class_id,
                    "signer_id": item.signer_id,
                    "n_frames": stats["n_frames"],
                    "n_sampled_ok": stats["n_sampled_ok"],
                    "frac_frames_with_pose": stats["frac_frames_with_pose"],
                    "frac_frames_with_left_hand": stats["frac_frames_with_left_hand"],
                    "frac_frames_with_right_hand": stats["frac_frames_with_right_hand"],
                    "frac_frames_with_any_hand": stats["frac_frames_with_any_hand"],
                }
            )

            if (n_cached + n_extracted) % 25 == 0:
                print(f"  ...{n_cached + n_extracted}/{len(items)} ({n_cached} cached, {n_extracted} extracted)")
    finally:
        if detector is not None:
            detector.close()

    if not rows:
        print("No videos matched the expected <NN>_<class_id>.mp4 pattern.", file=sys.stderr)
        return 1

    features_arr = np.stack(all_features).astype(np.float32)
    mask_arr = np.stack(all_masks).astype(np.uint8)
    np.save(out / "features.npy", features_arr)
    np.save(out / "mask.npy", mask_arr)

    manifest_path = out / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "video_path",
                "class_id",
                "signer_id",
                "n_frames",
                "n_sampled_ok",
                "frac_frames_with_pose",
                "frac_frames_with_left_hand",
                "frac_frames_with_right_hand",
                "frac_frames_with_any_hand",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(
        f"Done. {len(rows)} videos ({n_cached} from cache, {n_extracted} newly extracted). "
        f"features.npy {features_arr.shape}, mask.npy {mask_arr.shape} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
