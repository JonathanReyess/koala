"""Shared, pure-numpy feature transforms for train.py/evaluate.py:
mirroring (left/right flip), body normalization, and idle-frame trimming.

Layout reminder (same 47-point scheme as extract_landmarks.py /
src/backend/app.py):
    0-20   left hand  (landmark 0 of this block = wrist)
    21-41  right hand (landmark 0 of this block = wrist)
    42     nose
    43     left shoulder
    44     right shoulder
    45     left elbow
    46     right elbow

All functions here operate on "frame-major" arrays -- coords as
(T, J, 3), mask as (T, J) -- since that's the natural shape for per-frame
reasoning (motion, per-frame mid-shoulder, etc). train.py/evaluate.py's
features arrays are "channel-major", (N, 3, T, J) (what the CNN wants), so
use `chw_to_tjc` / `tjc_to_chw` to convert one clip at a time.
"""
from __future__ import annotations

import numpy as np

LEFT_HAND = slice(0, 21)
RIGHT_HAND = slice(21, 42)
LEFT_WRIST = 0  # landmark 0 within the left-hand block
RIGHT_WRIST = 21  # landmark 0 within the right-hand block
NOSE = 42
LEFT_SHOULDER = 43
RIGHT_SHOULDER = 44
LEFT_ELBOW = 45
RIGHT_ELBOW = 46

DEFAULT_TRIM_MOTION_THRESHOLD = 0.02  # normalized units/frame; see trim_idle()


def chw_to_tjc(x: np.ndarray) -> np.ndarray:
    """(3, T, J) -> (T, J, 3)"""
    return np.transpose(x, (1, 2, 0))


def tjc_to_chw(x: np.ndarray) -> np.ndarray:
    """(T, J, 3) -> (3, T, J)"""
    return np.transpose(x, (2, 0, 1))


def infer_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    """coords: (..., 3) -> boolean mask (...): True where NOT exactly (0,0,0).

    Stand-in for a real per-joint mask when one isn't available -- e.g. the
    legacy .pkl (data/KSL77_joint_stream_47pt.pkl), which silently
    zero-fills missing joints instead of recording a mask at all. Works on
    both a single clip (T, J, 3) and a batch (N, T, J, 3).
    """
    return np.any(coords != 0, axis=-1)


def normalize_body_bulk(features: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """normalize_body applied per-clip across a whole (N, 3, T, J) dataset.
    `mask` is (N, T, J) if available, else inferred per-clip (see
    infer_mask_from_coords) -- e.g. for the legacy .pkl.
    """
    out = np.empty_like(features)
    for i in range(features.shape[0]):
        coords = chw_to_tjc(features[i])
        m = mask[i].astype(bool) if mask is not None else infer_mask_from_coords(coords)
        out[i] = tjc_to_chw(normalize_body(coords, m))
    return out


def trim_idle_bulk(
    features: np.ndarray,
    mask: np.ndarray | None = None,
    motion_threshold: float = DEFAULT_TRIM_MOTION_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray | None]:
    """trim_idle applied per-clip across a whole (N, 3, T, J) dataset."""
    out_features = np.empty_like(features)
    out_mask = np.empty_like(mask) if mask is not None else None
    for i in range(features.shape[0]):
        coords = chw_to_tjc(features[i])
        m = mask[i].astype(bool) if mask is not None else infer_mask_from_coords(coords)
        new_coords, new_mask = trim_idle(coords, m, motion_threshold=motion_threshold)
        out_features[i] = tjc_to_chw(new_coords)
        if out_mask is not None:
            out_mask[i] = new_mask.astype(mask.dtype)
    return out_features, out_mask


# ---------------------------------------------------------------------------
# Mirror (left/right flip)
# ---------------------------------------------------------------------------


def mirror_clip(coords: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left/right-flips one clip: reflects x through each frame's mid-shoulder
    x, then swaps the left/right hand blocks and the shoulder/elbow pairs (so
    "left hand" data ends up back in the left-hand slots, now describing what
    was the mirrored right hand, etc).

    coords: (T, J, 3) normalized xyz. mask: (T, J).
    Idempotent: mirror_clip(*mirror_clip(coords, mask)) == (coords, mask),
    because the per-frame mid-shoulder x used for the reflection is symmetric
    in the two shoulders (so it's unchanged by swapping them), and swapping
    a pair of slots twice is the identity.
    """
    T = coords.shape[0]
    mid_shoulder_x = np.where(
        mask[:, LEFT_SHOULDER] & mask[:, RIGHT_SHOULDER],
        (coords[:, LEFT_SHOULDER, 0] + coords[:, RIGHT_SHOULDER, 0]) / 2.0,
        np.nan,
    )
    valid = ~np.isnan(mid_shoulder_x)
    if valid.any():
        clip_fallback = float(np.mean(mid_shoulder_x[valid]))
    else:
        # No frame has both shoulders detected at all: reflect about the
        # frame's own horizontal center (coordinates are normalized to
        # [0, 1] of the frame), the least-assumption fallback.
        clip_fallback = 0.5
    mid_shoulder_x = np.where(valid, mid_shoulder_x, clip_fallback)

    mirrored_coords = coords.copy()
    mirrored_coords[:, :, 0] = 2.0 * mid_shoulder_x[:, None] - coords[:, :, 0]

    mirrored_mask = mask.copy()

    def swap(a, i, j):
        a[:, [i, j]] = a[:, [j, i]]

    # Hands (blocks of 21)
    left_c = mirrored_coords[:, LEFT_HAND, :].copy()
    right_c = mirrored_coords[:, RIGHT_HAND, :].copy()
    mirrored_coords[:, LEFT_HAND, :] = right_c
    mirrored_coords[:, RIGHT_HAND, :] = left_c
    left_m = mirrored_mask[:, LEFT_HAND].copy()
    right_m = mirrored_mask[:, RIGHT_HAND].copy()
    mirrored_mask[:, LEFT_HAND] = right_m
    mirrored_mask[:, RIGHT_HAND] = left_m

    # Shoulders / elbows (single joints)
    swap(mirrored_coords, LEFT_SHOULDER, RIGHT_SHOULDER)
    swap(mirrored_coords, LEFT_ELBOW, RIGHT_ELBOW)
    swap(mirrored_mask, LEFT_SHOULDER, RIGHT_SHOULDER)
    swap(mirrored_mask, LEFT_ELBOW, RIGHT_ELBOW)

    assert mirrored_coords.shape == (T,) + coords.shape[1:]
    return mirrored_coords, mirrored_mask


# ---------------------------------------------------------------------------
# Body normalization
# ---------------------------------------------------------------------------


def normalize_body(coords: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Subtracts the clip's mean mid-shoulder point (averaged over frames
    where both shoulders are detected) and divides x/y/z by the clip's mean
    shoulder width (same frames), so different signers/distances-from-camera
    land in a comparable coordinate frame. Missing joints (mask == 0) stay
    exactly 0 after normalizing (not "shifted" to -mid_shoulder), so a
    downstream "is this joint zero" check still means "missing", not "at the
    body center".

    If no frame has both shoulders detected, returns coords unchanged (can't
    define a body frame for this clip).
    """
    both_shoulders = mask[:, LEFT_SHOULDER] & mask[:, RIGHT_SHOULDER]
    if not both_shoulders.any():
        return coords.copy()

    left = coords[both_shoulders, LEFT_SHOULDER, :]
    right = coords[both_shoulders, RIGHT_SHOULDER, :]
    mid_shoulder = ((left + right) / 2.0).mean(axis=0)  # (3,)
    shoulder_width = float(np.linalg.norm(left - right, axis=1).mean())
    shoulder_width = max(shoulder_width, eps)

    normalized = (coords - mid_shoulder[None, None, :]) / shoulder_width
    normalized = np.where(mask[:, :, None].astype(bool), normalized, 0.0)
    return normalized.astype(coords.dtype)


# ---------------------------------------------------------------------------
# Idle-frame trimming
# ---------------------------------------------------------------------------


def _wrist_motion(coords: np.ndarray, mask: np.ndarray, wrist_idx: int) -> np.ndarray:
    """Frame-to-frame xy displacement of one wrist; 0 where either endpoint
    of the pair is undetected or at t=0 (no previous frame)."""
    T = coords.shape[0]
    motion = np.zeros(T, dtype=np.float32)
    both_detected = mask[1:, wrist_idx] & mask[:-1, wrist_idx]
    delta = coords[1:, wrist_idx, :2] - coords[:-1, wrist_idx, :2]
    dist = np.linalg.norm(delta, axis=1)
    motion[1:] = np.where(both_detected, dist, 0.0)
    return motion


def trim_idle(
    coords: np.ndarray,
    mask: np.ndarray,
    motion_threshold: float = DEFAULT_TRIM_MOTION_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Finds the first/last sampled frame where either hand is both detected
    and moving faster than `motion_threshold` (normalized units/frame,
    frame-to-frame wrist displacement), crops to that inclusive window, and
    resamples back to the original number of frames T: linear interpolation
    for coords, nearest-neighbor for mask (so mask values stay exactly 0/1).

    If no frame meets the "detected and moving" bar (e.g. a totally static
    or fully-undetected clip), returns the clip unchanged -- there's no
    active window to trim to.
    """
    T = coords.shape[0]
    left_motion = _wrist_motion(coords, mask, LEFT_WRIST)
    right_motion = _wrist_motion(coords, mask, RIGHT_WRIST)
    active = (
        (mask[:, LEFT_WRIST].astype(bool) & (left_motion > motion_threshold))
        | (mask[:, RIGHT_WRIST].astype(bool) & (right_motion > motion_threshold))
    )

    if not active.any():
        return coords.copy(), mask.copy()

    active_idx = np.where(active)[0]
    first, last = int(active_idx.min()), int(active_idx.max())
    if first == last:
        # Single active frame: nothing to resample a window from besides a
        # constant -- repeat it T times rather than dividing by a zero-length span.
        cropped_coords = np.repeat(coords[first : first + 1], T, axis=0)
        cropped_mask = np.repeat(mask[first : first + 1], T, axis=0)
        return cropped_coords, cropped_mask

    cropped_coords = coords[first : last + 1]
    cropped_mask = mask[first : last + 1]
    cropped_len = cropped_coords.shape[0]

    src_positions = np.arange(cropped_len, dtype=np.float64)
    dst_positions = np.linspace(0, cropped_len - 1, T)

    out_coords = np.empty((T,) + coords.shape[1:], dtype=coords.dtype)
    for j in range(coords.shape[1]):
        for c in range(coords.shape[2]):
            out_coords[:, j, c] = np.interp(dst_positions, src_positions, cropped_coords[:, j, c])

    nearest_idx = np.clip(np.round(dst_positions).astype(int), 0, cropped_len - 1)
    out_mask = cropped_mask[nearest_idx]

    return out_coords, out_mask
