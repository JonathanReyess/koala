"""Unit tests for transforms.py. Run with:
    python -m unittest test_transforms.py -v
(or `pytest test_transforms.py -v` if pytest is installed -- pytest can run
unittest.TestCase classes directly, no extra config needed.)
"""
import unittest

import numpy as np

from transforms import (
    LEFT_ELBOW,
    LEFT_HAND,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    mirror_clip,
    normalize_body,
    trim_idle,
)

NUM_JOINTS = 47
T = 32


def make_clip(rng: np.random.Generator, all_shoulders_detected: bool = True):
    coords = rng.uniform(0.0, 1.0, size=(T, NUM_JOINTS, 3)).astype(np.float32)
    mask = rng.integers(0, 2, size=(T, NUM_JOINTS)).astype(bool)
    if all_shoulders_detected:
        mask[:, LEFT_SHOULDER] = 1
        mask[:, RIGHT_SHOULDER] = 1
    return coords, mask


class TestMirror(unittest.TestCase):
    def test_mirror_twice_is_identity_all_shoulders_detected(self):
        rng = np.random.default_rng(0)
        coords, mask = make_clip(rng, all_shoulders_detected=True)
        c1, m1 = mirror_clip(coords, mask)
        c2, m2 = mirror_clip(c1, m1)
        np.testing.assert_allclose(c2, coords, atol=1e-6)
        np.testing.assert_array_equal(m2, mask)

    def test_mirror_twice_is_identity_partial_shoulder_detection(self):
        # Exercises the per-frame-fallback-to-clip-mean path.
        rng = np.random.default_rng(1)
        coords, mask = make_clip(rng, all_shoulders_detected=False)
        # Force at least one frame with both shoulders detected so the
        # "clip mean" fallback (not the 0.5 global fallback) is exercised.
        mask[0, LEFT_SHOULDER] = 1
        mask[0, RIGHT_SHOULDER] = 1
        c1, m1 = mirror_clip(coords, mask)
        c2, m2 = mirror_clip(c1, m1)
        np.testing.assert_allclose(c2, coords, atol=1e-6)
        np.testing.assert_array_equal(m2, mask)

    def test_mirror_twice_is_identity_no_shoulders_detected(self):
        # Exercises the global 0.5 fallback path.
        rng = np.random.default_rng(2)
        coords, mask = make_clip(rng, all_shoulders_detected=False)
        mask[:, LEFT_SHOULDER] = 0
        mask[:, RIGHT_SHOULDER] = 0
        c1, m1 = mirror_clip(coords, mask)
        c2, m2 = mirror_clip(c1, m1)
        np.testing.assert_allclose(c2, coords, atol=1e-6)
        np.testing.assert_array_equal(m2, mask)

    def test_mirror_swaps_hand_blocks_and_shoulder_elbow_pairs(self):
        coords = np.zeros((1, NUM_JOINTS, 3), dtype=np.float32)
        mask = np.zeros((1, NUM_JOINTS), dtype=bool)
        coords[0, LEFT_SHOULDER] = [0.4, 0.5, 0.0]
        coords[0, RIGHT_SHOULDER] = [0.6, 0.5, 0.0]
        mask[0, LEFT_SHOULDER] = 1
        mask[0, RIGHT_SHOULDER] = 1
        coords[0, LEFT_WRIST] = [0.3, 0.2, 0.0]  # left hand's wrist (joint 0)
        mask[0, LEFT_HAND] = 1
        coords[0, LEFT_ELBOW] = [0.35, 0.6, 0.0]
        mask[0, LEFT_ELBOW] = 1

        mirrored_coords, mirrored_mask = mirror_clip(coords, mask)

        # mid_shoulder_x = 0.5; left hand's wrist (0.3) reflects to 0.7 and
        # should now live in the RIGHT hand slot.
        self.assertAlmostEqual(mirrored_coords[0, RIGHT_WRIST, 0], 0.7, places=5)
        self.assertEqual(mirrored_mask[0, RIGHT_WRIST], 1)
        self.assertEqual(mirrored_mask[0, LEFT_HAND].max(), 0)  # left hand slot now empty
        # shoulders swapped: reflected left-shoulder (0.6) now sits in the RIGHT_SHOULDER slot
        self.assertAlmostEqual(mirrored_coords[0, RIGHT_SHOULDER, 0], 0.6, places=5)
        self.assertAlmostEqual(mirrored_coords[0, LEFT_SHOULDER, 0], 0.4, places=5)
        # elbow moved from LEFT to RIGHT slot
        self.assertEqual(mirrored_mask[0, RIGHT_ELBOW], 1)
        self.assertEqual(mirrored_mask[0, LEFT_ELBOW], 0)


class TestNormalizeBody(unittest.TestCase):
    def test_shoulder_width_becomes_one(self):
        rng = np.random.default_rng(3)
        coords, mask = make_clip(rng, all_shoulders_detected=True)
        # Fixed, constant shoulder geometry every frame -> mean == that
        # constant -> normalized width should land on exactly 1.0.
        width = 0.18
        mid_x, mid_y = 0.5, 0.45
        coords[:, LEFT_SHOULDER] = [mid_x - width / 2, mid_y, 0.0]
        coords[:, RIGHT_SHOULDER] = [mid_x + width / 2, mid_y, 0.0]

        normalized = normalize_body(coords, mask)

        new_width = np.linalg.norm(
            normalized[:, LEFT_SHOULDER, :] - normalized[:, RIGHT_SHOULDER, :], axis=1
        )
        np.testing.assert_allclose(new_width, 1.0, atol=1e-5)

    def test_missing_joints_stay_zero(self):
        rng = np.random.default_rng(4)
        coords, mask = make_clip(rng, all_shoulders_detected=True)
        mask[:, LEFT_WRIST] = 0  # never detected
        normalized = normalize_body(coords, mask)
        np.testing.assert_array_equal(normalized[:, LEFT_WRIST, :], 0.0)

    def test_no_shoulders_detected_leaves_clip_unchanged(self):
        rng = np.random.default_rng(5)
        coords, mask = make_clip(rng, all_shoulders_detected=False)
        mask[:, LEFT_SHOULDER] = 0
        mask[:, RIGHT_SHOULDER] = 0
        normalized = normalize_body(coords, mask)
        np.testing.assert_array_equal(normalized, coords)


    def test_non_bool_mask_raises(self):
        """Regression: a uint8 mask used to make normalize_body silently use
        only frames 0/1 as reference frames (integer fancy-indexing); it (and
        mirror_clip) must now refuse non-bool masks."""
        rng = np.random.default_rng(7)
        coords, mask = make_clip(rng)
        for bad in (mask.astype(np.uint8), mask.astype(int), mask.astype(np.float32)):
            with self.assertRaisesRegex(TypeError, "mask must be a numpy bool array"):
                normalize_body(coords, bad)
            with self.assertRaisesRegex(TypeError, "mask must be a numpy bool array"):
                mirror_clip(coords, bad)
        normalize_body(coords, mask)  # bool is fine
        mirror_clip(coords, mask)

    def test_bool_mask_uses_all_shoulder_frames(self):
        """Reference frames must be *all* frames with both shoulders, not frames 0/1."""
        rng = np.random.default_rng(8)
        coords, mask = make_clip(rng)
        coords[:, LEFT_SHOULDER] += rng.normal(0, 0.05, size=(T, 3)).astype(np.float32)
        out = normalize_body(coords, mask)
        left = coords[:, LEFT_SHOULDER].astype(np.float64)
        right = coords[:, RIGHT_SHOULDER].astype(np.float64)
        width = np.linalg.norm(left - right, axis=1).mean()
        mid = ((left + right) / 2).mean(0)
        expected = (coords[:, LEFT_SHOULDER] - mid) / width
        np.testing.assert_allclose(out[:, LEFT_SHOULDER], expected, atol=1e-4)


class TestTrimIdle(unittest.TestCase):
    def test_recovers_active_window_from_idle_padded_clip(self):
        T_local = 32
        coords = np.zeros((T_local, NUM_JOINTS, 3), dtype=np.float32)
        mask = np.zeros((T_local, NUM_JOINTS), dtype=np.uint8)

        idle_start, idle_end = 6, 6  # 6 idle frames at the start and end
        active_start, active_end = idle_start, T_local - idle_end  # [6, 26)

        # Idle padding: hand detected but stationary at a fixed point.
        mask[:, LEFT_WRIST] = 1
        coords[:idle_start, LEFT_WRIST, :2] = [0.5, 0.5]
        coords[T_local - idle_end :, LEFT_WRIST, :2] = [0.5, 0.5]

        # Active window: hand sweeps left -> right, well above the motion threshold.
        n_active = active_end - active_start
        xs = np.linspace(0.1, 0.9, n_active)
        coords[active_start:active_end, LEFT_WRIST, 0] = xs
        coords[active_start:active_end, LEFT_WRIST, 1] = 0.5

        trimmed_coords, trimmed_mask = trim_idle(coords, mask, motion_threshold=0.02)

        # The trimmed+resampled clip's wrist-x trajectory should span
        # (close to) the active window's true range, not the idle 0.5 padding.
        trimmed_x = trimmed_coords[:, LEFT_WRIST, 0]
        self.assertLess(trimmed_x.min(), 0.15)
        self.assertGreater(trimmed_x.max(), 0.85)
        # And it should preserve the left->right sweep trend (idle padding
        # trimmed off, not averaged in). Checked via first-quarter vs.
        # last-quarter mean rather than strict np.diff monotonicity: the
        # motion detector legitimately may pull in one boundary frame where
        # the hand is decelerating back toward the idle position (motion
        # into *and* out of that single frame both exceed the threshold --
        # an inherent ambiguity of frame-difference motion detection, not a
        # bug), which can introduce one small local dip right at the edge.
        first_quarter = trimmed_x[: T_local // 4].mean()
        last_quarter = trimmed_x[-T_local // 4 :].mean()
        self.assertLess(first_quarter, 0.3)
        self.assertGreater(last_quarter, 0.7)
        self.assertEqual(trimmed_coords.shape, coords.shape)
        self.assertEqual(trimmed_mask.shape, mask.shape)

    def test_no_active_window_leaves_clip_unchanged(self):
        T_local = 16
        coords = np.zeros((T_local, NUM_JOINTS, 3), dtype=np.float32)
        mask = np.zeros((T_local, NUM_JOINTS), dtype=np.uint8)
        mask[:, LEFT_WRIST] = 1
        coords[:, LEFT_WRIST, :2] = 0.5  # perfectly static the whole clip

        trimmed_coords, trimmed_mask = trim_idle(coords, mask, motion_threshold=0.02)
        np.testing.assert_array_equal(trimmed_coords, coords)
        np.testing.assert_array_equal(trimmed_mask, mask)


if __name__ == "__main__":
    unittest.main()
