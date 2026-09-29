"""Tests for scripts/calibrate_thresholds.py. Run with:
    python -m unittest test_calibrate_thresholds -v
The grading cases mirror src/frontend/src/lib/inference/inference.test.ts so the
Python simulation and the browser grading stay in lockstep.
"""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("calibrate_thresholds", REPO / "scripts" / "calibrate_thresholds.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)

K = 10


def P(entries):
    p = np.zeros(K)
    rest = K - len(entries)
    p[:] = (1 - sum(entries.values())) / rest
    for d, v in entries.items():
        p[d] = v
    return p


def grade(p, target=1, c=0.4, cl=0.15, cf=0.6):
    return cal.grade_attempt(p, target, c, cl, cf)


class TestGradeAttempt(unittest.TestCase):
    def test_correct_needs_top1_and_confidence(self):
        self.assertEqual(grade(P({1: 0.4})), "correct")
        self.assertEqual(grade(P({1: 0.39})), "close")  # top-1 but under CORRECT_MIN

    def test_close(self):
        self.assertEqual(grade(P({2: 0.5, 3: 0.2, 1: 0.15})), "close")
        self.assertNotEqual(grade(P({2: 0.5, 3: 0.3, 1: 0.14})), "close")  # too improbable
        self.assertNotEqual(grade(P({2: 0.3, 3: 0.25, 4: 0.2, 1: 0.16})), "close")  # outside top-3

    def test_confused_and_incorrect(self):
        self.assertEqual(grade(P({2: 0.6, 1: 0.05})), "confused")
        self.assertEqual(grade(P({2: 0.59, 1: 0.05})), "incorrect")
        self.assertEqual(grade(P({2: 0.3, 3: 0.2, 4: 0.15, 1: 0.01})), "incorrect")

    def test_close_checked_before_confused(self):
        self.assertEqual(grade(P({2: 0.7, 1: 0.2})), "close")


class TestSweeps(unittest.TestCase):
    def setUp(self):
        # 4 clips over K=10: two correct (p=0.9, 0.3), one wrong confident (0.8), one wrong unsure (0.3).
        self.y = np.array([0, 1, 2, 3])
        self.probs = np.stack([P({0: 0.9}), P({1: 0.3}), P({5: 0.8, 2: 0.05}), P({6: 0.3, 3: 0.2})])

    def test_correct_min(self):
        rows = {r["t"]: r for r in cal.sweep_correct_min(self.probs, self.y, np.array([0.2, 0.5]))}
        self.assertAlmostEqual(rows[0.2]["correct_accepted"], 1.0)
        self.assertAlmostEqual(rows[0.5]["correct_accepted"], 0.5)
        # wrong clips with p(top1) >= 0.5: just the 0.8 one -> 1 / (4 * 9)
        self.assertAlmostEqual(rows[0.5]["false_accept"], 1 / 36)

    def test_confusion_min(self):
        rows = {r["t"]: r for r in cal.sweep_confusion_min(self.probs, self.y, np.array([0.25, 0.75]))}
        # t=0.25: named clips are all four (0.9, 0.3, 0.8, 0.3); 2 wrong of 2 wrong shown
        self.assertAlmostEqual(rows[0.25]["wrong_shown"], 1.0)
        self.assertAlmostEqual(rows[0.25]["naming_precision"], 0.5)
        self.assertAlmostEqual(rows[0.25]["coverage"], 1.0)
        # t=0.75: named = {0.9 (correct), 0.8 (wrong)}
        self.assertAlmostEqual(rows[0.75]["wrong_shown"], 0.5)
        self.assertAlmostEqual(rows[0.75]["naming_precision"], 0.5)
        self.assertAlmostEqual(rows[0.75]["wrong_shown_share_of_all"], 0.25)

    def test_close_min(self):
        rows = {r["t"]: r for r in cal.sweep_close_min(self.probs, self.y, np.array([0.1, 0.25]))}
        # near-misses (target in top-3, not top-1): clip 2 (p=0.05) and clip 3 (p=0.2)
        self.assertAlmostEqual(rows[0.1]["close_kept"], 0.5)  # only clip 3 clears 0.1
        self.assertAlmostEqual(rows[0.25]["close_kept"], 0.0)
        self.assertGreaterEqual(rows[0.1]["false_close"], rows[0.25]["false_close"])

    def test_suggest_respects_rules(self):
        grid = cal.threshold_grid()
        s = cal.suggest(
            cal.sweep_correct_min(self.probs, self.y, grid),
            cal.sweep_close_min(self.probs, self.y, grid),
            cal.sweep_confusion_min(self.probs, self.y, grid),
            max_false_accept=0.02, max_false_close=0.5, min_naming_precision=0.5,
        )
        # false accepts (wrong clips with p(top1) >= t, / 36 attempts): 0.8 clip alone = 2.8%, 0.3 clip = 2.8%;
        # both wrong clips clear t<=0.3, only the 0.8 one clears t<=0.8, none clear t>0.8.
        self.assertGreater(s["CORRECT_MIN"], 0.8)
        self.assertTrue(all(k in s for k in ("CORRECT_MIN", "CLOSE_MIN", "CONFUSION_MIN")))


try:
    import torch  # noqa: F401
    import sklearn  # noqa: F401
    import pandas  # noqa: F401
    HAVE_STACK = True
except ImportError:  # pragma: no cover
    HAVE_STACK = False


@unittest.skipUnless(HAVE_STACK, "needs torch, scikit-learn and pandas")
class TestEndToEnd(unittest.TestCase):
    """Synthetic 20-signer dataset + randomly initialised checkpoints -> predictions.csv / probs.npz."""

    def test_end_to_end(self):
        import pandas as pd
        import torch

        import transforms
        from dataset import signer_kfold_splits
        from evaluate import evaluate_split, predict_probs
        from models import PoseCNN_LSTM_Attn

        rng = np.random.default_rng(0)
        n_signers, n_classes, seed = 20, 6, 42
        rows = [(s, c) for s in range(n_signers) for c in range(1, n_classes + 1)]
        n = len(rows)
        features = rng.uniform(0.2, 0.8, size=(n, 3, 32, 47)).astype(np.float32)
        mask = np.ones((n, 32, 47), dtype=np.uint8)
        manifest = pd.DataFrame(
            {"idx": range(n), "video_path": [f"v{i}" for i in range(n)], "class_id": [c for _, c in rows],
             "signer_id": [f"{s:02d}" for s, _ in rows], "n_frames": 40, "n_sampled_ok": 32,
             "frac_frames_with_pose": 1.0, "frac_frames_with_any_hand": 1.0}
        )
        labels = manifest["class_id"].to_numpy()
        signer_ids = manifest["signer_id"].to_numpy()

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "features.npy", features)
            np.save(tmp / "mask.npy", mask)
            manifest.to_csv(tmp / "manifest.csv", index=False)
            splits = signer_kfold_splits(labels, signer_ids, n_splits=5, seed=seed)
            run = tmp / "run"
            torch.manual_seed(0)
            for sp in splits:
                d = run / f"fold_{sp.fold}"
                d.mkdir(parents=True)
                model = PoseCNN_LSTM_Attn(num_classes=n_classes, cnn_hidden=16, lstm_hidden=16, dropout_rate=0.1)
                torch.save(model.state_dict(), d / "model.pt")
                (d / "config.json").write_text(json.dumps(
                    {"split_mode": "signer_kfold", "seed": seed, "cnn_hidden": 16, "lstm_hidden": 16, "dropout": 0.1,
                     "min_hand_frac": 0.0, "normalize_body": True, "trim_idle": False}))
                (d / "label_map.json").write_text(json.dumps(
                    {"label_map": {str(k): v for k, v in sp.label_map.items()},
                     "reverse_label_map": {str(k): v for k, v in sp.reverse_label_map.items()}}))
            out = tmp / "cal"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cal.main(["--features", str(tmp / "features.npy"), "--mask", str(tmp / "mask.npy"),
                               "--manifest", str(tmp / "manifest.csv"), "--checkpoint-dir", str(run),
                               "--tta-mirror", "--out-dir", str(out)])
            self.assertEqual(rc, 0)
            text = buf.getvalue()
            self.assertIn("Suggested thresholds", text)
            self.assertIn("CONFUSION_MIN", text)

            df = pd.read_csv(out / "predictions.csv")
            self.assertEqual(len(df), n)  # every clip is held out in exactly one fold
            self.assertEqual(df["sample_idx"].nunique(), n)
            for col in ("target_prob", "top1_word", "top1_prob", "correct", "signer_id", "fold", "target_rank"):
                self.assertIn(col, df.columns)
            z = np.load(out / "probs.npz")
            self.assertEqual(z["probs"].shape, (n, n_classes))
            np.testing.assert_allclose(z["probs"].sum(1), 1.0, atol=1e-5)

            # Cross-check against evaluate.evaluate_split (same probs path) for fold 0.
            feats = transforms.normalize_body_bulk(features, mask)
            sp = splits[0]
            model = PoseCNN_LSTM_Attn(num_classes=n_classes, cnn_hidden=16, lstm_hidden=16, dropout_rate=0.1)
            model.load_state_dict(torch.load(run / "fold_0" / "model.pt"))
            model.eval()
            ev = evaluate_split(model, feats, labels, sp, signer_ids, mask=mask, tta_mirror=True)
            fold0 = df[df["fold"] == 0]
            self.assertAlmostEqual(fold0["correct"].mean(), ev["top1"], places=6)

            # analysis-only path
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(cal.main(["--from-npz", str(out / "probs.npz")]), 0)
            self.assertIn("CORRECT_MIN", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
