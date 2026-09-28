"""Unit tests for train.py's --split-mode full (+ --stop-epoch) and
export_onnx.py's PyTorch-vs-onnxruntime parity check. Uses tiny synthetic
data so this runs in a few seconds, not real KSL-77 data.

Run with:
    python -m unittest test_train_full_and_export.py -v
"""
import json
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_onnx
import train
from models import NUM_JOINTS


def make_tiny_legacy_pkl(path: Path, n_classes: int = 4, per_class: int = 15, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = n_classes * per_class
    labels = np.repeat(np.arange(1, n_classes + 1), per_class)
    features = np.zeros((n, 3, 32, NUM_JOINTS), dtype=np.float32)
    for i, lbl in enumerate(labels):
        features[i] = lbl * 0.05 + rng.normal(0, 0.01, size=(3, 32, NUM_JOINTS))
    with open(path, "wb") as f:
        pickle.dump((features, labels), f)


class TestSplitModeFull(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pkl_path = self.tmp / "tiny.pkl"
        make_tiny_legacy_pkl(self.pkl_path)
        self.out_dir = self.tmp / "run_full"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_requires_stop_epoch(self):
        rc = train.main(
            ["--pkl", str(self.pkl_path), "--split-mode", "full", "--out-dir", str(self.out_dir)]
        )
        self.assertEqual(rc, 2)
        self.assertFalse((self.out_dir / "fold_0" / "model.pt").exists())

    def test_stops_at_stop_epoch_and_scheduler_t_max_matches_epochs(self):
        full_epochs = 10
        stop_epoch = 2
        rc = train.main(
            [
                "--pkl", str(self.pkl_path),
                "--split-mode", "full",
                "--stop-epoch", str(stop_epoch),
                "--epochs", str(full_epochs),
                "--batch-size", "4",
                "--out-dir", str(self.out_dir),
            ]
        )
        self.assertEqual(rc, 0)

        fold_dir = self.out_dir / "fold_0"
        self.assertTrue((fold_dir / "model.pt").exists())
        self.assertTrue((fold_dir / "label_map.json").exists())

        with open(fold_dir / "config.json") as f:
            config = json.load(f)

        # Trained exactly --stop-epoch epochs...
        self.assertEqual(config["epochs_run"], stop_epoch)
        self.assertEqual(config["stop_epoch"], stop_epoch)
        # ...but the scheduler was built for the full --epochs, not stop_epoch
        # -- i.e. --stop-epoch changes when we stop, not the LR schedule's shape.
        self.assertEqual(config["scheduler_t_max"], full_epochs)
        self.assertNotEqual(config["scheduler_t_max"], config["stop_epoch"])

        # "full" mode has no validation/test split and no early-stopping concept.
        self.assertEqual(config["split_mode"], "full")
        self.assertIsNone(config["best_val_acc"])
        self.assertIsNone(config["best_epoch"])
        self.assertEqual(config["n_val"], 0)
        self.assertEqual(config["n_test"], 0)
        self.assertEqual(config["n_train"], 60)  # 4 classes * 15 samples

        # Checkpoint loads back cleanly with the recorded hyperparameters.
        from models import PoseCNN_LSTM_Attn

        with open(fold_dir / "label_map.json") as f:
            label_map = json.load(f)["label_map"]
        model = PoseCNN_LSTM_Attn(
            num_classes=len(label_map),
            cnn_hidden=config["cnn_hidden"],
            lstm_hidden=config["lstm_hidden"],
            dropout_rate=config["dropout"],
        )
        state = torch.load(fold_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state)  # raises on any shape mismatch

    def test_best_epoch_recorded_for_cv_modes(self):
        # Not "full" mode -- confirms the new best_epoch field also lands
        # correctly for the pre-existing random/signer_kfold paths.
        out_dir = self.tmp / "run_random"
        rc = train.main(
            [
                "--pkl", str(self.pkl_path),
                "--split-mode", "random",
                "--epochs", "3",
                "--patience", "0",
                "--batch-size", "4",
                "--out-dir", str(out_dir),
            ]
        )
        self.assertEqual(rc, 0)
        with open(out_dir / "fold_0" / "config.json") as f:
            config = json.load(f)
        self.assertIn("best_epoch", config)
        self.assertIsInstance(config["best_epoch"], int)
        self.assertGreaterEqual(config["best_epoch"], 0)
        self.assertLess(config["best_epoch"], config["epochs_run"])


class TestExportOnnxParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("onnxruntime not installed")

        cls.tmp = Path(tempfile.mkdtemp())
        cls.pkl_path = cls.tmp / "tiny.pkl"
        make_tiny_legacy_pkl(cls.pkl_path)
        cls.out_dir = cls.tmp / "run_full"
        rc = train.main(
            [
                "--pkl", str(cls.pkl_path),
                "--split-mode", "full",
                "--stop-epoch", "1",
                "--epochs", "5",
                "--batch-size", "4",
                "--out-dir", str(cls.out_dir),
            ]
        )
        assert rc == 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_export_and_parity_check_pass(self):
        onnx_path = self.out_dir / "fold_0" / "model.onnx"
        rc = export_onnx.main(["--checkpoint-dir", str(self.out_dir), "--out", str(onnx_path)])
        self.assertEqual(rc, 0)
        self.assertTrue(onnx_path.exists())

        # Also exercise parity_check() directly (not just via main()'s exit code).
        model, _config, _maps = export_onnx.load_checkpoint(self.out_dir / "fold_0")
        ok = export_onnx.parity_check(model, onnx_path)
        self.assertTrue(ok)

    def test_resolve_fold_dir_ambiguous_raises(self):
        multi_dir = self.tmp / "multi"
        (multi_dir / "fold_0").mkdir(parents=True)
        (multi_dir / "fold_1").mkdir(parents=True)
        with self.assertRaises(ValueError):
            export_onnx.resolve_fold_dir(multi_dir)


if __name__ == "__main__":
    unittest.main()
