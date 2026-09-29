"""Tests for scripts/make_demo_clips.py. Run with:
    python -m unittest test_make_demo_clips -v
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("make_demo_clips", REPO / "scripts" / "make_demo_clips.py")
mdc = importlib.util.module_from_spec(spec)
sys.modules["make_demo_clips"] = mdc  # dataclasses needs the module registered
spec.loader.exec_module(mdc)


def mask_with_hands(active_samples, n=32):
    m = np.zeros((n, 47), dtype=np.uint8)
    m[:, 43:45] = 1  # shoulders always visible: must NOT count as "active"
    for k in active_samples:
        m[k, 5] = 1
    return m


class TestActiveWindow(unittest.TestCase):
    def test_trims_idle_start_and_end_with_context(self):
        n, fps = 96, 30.0  # 3.2 s; samples every ~3 frames
        start, end = mdc.active_window(mask_with_hands(range(10, 20)), n, fps, pad_seconds=0.25, min_seconds=0.5)
        idx = mdc.sampled_frame_indices(n)
        self.assertAlmostEqual(start, idx[9] / fps - 0.25, places=6)
        self.assertAlmostEqual(end, (idx[20] + 1) / fps + 0.25, places=6)
        self.assertGreater(start, 0.5)
        self.assertLess(end, 3.2 - 0.5)

    def test_shoulders_alone_do_not_count_as_signing(self):
        start, end = mdc.active_window(mask_with_hands([]), 90, 30.0)
        self.assertEqual((start, end), (0.0, 3.0))  # no hands ever -> whole clip

    def test_clamped_to_clip_and_min_duration(self):
        start, end = mdc.active_window(mask_with_hands([0, 31]), 60, 30.0)
        self.assertEqual((start, end), (0.0, 2.0))
        start, end = mdc.active_window(mask_with_hands([15]), 90, 30.0, pad_seconds=0.0, min_seconds=1.5)
        self.assertAlmostEqual(end - start, 1.5, places=6)
        self.assertGreaterEqual(start, 0.0)


class TestSelection(unittest.TestCase):
    def setUp(self):
        # class 7: sample 0/1 same signer 'A' (best), 2 signer 'B', 3 signer 'C' but wrong in one run, 4 low hands
        self.manifest = pd.DataFrame({
            "signer_id": ["A", "A", "B", "C", "D"],
            "video_path": [f"/x/07/{s}_07.MP4" for s in "AABCD"],
            "n_frames": 60,
            "frac_frames_with_any_hand": [0.9, 0.9, 0.8, 0.9, 0.2],
        })
        rows = []
        for sample, probs, correct in [
            (0, [0.99, 0.99], [True, True]), (1, [0.97, 0.97], [True, True]),
            (2, [0.90, 0.80], [True, True]), (3, [0.99, 0.30], [True, False]), (4, [1.0, 1.0], [True, True]),
        ]:
            for run, (p, c) in enumerate(zip(probs, correct)):
                rows.append({"run": f"r{run}", "sample_idx": sample, "target_class_id": 7, "target_prob": p, "correct": c})
        self.pred = pd.DataFrame(rows)

    def test_ranking_prefers_all_correct_hands_visible_then_confidence(self):
        cands = mdc.rank_candidates(self.pred, self.manifest, 7)
        self.assertEqual(list(cands["sample_idx"]), [0, 1, 2, 4, 3])
        self.assertTrue(cands.iloc[0]["all_correct"])

    def test_pick_two_uses_different_signers(self):
        picks = mdc.pick_two(mdc.rank_candidates(self.pred, self.manifest, 7))
        self.assertEqual([int(p["sample_idx"]) for p in picks], [0, 2])
        self.assertEqual(len({p["signer_id"] for p in picks}), 2)

    def test_same_signer_fallback_only_when_needed(self):
        only_a = mdc.rank_candidates(self.pred, self.manifest, 7)
        only_a = only_a[only_a["signer_id"] == "A"]
        picks = mdc.pick_two(only_a)
        self.assertEqual(len(picks), 2)
        self.assertEqual({p["signer_id"] for p in picks}, {"A"})


class TestPathsAndVocab(unittest.TestCase):
    def test_remap_raw_path(self):
        p = mdc.remap_raw_path(Path("/data/raw"), "/content/drive/MyDrive/KSL_Project/KSL_Raw_Videos/07/03_07.MP4")
        self.assertEqual(p, Path("/data/raw/07/03_07.MP4"))

    def test_real_vocab_covers_67_classes(self):
        vocab = mdc.load_vocab(REPO / "src/frontend/src/data/vocab.json")
        self.assertEqual(len(vocab), 67)
        self.assertEqual(vocab[24], "nice")
        labels = json.loads((REPO / "src/frontend/public/models/ksl_f/label_map.json").read_text())["label_map"]
        self.assertEqual(set(vocab), {int(k) for k in labels})

    def test_words_with_clips_needs_both(self):
        with tempfile.TemporaryDirectory() as t:
            for n in ["a_example1.mp4", "a_example2.mp4", "b_example1.mp4", "c_example2.mp4"]:
                (Path(t) / n).touch()
            self.assertEqual(mdc.words_with_clips(Path(t)), {"a"})

    def test_default_targets_are_exactly_the_classes_without_clips(self):
        vocab = mdc.load_vocab(REPO / "src/frontend/src/data/vocab.json")
        with tempfile.TemporaryDirectory() as videos:
            # every word has clips except "what" (class 2) and "hobby" (class 6)
            for cid, w in vocab.items():
                if cid not in (2, 6):
                    for n in (1, 2):
                        (Path(videos) / f"{w}_example{n}.mp4").touch()
            args = Namespace(classes=None, all=False, existing_videos=videos, raw_videos="/raw", pad=0.25, min_seconds=1.0)
            manifest = pd.DataFrame({"signer_id": ["A", "B"], "video_path": ["/x/02/00_02.MP4", "/x/02/01_02.MP4"],
                                     "n_frames": 60, "frac_frames_with_any_hand": 0.9})
            pred = pd.DataFrame({"run": "r0", "sample_idx": [0, 1], "target_class_id": 2, "target_prob": 0.9, "correct": True})
            mask = np.stack([mask_with_hands(range(5, 25))] * 2)
            plans, warnings = mdc.build_plans(args, vocab, manifest, mask, pred, lambda s, r: 30.0)
        # only class 2 has predictions here; class 6 is targeted but skipped with a warning
        self.assertEqual({p.class_id for p in plans}, {2})
        self.assertEqual(len(warnings), 1)
        self.assertIn("hobby", warnings[0])
        self.assertEqual([p.example for p in plans], [1, 2])
        self.assertTrue(all(p.word == "what" for p in plans))

    def test_nothing_to_do_when_every_class_has_clips(self):
        vocab = mdc.load_vocab(REPO / "src/frontend/src/data/vocab.json")
        with tempfile.TemporaryDirectory() as videos:
            for w in vocab.values():
                for n in (1, 2):
                    (Path(videos) / f"{w}_example{n}.mp4").touch()
            args = Namespace(classes=None, all=False, existing_videos=videos, raw_videos="/raw", pad=0.25, min_seconds=1.0)
            plans, warnings = mdc.build_plans(args, vocab, pd.DataFrame(), np.zeros((0, 32, 47)), pd.DataFrame(), lambda s, r: 30.0)
        self.assertEqual((plans, warnings), ([], []))


def signer_table(**overrides):
    base = {f"{i:02d}": {"presentation": "M" if i % 2 == 0 else "F", "dominant_hand": "R", "exclude": False, "notes": ""} for i in range(20)}
    for k, v in overrides.items():
        base[k.lstrip("s")] = {**base[k.lstrip("s")], **v}
    return base


class TestSignerTable(unittest.TestCase):
    def write(self, text):
        d = tempfile.mkdtemp()
        p = Path(d) / "signers.csv"
        p.write_text(text)
        return p

    def test_load_signers_normalizes_ids_presentation_and_exclude(self):
        p = self.write("signer_id,presentation,dominant_hand,exclude,notes\n0,m,R,0,\n08,unknown,L,1,lefty\n9,F,r,0,\n")
        s = mdc.load_signers(p)
        self.assertEqual(s["00"]["presentation"], "M")
        self.assertTrue(s["08"]["exclude"])
        self.assertEqual(s["08"]["dominant_hand"], "L")
        self.assertEqual(s["09"], {"presentation": "F", "dominant_hand": "R", "exclude": False, "notes": ""})

    def test_load_signers_rejects_bad_presentation_and_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "presentation"):
            mdc.load_signers(self.write("signer_id,presentation,exclude\n00,X,0\n"))
        with self.assertRaisesRegex(ValueError, "missing column"):
            mdc.load_signers(self.write("signer_id,notes\n00,\n"))

    def test_template_from_contact_sheet_script_round_trips(self):
        spec2 = importlib.util.spec_from_file_location("signer_contact_sheet", REPO / "scripts" / "signer_contact_sheet.py")
        sys.path.insert(0, str(REPO / "scripts"))
        sc = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(sc)
        d = Path(tempfile.mkdtemp()) / "signers.csv"
        sc.write_template(d, [f"{i:02d}" for i in range(20)])
        s = mdc.load_signers(d)
        self.assertEqual(len(s), 20)
        self.assertTrue(s["08"]["exclude"] and s["08"]["dominant_hand"] == "L")
        self.assertTrue(all(not v["exclude"] and v["dominant_hand"] == "R" for k, v in s.items() if k != "08"))
        self.assertTrue(all(v["presentation"] == "UNKNOWN" for v in s.values()))


class TestBalanced(unittest.TestCase):
    def setUp(self):
        # class 7; signers: 00 M, 01 F, 02 M, 03 F, 08 (excluded, best of all)
        ids = ["08", "00", "01", "02", "03"]
        self.manifest = pd.DataFrame({
            "signer_id": ids, "video_path": [f"/x/07/{s}_07.MP4" for s in ids],
            "n_frames": 60, "frac_frames_with_any_hand": 0.9,
        })
        probs = {0: 0.999, 1: 0.90, 2: 0.99, 3: 0.95, 4: 0.97}  # sample_idx -> p
        self.pred = pd.DataFrame([{"run": "r0", "sample_idx": i, "target_class_id": 7, "target_prob": p, "correct": True}
                                  for i, p in probs.items()])
        self.signers = signer_table(s08={"presentation": "M", "exclude": True})

    def test_excluded_signers_are_never_candidates(self):
        c = mdc.rank_candidates(self.pred, self.manifest, 7, self.signers)
        self.assertNotIn("08", set(c["signer_id"]))
        self.assertEqual(list(c["sample_idx"]), [2, 4, 3, 1])
        self.assertEqual(list(c["presentation"]), ["F", "F", "M", "M"])  # 01 .99, 03 .97, 02 .95, 00 .90

    def test_balanced_picks_best_male_and_best_female_ordered_best_first(self):
        c = mdc.rank_candidates(self.pred, self.manifest, 7, self.signers)
        picks, note = mdc.pick_balanced(c)
        self.assertIsNone(note)
        # best F is 01 (0.99); best M is 02 (0.95) even though F 03 (0.97) outranks it
        self.assertEqual([p["signer_id"] for p in picks], ["01", "02"])
        self.assertEqual({p["presentation"] for p in picks}, {"M", "F"})

    def test_balanced_falls_back_to_best_two_when_a_presentation_is_missing(self):
        signers = signer_table(**{f"s{i:02d}": {"presentation": "M"} for i in range(0, 20)})  # nobody is F
        signers["08"]["exclude"] = True
        c = mdc.rank_candidates(self.pred, self.manifest, 7, signers)
        picks, note = mdc.pick_balanced(c)
        self.assertEqual(len(picks), 2)
        self.assertIn("no valid F clip", note)
        self.assertEqual([int(p["sample_idx"]) for p in picks], [2, 4])

    def test_build_plans_reports_presentation_and_fallbacks(self):
        vocab = {7: "me", 8: "you"}
        # class 8 has only male candidates
        manifest = pd.concat([self.manifest, pd.DataFrame({
            "signer_id": ["00", "02"], "video_path": ["/x/08/00_08.MP4", "/x/08/02_08.MP4"], "n_frames": 60,
            "frac_frames_with_any_hand": 0.9})], ignore_index=True)
        pred = pd.concat([self.pred, pd.DataFrame([{"run": "r0", "sample_idx": i, "target_class_id": 8, "target_prob": 0.9, "correct": True} for i in (5, 6)])])
        mask = np.stack([mask_with_hands(range(5, 25))] * len(manifest))
        args = Namespace(classes=None, all=True, existing_videos="/none", raw_videos="/raw", pad=0.5, min_seconds=1.0, balance_presentation=True)
        plans, _ = mdc.build_plans(args, vocab, manifest, mask, pred, lambda s, r: 30.0, self.signers)
        by_word = {}
        for pl in plans:
            by_word.setdefault(pl.word, []).append(pl)
        self.assertEqual({p.presentation for p in by_word["me"]}, {"M", "F"})
        self.assertTrue(all(p.fallback is None for p in by_word["me"]))
        self.assertTrue(all(p.fallback and "F" in p.fallback for p in by_word["you"]))


class TestOutputNaming(unittest.TestCase):
    def test_all_mode_names_match_exactly_what_the_app_expects(self):
        """--all writes <english>_example{1,2}.mp4 for every vocab word: identical to the clips the app loads
        today (incl. 'how many', 'do effort', 'Seoul', and words whose class label differs, e.g. please/worried/nice)."""
        vocab = mdc.load_vocab(REPO / "src/frontend/src/data/vocab.json")
        signers = signer_table()
        rows, preds = [], []
        for cid in vocab:
            for k, sid in enumerate(["00", "01", "02"]):
                rows.append({"signer_id": sid, "video_path": f"/x/{cid:02d}/{sid}_{cid:02d}.MP4", "n_frames": 60, "frac_frames_with_any_hand": 0.9})
                preds.append({"run": "r0", "sample_idx": len(rows) - 1, "target_class_id": cid, "target_prob": 0.9 - 0.1 * k, "correct": True})
        manifest, pred = pd.DataFrame(rows), pd.DataFrame(preds)
        mask = np.stack([mask_with_hands(range(5, 25))] * len(manifest))
        args = Namespace(classes=None, all=True, existing_videos="/none", raw_videos="/raw", pad=0.5, min_seconds=1.0, balance_presentation=True)
        plans, warnings = mdc.build_plans(args, vocab, manifest, mask, pred, lambda s, r: 30.0, signers)
        names = {f"{p.word}_example{p.example}.mp4" for p in plans}
        on_disk = {f.name for f in (REPO / "src/frontend/public/videos").glob("*.mp4")}
        self.assertEqual(names, on_disk)
        self.assertEqual(len(plans), 134)
        self.assertEqual(warnings, [])
        for n in ("how many_example1.mp4", "do effort_example2.mp4", "Seoul_example1.mp4", "please_example1.mp4", "worried_example2.mp4", "nice_example1.mp4"):
            self.assertIn(n, names)
        self.assertTrue(all(n.endswith(".mp4") and n == n[:-4] + ".mp4" for n in names))


if __name__ == "__main__":
    unittest.main()
