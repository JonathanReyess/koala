# Results

Comparison of the model across three conditions. **Rows (b) and (c) have not
been run yet** — they require re-extracting landmarks from the raw KSL-77
videos (`extract_landmarks.py`), which live in Google Drive and aren't
reachable from this environment. Row (a) is real, reproduced from the
existing `src/backend/best_model.pt` checkpoint and the existing
`data/KSL77_joint_stream_47pt.pkl` feature file (both already in the repo)
using `dataset.py`'s `random_split` (seed 42) and `evaluate.py`.

**No number below is described as an improvement over another unless it is
verified on row (c).** Row (a) vs (b) alone cannot tell you anything about
generalization to a new signer, because (a)'s "signer-out" numbers do not
exist for the current checkpoint (see AUDIT.md §4) — the two runs would only
differ in feature-extraction code, not in whether the split leaks signers.

| | Features | Split | Top-1 | Top-3 | Macro-F1 | Notes |
|---|---|---|---|---|---|---|
| (a) Old checkpoint, random split | `data/KSL77_joint_stream_47pt.pkl` (legacy notebook extraction, silent zero-fill for missing landmarks) | `random` (stratified 80/10/10, seed 42, **not** signer-grouped) | **88.21%** | **95.53%** | **0.8806** | Reproduced directly from `src/backend/best_model.pt` via `dataset.random_split(seed=42)` + `evaluate.py`; matches `notebook/KSL.ipynb` Cell 9's own reported 88.21% exactly (see AUDIT.md §8). Test set is only 246 samples (3-4/class) — high variance. |
| (b) Retrained, new features, random split | `extract_landmarks.py` output (Tasks API, explicit per-joint mask) | `random` (same as (a), for apples-to-apples on features only) | *pending* | *pending* | *pending* | Not run — needs raw KSL-77 videos + a Colab GPU session (`python extract_landmarks.py --src .../KSL_Raw_Videos --out .../features_v2`, then `python train.py --features .../features_v2/features.npy --mask .../features_v2/mask.npy --manifest .../features_v2/manifest.csv --split-mode random --out-dir runs/b_random`). Comparing this to (a) only tells you whether the new extraction (mask-aware, Tasks API, no silent zero-fill) changes what the model learns — it does **not** tell you anything about signer generalization, since neither (a) nor (b) is signer-grouped. |
| (c) Retrained, new features, signer-out 5-fold | same as (b) | `signer_kfold` (GroupKFold by signer_id, 5 folds; see dataset.py) | *pending* (mean ± std across 5 folds) | *pending* | *pending* | Not run — `python train.py --features .../features_v2/features.npy --mask .../features_v2/mask.npy --manifest .../features_v2/manifest.csv --split-mode signer_kfold --out-dir runs/c_signer_kfold`, then `evaluate.py` on that dir. **This is the only row that answers "does the model work on a signer it has never seen," which is the actual open question from AUDIT.md §4.** Expect this number to be lower than (a)/(b) — that would not be a regression, it would be (a)/(b) turning out to have been optimistic. |

## How to fill in (b) and (c)

```bash
# On Colab, after `git clone` + `pip install -r requirements.txt`:
python extract_landmarks.py --device-check   # confirm mediapipe/platform first
python scripts/smoke_test_colab.py --videos-dir /content/drive/MyDrive/KSL_Project/KSL_Raw_Videos/77 --n 3
python extract_landmarks.py \
  --src /content/drive/MyDrive/KSL_Project/KSL_Raw_Videos \
  --out /content/drive/MyDrive/KSL_Project/features_v2

python train.py --features .../features_v2/features.npy --mask .../features_v2/mask.npy \
  --manifest .../features_v2/manifest.csv --split-mode random --out-dir runs/b_random
python evaluate.py --features .../features_v2/features.npy --mask .../features_v2/mask.npy \
  --manifest .../features_v2/manifest.csv --checkpoint-dir runs/b_random --out runs/b_random/eval.json

python train.py --features .../features_v2/features.npy --mask .../features_v2/mask.npy \
  --manifest .../features_v2/manifest.csv --split-mode signer_kfold --out-dir runs/c_signer_kfold
python evaluate.py --features .../features_v2/features.npy --mask .../features_v2/mask.npy \
  --manifest .../features_v2/manifest.csv --checkpoint-dir runs/c_signer_kfold --out runs/c_signer_kfold/eval.json
```

Then replace the *pending* cells above with `evaluate.py`'s printed
top1/top3/macro_f1 (mean ± std for row (c)), and paste `per_signer_accuracy_combined`
from `runs/c_signer_kfold/eval.json` as a follow-up table if any signer stands
out as much worse than the rest — that's the actual generalization signal
this whole exercise is for.

## Known real-world data issue: corrupted videos

The full Colab extraction run hit at least one raw video that fails to
decode at all ("moov atom not found" — a truncated/corrupted MP4).
`extract_landmarks.py` still writes a row for it (all-zero features,
`n_sampled_ok = 0`, `frac_frames_with_pose = 0`), since dropping it silently
at extraction time would make `manifest.csv` not account for every
discovered video. **`dataset.py` now drops these rows before any split**
(`drop_undetected_samples`, called from `load_manifest_dataset`), so a
corrupted file can never end up as a real training or test example. It
drops a row if `n_sampled_ok == 0`, `frac_frames_with_pose == 0`, or (opt-in
via `--min-hand-frac`, default `0.0` = off) `frac_frames_with_any_hand` is
below a threshold — printing which `video_path`s were dropped and why, and
filtering `features`/`mask`/`manifest` together so row alignment is never
broken. `manifest.csv` also now includes `n_sampled_ok`,
`frac_frames_with_left_hand`, and `frac_frames_with_right_hand` (previously
only computed internally, not written out) so this — and any future
threshold tuning — doesn't require re-running MediaPipe, just re-reading the
existing cache.

`train.py` records `min_hand_frac` in each fold's `config.json`; `evaluate.py`
reads it back automatically so it reconstructs the *exact* same filtered
dataset the checkpoint's split was built from, and refuses to run if you
pass a `--min-hand-frac` that doesn't match what training used (that would
silently evaluate against a different sample set than the one the split
indices actually correspond to).

## What's been verified so far (this environment, no raw videos/Drive access)

- Verified end-to-end with a synthetic manifest containing a deliberately
  corrupted row (`n_sampled_ok=0`) and a deliberately pose-less row
  (`frac_frames_with_pose=0`): both are dropped, `features`/`mask`/`manifest`
  stay aligned afterward, and `--min-hand-frac` correctly becomes an opt-in
  extra filter. `train.py` + `evaluate.py` were re-run end-to-end against
  this dataset too: training skips the corrupted row, and `evaluate.py`
  automatically reconstructs the identical filtered split from
  `config.json` and correctly refuses a mismatched `--min-hand-frac`
  override.
- `dataset.random_split(seed=42)` reproduces the notebook's own split and
  `src/backend/best_model.pt`'s exact reported 88.21% test accuracy — see
  row (a) above, and AUDIT.md §8.
- `models/pose_cnn_lstm_attn.py` (the fresh, canonical model definition)
  loads `src/backend/best_model.pt` without any shape mismatch and
  reproduces the same forward pass — confirming it (not the deleted
  `models/model_architecture.py`) is the correct architecture to build on.
- `dataset.signer_kfold_splits` + `check_signer_partition` were exercised
  against a synthetic 20-signer, 10-class dataset: `GroupKFold` correctly
  put every signer in exactly one test fold, `GroupShuffleSplit` correctly
  carved a disjoint validation-signer set out of each fold's training
  signers, and the sanity check both passed on a correct partition and
  correctly raised `AssertionError` when a violation was deliberately
  injected.
- `train.py` + `evaluate.py` were run end-to-end in both `random` mode (on
  the real `data/KSL77_joint_stream_47pt.pkl`) and `signer_kfold` mode (on a
  synthetic class-separable manifest dataset), including the
  cross-fold mean/std aggregation and the combined out-of-fold per-signer
  accuracy table. This caught one real bug (`BatchNorm1d` raising on a
  trailing training batch of size 1) which is now fixed via `drop_last=True`
  on the training loader.
- `extract_landmarks.py`'s non-MediaPipe logic (video discovery, the
  `<NN>_<class_id>.mp4` regex, folder/filename class-id mismatch
  logging+`--strict`, per-video caching/resume, `result_to_47pt`'s
  frame-to-47-point conversion, and the wrist-proximity hand-assignment
  heuristic) was exercised with a monkeypatched fake detector and real
  tiny synthetic `.mp4` files, since real MediaPipe Tasks graphs
  (`HolisticLandmarker`/`HandLandmarker`/`PoseLandmarker`) crash in this
  local macOS environment with `Check failed: service_ Service is
  unavailable` from a Metal-backed calculator — a known environment issue,
  not a bug in this repo (see `extract_landmarks.py --device-check`).
- **Update: `scripts/smoke_test_colab.py` has since been run for real, on
  Colab** (mediapipe 0.10.35, Python 3.13) against real KSL-77 clips.
  `HolisticLandmarker` loaded and ran successfully; the raw result exposed
  the expected fields (`left_hand_landmarks`, `right_hand_landmarks`,
  `pose_landmarks`, etc.), matching what `result_to_47pt()` was written
  against. Output shapes were exactly `(3, 32, 47)` / `(32, 47)` with no
  NaNs. Per-clip hand-detection fraction ranged 69–97%, pose detection was
  100%. `requirements.txt` is now pinned to `mediapipe==0.10.35` (the
  version actually verified, not just the version that happened to already
  be in the notebook's Colab cache).
  - This run also surfaced a MediaPipe warning — `Using NORM_RECT without
    IMAGE_DIMENSIONS is only supported for the square ROI` — on non-square
    KSL-77 clips. `scripts/smoke_test_colab.py` now has `--save-overlay DIR`,
    which renders our own `(3, 32, 47)` feature array (not the raw MediaPipe
    result, so it also exercises our own left/right/pose indexing) back onto
    frames 0/10/20/31 of each test clip, specifically to catch x/y offset
    before trusting this on the full corpus. It also now prints each video's
    frame width×height and separate left-hand/right-hand/pose detection
    fractions (previously only a combined "any hand" fraction was surfaced).
    Verify the overlay PNGs land on the actual hands/shoulders before running
    the full extraction — see `extract_landmarks.py`'s and
    `scripts/smoke_test_colab.py`'s own printed guidance for what's been
    checked on our side vs. what only the overlay can confirm.
  - `Detector` now has a `close()`/context-manager so the underlying
    MediaPipe Tasks graph(s) are released explicitly instead of relying on
    `__del__` at interpreter shutdown (which raised a spurious `TypeError`
    on exit).
