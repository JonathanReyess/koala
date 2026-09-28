# Results

Comparison of the model across three conditions, run on Colab (T4) against
the real KSL-77 raw videos. **(c) — signer-out 5-fold — is the headline
number.** It's the only row that actually answers "does the model work on a
signer it has never seen," which was the open question from AUDIT.md §4. (a)
and (b) are useful context (they isolate what changed between the old and
new feature-extraction pipelines on a matched, non-signer-grouped split) but
neither one tells you anything about signer generalization by itself.

**No number below is described as an improvement over another unless it is
verified on row (c).** In particular: (b)'s top-1 (83.74%) is *lower* than
(a)'s (88.21%) on the same kind of split — that is not evidence the new
Tasks-API extraction is worse, since (a) and (b) use different underlying
features (legacy silent-zero-fill vs. explicit-mask Tasks API) evaluated on
different concrete train/test partitions (the random split's *indices*
aren't guaranteed identical between the two feature sets, since (b)'s
dataset went through `drop_undetected_samples`, which (a)'s legacy pkl never
does). The only number that should be trusted as "how well does this
approach actually generalize" is (c).

| | Features | Split | Top-1 | Top-3 | Macro-F1 | Notes |
|---|---|---|---|---|---|---|
| (a) Old checkpoint, random split | `data/KSL77_joint_stream_47pt.pkl` (legacy notebook extraction, silent zero-fill for missing landmarks) | `random` (stratified 80/10/10, seed 42, **not** signer-grouped) | 88.21% | 95.53% | 0.8806 | Reproduced directly from `src/backend/best_model.pt` via `dataset.random_split(seed=42)` + `evaluate.py`; matches `notebook/KSL.ipynb` Cell 9's own reported 88.21% exactly (see AUDIT.md §8). Test set is only 246 samples (3-4/class) — high variance. |
| (b) Retrained, new Tasks features, random split | `extract_landmarks.py` output (Tasks API, explicit per-joint mask) | `random` (same split *mode* as (a), not the same indices — see caveat above) | 83.74% | 93.90% | 0.8367 | Run on Colab (T4). Not comparable to (a) as an "improvement/regression" — different underlying features and dataset (see caveat above). Useful only as (b) vs (c)'s own gap, below. |
| **(c) Retrained, new Tasks features, signer-out 5-fold** ← headline | same features as (b) | `signer_kfold` (`GroupKFold` by `signer_id`, 5 folds; see `dataset.py`) | **78.22% ± 7.56%** | **90.51% ± 7.89%** | **0.7850 ± 0.0704** | Run on Colab (T4). Per-fold top-1: 0.691, 0.697, 0.835, 0.879, 0.810 — the fold-to-fold spread itself (±7.6 points) is the headline finding: which 4 signers land in the held-out fold matters more than run-to-run noise would suggest. **The (b)→(c) gap (83.74% → 78.22%, about 5.5 points) is the actual cost of the random split's signer leakage** that AUDIT.md §4 flagged as an unmeasurable risk — now measured. |
| (b'), (c') Same as (b)/(c), + `--mirror-aug`/`--normalize-body`/`--trim-idle` | *pending* | *pending* | *pending* | *pending* | *pending* | Not run yet. Placeholder rows for the ablations enabled by `transforms.py` (see below) — fill in once run, and remember the "no improvement claim without (c)" rule applies to these too: an ablation only counts as a real improvement if it improves the **signer-out** number, not the random-split one. |

## Per-signer diagnosis (from (c)'s combined out-of-fold per-signer accuracy)

Three signers account for most of (c)'s fold-to-fold spread:

- **Signer 08 — 25% accuracy.** The *only* signer with left-hand motion exceeding right-hand motion (0.033 vs 0.017) and left-hand detection exceeding right-hand detection (0.64 vs 0.34). Likely left-handed, or the clip is mirrored relative to the rest of the corpus — either way, this signer's dominant hand doesn't match the convention the other 19 signers (and thus most of the training data) establish.
- **Signer 14 — 55% accuracy.** Mean nose y = 0.60, vs. ~0.24–0.53 for every other signer — a framing issue (camera positioned differently, or signer seated/positioned lower in frame), not a signing-quality problem.
- **Signer 18 — 58% accuracy.** Smallest mean shoulder width (0.158 — farthest from the camera of any signer) and unusually short clips (mean 35 frames vs. 50–145 for everyone else), i.e. tightly trimmed with no idle frames before/after the sign — the 32-frame `np.linspace` sampling has much less margin to work with here than on a typically-padded clip.
- **Excluding signer 08 alone, signer-out top-1 rises from 78.22% to ~81%** — one atypical (possibly mirrored/left-handed) signer is disproportionately responsible for (c)'s worst-case spread, not a general failure to generalize across signers.

This is direct motivation for the ablations above: `--mirror-aug`/`--tta-mirror` target signer 08's case specifically (handedness/mirroring shouldn't matter if the model sees both orientations during training and/or is evaluated on both), `--normalize-body` targets signer 18's case (distance-from-camera shouldn't change the normalized geometry), and `--trim-idle` targets signer 18's short-clip case from a different angle (recovering the active window regardless of how much idle padding surrounds it).

## How to fill in (b'), (c')

```bash
# On Colab, after `git clone` + `pip install -r requirements.txt`:
FEATS=/content/drive/MyDrive/KSL_Project/features_v2

python scripts/compare_hand_detection.py --old-pkl data/KSL77_joint_stream_47pt.pkl --new-mask $FEATS/mask.npy

python train.py --features $FEATS/features.npy --mask $FEATS/mask.npy --manifest $FEATS/manifest.csv \
  --split-mode random --out-dir runs/b_prime_random --mirror-aug --normalize-body --trim-idle
python evaluate.py --features $FEATS/features.npy --mask $FEATS/mask.npy --manifest $FEATS/manifest.csv \
  --checkpoint-dir runs/b_prime_random --tta-mirror --out runs/b_prime_random/eval.json

python train.py --features $FEATS/features.npy --mask $FEATS/mask.npy --manifest $FEATS/manifest.csv \
  --split-mode signer_kfold --out-dir runs/c_prime_signer_kfold --mirror-aug --normalize-body --trim-idle
python evaluate.py --features $FEATS/features.npy --mask $FEATS/mask.npy --manifest $FEATS/manifest.csv \
  --checkpoint-dir runs/c_prime_signer_kfold --tta-mirror --out runs/c_prime_signer_kfold/eval.json
```

Then replace the *pending* cells above with `evaluate.py`'s printed
top1/top3/macro_f1 (mean ± std for row (c')), and only call it an
improvement if (c') > (c) — not if (b') > (b).

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

- `transforms.py` (`mirror_clip`, `normalize_body`, `trim_idle`) has a
  dedicated unit test suite, `test_transforms.py` (`python -m unittest
  test_transforms.py`, 9 tests, all passing): mirroring twice returns the
  exact original clip (checked for the all-shoulders-detected case, a
  partial-shoulder-detection case exercising the per-frame/clip-mean
  fallback, and a no-shoulders-detected case exercising the global 0.5
  fallback), a direct check that mirroring actually swaps the left/right
  hand blocks and shoulder/elbow pairs into the correct slots, body
  normalization drives shoulder width to exactly 1.0 (on a clip with
  constant shoulder geometry) while leaving already-missing joints at
  exactly 0, and idle-trimming recovers an active window deliberately
  padded with static idle frames at both ends (and leaves a fully-static
  clip unchanged, since there's no active window to trim to).
- `--mirror-aug`, `--normalize-body`, `--trim-idle` (+
  `--trim-idle-motion-threshold`), and `--tta-mirror` were all run
  end-to-end together against real data (the legacy `.pkl`, which has no
  real mask — exercising the "infer mask from all-zero joints" fallback
  path) with `train.py`/`evaluate.py`, with no errors. `evaluate.py`
  correctly auto-resolves `--normalize-body`/`--trim-idle`/
  `--trim-idle-motion-threshold` from the checkpoint's `config.json` (same
  pattern as `--min-hand-frac`) and correctly refuses to run when an
  explicitly-passed value disagrees with what training used — verified
  both directions. `--split-mode signer_kfold` was also re-run with
  `--mirror-aug` against the synthetic signer-labeled dataset, including
  the new fold-marked per-signer accuracy table.
- `scripts/compare_hand_detection.py` was run against the real
  `data/KSL77_joint_stream_47pt.pkl` (numpy-only, no torch/mediapipe
  needed): left hand detected in 44.75% of frames, right hand 58.97%, any
  hand 66.29%, pose 99.86% — this is the legacy-pipeline baseline the
  Colab `--new-mask` run should be compared against.
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
