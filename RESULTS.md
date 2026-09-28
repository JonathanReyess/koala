# Results

**Headline: config f (`--mirror-aug --normalize-body`, `--tta-mirror` at
eval), 3-seed mean on signer-out 5-fold: 83.2% top-1 / 94.1% top-3 on
unseen signers.** This is the production recipe (see "Decisions" below).
Everything below builds up to that number: first the (a)/(b)/(c) baseline
comparison that established signer-out (c) as the metric that matters, then
the per-signer diagnosis that motivated `transforms.py`, then the ablation
sweep and seed check that produced config f.

## Baseline: does the split matter?

Comparison of the model across three conditions, run on Colab (T4) against
the real KSL-77 raw videos. **(c) — signer-out 5-fold — is the headline
number** of *this section*. It's the only row that actually answers "does
the model work on a signer it has never seen," which was the open question
from AUDIT.md §4. (a) and (b) are useful context (they isolate what changed
between the old and new feature-extraction pipelines on a matched,
non-signer-grouped split) but neither one tells you anything about signer
generalization by itself.

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

All rows below this one are `signer_kfold` (5 folds) only — the random
split (a)/(b) already served its purpose (motivating why signer-out is the
metric that matters) and isn't re-run per ablation.

## Per-signer diagnosis (from (c)'s combined out-of-fold per-signer accuracy)

Three signers account for most of (c)'s fold-to-fold spread:

- **Signer 08 — 25% accuracy.** The *only* signer with left-hand motion exceeding right-hand motion (0.033 vs 0.017) and left-hand detection exceeding right-hand detection (0.64 vs 0.34). Likely left-handed, or the clip is mirrored relative to the rest of the corpus — either way, this signer's dominant hand doesn't match the convention the other 19 signers (and thus most of the training data) establish.
- **Signer 14 — 55% accuracy.** Mean nose y = 0.60, vs. ~0.24–0.53 for every other signer — a framing issue (camera positioned differently, or signer seated/positioned lower in frame), not a signing-quality problem.
- **Signer 18 — 58% accuracy.** Smallest mean shoulder width (0.158 — farthest from the camera of any signer) and unusually short clips (mean 35 frames vs. 50–145 for everyone else), i.e. tightly trimmed with no idle frames before/after the sign — the 32-frame `np.linspace` sampling has much less margin to work with here than on a typically-padded clip.
- **Excluding signer 08 alone, signer-out top-1 rises from 78.22% to ~81%** — one atypical (possibly mirrored/left-handed) signer is disproportionately responsible for (c)'s worst-case spread, not a general failure to generalize across signers.

This is direct motivation for the ablations below: `--mirror-aug`/`--tta-mirror` target signer 08's case specifically (handedness/mirroring shouldn't matter if the model sees both orientations during training and/or is evaluated on both), `--normalize-body` targets signer 18's case (distance-from-camera shouldn't change the normalized geometry), and `--trim-idle` targets signer 18's short-clip case from a different angle (recovering the active window regardless of how much idle padding surrounds it).

Hand-detection check ([`scripts/compare_hand_detection.py`](scripts/compare_hand_detection.py), old legacy `.pkl` vs. the real Colab `mask.npy`): any-hand detected in 66.3% of frames (legacy) vs. 65.5% (new Tasks API) — **the new Tasks-API extraction is not finding hands less often than the old legacy one**; the (b) vs (a) gap earlier is not explained by worse hand detection.

## Ablation sweep (all `signer_kfold`, 5 folds; runs d–g evaluated with `--tta-mirror`)

| run | flags | top-1 | top-3 | macro-F1 | signer 08 | signer 14 | signer 18 |
|---|---|---|---|---|---|---|---|
| c (baseline) | none | 0.782 ± 0.076 | 0.905 ± 0.079 | 0.785 | 0.25 | 0.55 | 0.58 |
| c + TTA only | eval `--tta-mirror`, no mirror-training | 0.715 ± 0.056 | 0.887 ± 0.066 | 0.737 | 0.46 | 0.53 | 0.48 |
| d | `--mirror-aug` | 0.782 ± 0.070 | 0.915 ± 0.046 | 0.776 | 0.67 | 0.52 | 0.45 |
| e | `--normalize-body` | 0.803 ± 0.030 | 0.918 ± 0.031 | 0.800 | 0.81 | 0.66 | 0.64 |
| **f** | `--mirror-aug --normalize-body` | **0.835 ± 0.057** | **0.949 ± 0.031** | **0.826** | 0.87 | 0.63 | 0.69 |
| g | f + `--trim-idle` | 0.837 ± 0.059 | 0.947 ± 0.032 | 0.832 | 0.86 | 0.56 | 0.70 |

## Seed check for config f (`--mirror-aug --normalize-body`, same folds, different seeds)

| seed | top-1 | top-3 | macro-F1 |
|---|---|---|---|
| 0 | 0.835 ± 0.057 | 0.949 ± 0.031 | 0.826 |
| 1 | 0.826 ± 0.073 | 0.936 ± 0.043 | 0.821 |
| 2 | 0.836 ± 0.068 | 0.938 ± 0.043 | 0.831 |
| **mean of 3 seeds** | **0.832** | **0.941** | **0.826** |

**Headline metric: 3-seed mean for config f — 83.2% top-1 / 94.1% top-3 on unseen signers.**

## Grading thresholds (confidence gates for the browser)

The browser grades on the averaged (TTA mirror) softmax probabilities of the deployed config-f model
(`src/frontend/src/lib/inference/config.ts`, logic in `grading.ts`), in this order:

1. **correct** — target is top-1 **and** p(target) ≥ `CORRECT_MIN`
2. **close** — target in the top 3 **and** p(target) ≥ `CLOSE_MIN` ("Almost — …"; not a miss for spaced repetition)
3. **confused** — top-1 ≠ target **and** p(top-1) ≥ `CONFUSION_MIN` → "That looked like <word>." (a miss)
4. otherwise **incorrect** → "Not quite — watch the example and try again." (a miss; never names a word)

**Calibration data:** `scripts/calibrate_thresholds.py --tta-mirror` on the 15 config-f `signer_kfold` folds
(`runs/f_mirror_norm`, `f_seed1`, `f_seed2`): **3,684 held-out predictions**, 67 classes, pooled top-1 **83.25%**,
top-3 **94.06%**. Raw output: `predictions.csv` + `probs.npz` on Drive (`KSL_Project/runs/calibration/`); the tables below
regenerate with `python scripts/calibrate_thresholds.py --from-npz probs.npz`.

> **Scope caveat:** every clip here is a *fluent signer* doing the *correct* sign, held out by signer. It says nothing
> about **learners** (who sign imperfectly) or about **random movement / non-signs**. "Impostor" rows below are a
> stand-in (a fluent clip of word A graded against target B ≠ A), not real learner mistakes. Re-calibrate (and
> especially re-check CLOSE_MIN) once real learner attempts are available.

### How to read the sweeps
- **CORRECT_MIN** — *correct accepted*: share of correct top-1 predictions that still pass the gate. *False accept*:
  share of (clip × wrong target) attempts that would be graded "correct" ("Perfect!" for the wrong sign).
- **CLOSE_MIN** — *close kept*: share of genuine near-misses (target in top 3 but not top 1) still called "close".
  *False close*: share of (clip × wrong target) attempts that would be called "close".
- **CONFUSION_MIN** — *wrong shown*: share of the model's wrong predictions that would be shown as "That looked like X"
  (each is a correctly signed clip we would mis-name). *Naming precision*: among clips where a word would be named, the
  share where the named word is the word actually signed. *Coverage*: share of all clips that get a name at all.

### CORRECT_MIN sweep

| t | correct accepted | false accept |
|---|---|---|
| 0.05 | 100.0% | 0.3% |
| 0.10 | 100.0% | 0.3% |
| 0.15 | 100.0% | 0.3% |
| 0.20 | 99.9% | 0.3% |
| 0.25 | 99.8% | 0.2% |
| 0.30 | 99.5% | 0.2% |
| 0.35 | 99.1% | 0.2% |
| 0.40 | 98.2% | 0.2% |
| 0.45 | 96.8% | 0.2% |
| 0.50 | 95.1% | 0.1% |
| 0.55 | 92.7% | 0.1% |
| 0.60 | 90.3% | 0.1% |
| 0.65 | 87.8% | 0.1% |
| 0.70 | 84.4% | 0.1% |
| 0.75 | 80.9% | 0.0% |
| 0.80 | 77.0% | 0.0% |
| 0.85 | 72.0% | 0.0% |
| 0.90 | 65.2% | 0.0% |
| 0.95 | 52.0% | 0.0% |

### CLOSE_MIN sweep

| t | close kept | false close |
|---|---|---|
| 0.05 | 91.0% | 1.2% |
| 0.10 | 77.4% | 0.8% |
| 0.15 | 64.6% | 0.6% |
| 0.20 | 48.5% | 0.5% |
| 0.25 | 37.9% | 0.4% |
| 0.30 | 27.4% | 0.3% |
| 0.35 | 15.1% | 0.2% |
| 0.40 | 7.5% | 0.2% |
| 0.45 | 2.8% | 0.2% |
| 0.50 | 0.0% | 0.1% |
| 0.55 | 0.0% | 0.1% |
| 0.60 | 0.0% | 0.1% |
| 0.65 | 0.0% | 0.1% |
| 0.70 | 0.0% | 0.1% |
| 0.75 | 0.0% | 0.0% |
| 0.80 | 0.0% | 0.0% |
| 0.85 | 0.0% | 0.0% |
| 0.90 | 0.0% | 0.0% |
| 0.95 | 0.0% | 0.0% |

### CONFUSION_MIN sweep

| t | wrong shown | of all clips | naming precision | coverage |
|---|---|---|---|---|
| 0.05 | 100.0% | 16.7% | 83.3% | 100.0% |
| 0.10 | 100.0% | 16.7% | 83.3% | 100.0% |
| 0.15 | 99.8% | 16.7% | 83.3% | 100.0% |
| 0.20 | 98.5% | 16.5% | 83.4% | 99.7% |
| 0.25 | 95.5% | 16.0% | 83.9% | 99.1% |
| 0.30 | 91.1% | 15.3% | 84.4% | 98.1% |
| 0.35 | 82.7% | 13.8% | 85.6% | 96.3% |
| 0.40 | 74.4% | 12.5% | 86.8% | 94.2% |
| 0.45 | 63.7% | 10.7% | 88.3% | 91.3% |
| 0.50 | 51.4% | 8.6% | 90.2% | 87.8% |
| 0.55 | 43.1% | 7.2% | 91.4% | 84.4% |
| 0.60 | 35.0% | 5.9% | 92.8% | 81.0% |
| 0.65 | 29.7% | 5.0% | 93.6% | 78.1% |
| 0.70 | 24.1% | 4.0% | 94.6% | 74.3% |
| 0.75 | 18.8% | 3.1% | 95.5% | 70.5% |
| 0.80 | 14.1% | 2.4% | 96.4% | 66.4% |
| 0.85 | 9.7% | 1.6% | 97.4% | 61.6% |
| 0.90 | 6.0% | 1.0% | 98.2% | 55.3% |
| 0.95 | 2.3% | 0.4% | 99.1% | 43.7% |

### Outcome mix

Share of each grade. *Genuine* = the signer signed the target word (all held-out clips, target = true label). *Impostor* = every clip × every wrong target.

| Thresholds (CORRECT / CLOSE / CONFUSION) | Genuine: correct | close | wrongly named | not quite | Impostor: "Perfect!" | close | named | not quite |
|---|---|---|---|---|---|---|---|---|
| Placeholders (0.40 / 0.15 / 0.60) | 81.8% | 8.5% | 4.3% | 5.5% | 0.2% | 0.4% | 80.8% | 18.6% |
| Suggested by the script (0.50 / 0.05 / 0.75) | 79.2% | 13.9% | 1.9% | 5.0% | 0.1% | 1.0% | 70.2% | 28.7% |
| **Chosen (shipped)** (0.50 / 0.10 / 0.75) | 79.2% | 12.4% | 2.5% | 5.9% | 0.1% | 0.7% | 70.4% | 28.8% |

("Wrongly named" = a correctly signed clip shown "That looked like <other word>"; "named" for impostors is mostly a
*correct* naming of the word actually signed, since the top-1 is usually the true sign.)

### Chosen values and why

| | value | basis |
|---|---|---|
| `CORRECT_MIN` | **0.50** | Script suggestion: the highest t that still accepts ≥ 95% of correct predictions. |
| `CONFUSION_MIN` | **0.75** | Script suggestion: the lowest t with naming precision ≥ 95%. Wrongly named genuine attempts drop from 4.3% (placeholder 0.60) to 1.9–2.5%. |
| `CLOSE_MIN` | **0.10** | Script suggested 0.05 (lowest t with false-close ≤ 2%). Raised to 0.10 by hand: false-close falls from 1.2% to 0.8% (a third fewer accidental "Almost" results), at the cost of keeping 77% instead of 91% of genuine near-misses. The intent is that random or unrelated movement shouldn't easily earn an "Almost" — a wrong call that flatters non-signing is worse than a missed "Almost" for a real near-miss, which just becomes "Not quite". |

Side effect of 0.10 vs 0.05 on genuine attempts (because "close" is checked before "confused"): close 13.9% → 12.4%,
wrongly named 1.9% → 2.5%, not quite 5.0% → 5.9%; correct is unchanged at 79.2%.

These are calibrated on fluent held-out signers only. Nothing here was measured on random movement or on learners;
treat CLOSE_MIN in particular as a starting point to revisit with real usage data.


## Decisions

- **Config f (`--mirror-aug --normalize-body` at train, `--tta-mirror` at eval) is the production recipe.** It's the best top-1/top-3/macro-F1 of the sweep, and the gain over baseline (c) (78.2% → 83.2%, ~5 points) holds up across all 3 seeds, not just the one that happened to be reported first.
- **`--trim-idle` is dropped from the production recipe (run g, not f).** Adding it to f produced no gain outside the noise floor (0.837 vs 0.835 top-1, well within f's own ±0.057 fold-to-fold std), it made signer 14 specifically *worse* (0.63 → 0.56 — the opposite of e/f's trend on that signer), and it's extra logic that would need to be ported to the browser for PR 3 with no measured benefit to justify that cost.
- **TTA-only (no mirror training) hurts, and that's expected, not a bug.** Averaging softmax over a clip and its mirror only helps if the model has actually learned to recognize the *mirrored* orientation of a sign — a model trained without `--mirror-aug` has never seen a mirrored clip, so mirroring its input at eval time just feeds it out-of-distribution data half the time, dragging every metric down (0.782 → 0.715 top-1) rather than up. `--mirror-aug` and `--tta-mirror` are a pair; using one without the other is not a meaningful configuration to compare against baseline.
- **Config f was selected by looking at CV results across 6 configs (c, c+TTA, d, e, f, g).** That means 83.2% carries some optimism from having picked the best-looking option out of several after seeing all of their signer-out numbers — the 3-seed check reduces (but does not eliminate) the risk that f specifically got a lucky draw of folds/initialization; it does not protect against having chosen the transform combination itself based on the same 5 folds it's being reported on. Treat 83.2% as a reasonable estimate, not a guaranteed number on a genuinely fresh signer.

## Next: produce the deployment checkpoint (Colab commands)

`--stop-epoch 34` = the median best epoch across the 15 config-f CV folds (3 seeds × 5 folds each) was epoch 33 (zero-indexed) → 34 epochs.

```bash
# On Colab, after `git clone` + `pip install -r requirements.txt`:
FEATS=/content/drive/MyDrive/KSL_Project/features_v2

# 1. Full training: config f (--mirror-aug --normalize-body), all 20 signers, --stop-epoch 34
python train.py --features $FEATS/features.npy --mask $FEATS/mask.npy --manifest $FEATS/manifest.csv \
  --split-mode full --stop-epoch 34 --mirror-aug --normalize-body --out-dir runs/full_model

# 2. ONNX export (opset 17, dynamic batch) + PyTorch-vs-onnxruntime parity check
python export_onnx.py --checkpoint-dir runs/full_model --out runs/full_model/fold_0/model.onnx

# 3. Golden vectors for PR 3's TypeScript port to test against (see docs/PREPROCESSING.md)
python scripts/generate_golden_vectors.py --checkpoint-dir runs/full_model \
  --features $FEATS/features.npy --mask $FEATS/mask.npy --manifest $FEATS/manifest.csv \
  --out golden_vectors.json
```

Note `--tta-mirror` is not passed to `train.py` above -- it's an eval-only flag (`evaluate.py`/`generate_golden_vectors.py` apply it themselves at inference time; nothing to pass at training time for it).

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

- `train.py --split-mode full --stop-epoch N` and `export_onnx.py` both have
  a dedicated test file, `test_train_full_and_export.py` (5 tests, all
  passing, tiny synthetic data): `full` mode requires `--stop-epoch`, trains
  exactly `N` epochs while `config.json`'s `scheduler_t_max` still equals
  `--epochs` (confirming `--stop-epoch` doesn't change the cosine
  schedule's shape, only when training stops), the resulting checkpoint
  reloads with no shape mismatch, and `best_epoch` is recorded correctly
  for the pre-existing `random`/`signer_kfold` modes too. `export_onnx.py`'s
  PyTorch-vs-onnxruntime parity check was run for real (not just via CLI
  exit code, but calling `parity_check()` directly): max abs diff on the
  order of `1e-7`–`1e-8` across random-noise, `[0,1]`-uniform, and
  varying-batch-size (1/2/4) inputs, all well under the `1e-4` tolerance;
  the ambiguous-multi-fold-checkpoint-dir error path was also verified.
  `scripts/generate_golden_vectors.py` was run against a toy checkpoint on
  both the legacy-`.pkl`-with-inferred-mask path and the
  manifest-with-real-mask path — correct shapes, and `probs_tta_averaged`
  summing to 1.0 as expected for a proper softmax average. (Real golden
  numbers require the actual config-f production checkpoint — see the
  Colab commands below.)
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
