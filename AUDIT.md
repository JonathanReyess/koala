# Koala Repo Audit

Scope: read-only investigation of the KSL sign-recognition app (notebook, backend, frontend, data, docs). No code was changed. All numbers below were verified by running code against the files in this repo (not just read from comments/docs) — see "How verified" notes.

---

## 1. Class count and label maps: 67 vs 77

**Both numbers are real and refer to different things — this is not a bug, but the docs conflate them.**

- `data/class_label.p` (== top-level `class_label.p`, byte-identical) is a dict with **101 keys** (`0`–`100`). Keys `1`–`77` hold real glosses (English, not Korean — see caveat below); keys `0` and `78`–`100` are blank `' '` padding. So **77** is the size of the raw KSL "folder ID" space used in the original dataset.
- The actual training data (`data/KSL77_joint_stream_47pt.pkl`) only contains **67 distinct labels**, drawn from folder IDs `1`–`77` with gaps (e.g. 12, 19, 28, 33, 35, 45, 46, 53, 73, 75 never appear). This matches `data/DATA_README.md`'s claim of "67 unique sign language folders" with gaps in the raw numbering — verified directly against the `.pkl` (`np.unique(y)` → 67 values).
- **"Korean glosses" caveat:** `data/class_label.txt` / `class_label.p` do not contain Korean text — they're English gloss words (`'hi'`, `'meat'`, `'bi bim rice'`, etc.), i.e. translations/labels for the signs, not Hangul. The only place actual Hangul appears is the frontend's hardcoded `WORD_LIST` in `src/frontend/src/pages/Learn.tsx` (31 words with a `korean` field, e.g. `안녕`, `만나다`) — a second, independently-typed vocabulary that overlaps but isn't generated from `class_label.p`.

**Reverse label map — three different versions exist in the repo, only one is deployed:**

| Location | Purpose | Status |
|---|---|---|
| `notebook/KSL.ipynb` (cell 6) | `label_map = {old: new for new, old in enumerate(sorted(unique_labels))}` — re-indexes the 67 sparse folder IDs to dense `0..66` for `CrossEntropyLoss`; `reverse_label_map` inverts it. | Ground truth. Reproduced independently below and confirmed correct. |
| `src/backend/app.py` (deployed) | Hardcodes the same 67-entry `0→folder_id` map, plus a 31-entry `perfect_mapped_classes` allowlist. | **Matches the notebook's actual mapping exactly** (verified by rebuilding `label_map` from the `.pkl` and diffing). Its `perfect_mapped_classes` list is *exactly* the 31 classes that scored F1=1.00 in the notebook's test run (verified against the printed classification report). This is the version actually running in production. |
| `models/config.py` (unused) | A second, differently-selected 31-entry `perfect_mapped_classes` + matching `reverse_label_map` subset. | **Stale/wrong.** 7 of its 31 "perfect" indices (9, 13, 21, 25, 45, 63, 64) do *not* have F1=1.00 in the actual eval, and it's missing 7 that do (16, 17, 19, 22, 39, 52, 59). See §7/§9 — `models/` is dead code, never imported by anything that runs. |

**Why a reverse map at all:** the model is trained on dense 0-indexed labels, but the frontend/UI and `class_label.p` speak in terms of original KSL folder IDs (1–77, with gaps). `reverse_label_map[model_output_index] → folder_id` translates one to the other so the frontend's `WORD_TO_ID_MAP` (folder-ID strings) can be compared to what the model predicts.

**Full 0-66 → folder-ID map** (recomputed from the `.pkl`, matches `app.py` and the notebook exactly):
```
0:1 1:2 2:3 3:4 4:5 5:6 6:7 7:8 8:9 9:10 10:11 11:13 12:14 13:15 14:16 15:17
16:18 17:20 18:21 19:22 20:23 21:24 22:25 23:26 24:27 25:29 26:30 27:31 28:32 29:34 30:36 31:37
32:38 33:39 34:40 35:41 36:42 37:43 38:44 39:47 40:48 41:49 42:50 43:51 44:52 45:54 46:55 47:56
48:57 49:58 50:59 51:60 52:61 53:62 54:63 55:64 56:65 57:66 58:67 59:68 60:69 61:70 62:71 63:72
64:74 65:76 66:77
```

---

## 2. Demo videos: location, provenance, loading

- Location: `src/frontend/public/videos/*.mp4` — 98 files, static assets shipped in the repo (not fetched from a backend or CDN). Total ~379 MB.
- Provenance: filenames are `{word}_example{1,2}.mp4`, one/two clips per gloss, each ~1–2 seconds long (checked via `ffprobe`, e.g. 1.71s). That length and the direct `{word}` naming (matching `class_label` glosses, not arbitrary names) is consistent with these being short **clips cut directly from the KSL-77 source videos**, not separately recorded. `Learn.tsx` (lines 481–491) explicitly attributes them in the UI: *"Video source: Yang et al., 'The Korean Sign Language Dataset for Action Recognition,' MMM 2020"* linking to the same paper cited in `ATTRIBUTION.md`.
- Loading: `VideoExampleCard.tsx` renders a plain HTML5 `<video>` with `<source src={`/videos/${word}_example${example}.mp4`} />` — served directly by Vite's static file handling / Vercel, no API call, no MediaPipe involved for playback.
- **Coverage mismatch:** there are demo videos for **49 distinct words**, but only **31** of those are in the practiceable `WORD_LIST` (`Learn.tsx`) that the recording/grading flow uses. The other **18 are unreachable dead assets** — shipped in the bundle, never selected by the practice queue:

  `care`, `effort`, `experience`, `finally`, `fine`, `good`, `introduction`, `invite`, `receive`, `see`, `sorry`, `special`, `test`, `thank`, `time`, `want`, `who`, `yesterday`

  (Not deleted as part of this pass — kept for now in case a future PR expands `WORD_LIST` to cover more of the 67 trained classes, at which point most of these become useful again.)
- **Bug found, fixed in PR 1:** one file was `how many_example2.MP4` (uppercase extension), while the code always requests lowercase `.mp4`. This would 404 in case-sensitive production hosting (Vercel/Linux) even though it silently worked in local macOS dev (case-insensitive filesystem). Renamed to `how many_example2.mp4`, and `src/frontend/scripts/check-videos.mjs` (wired into `npm run prebuild` / `npm run check:videos`) now fails the build if this regresses or any other practiceable word's video goes missing/mismatched.

---

## 3. MediaPipe setup

- **Legacy Holistic solutions API throughout** — `from mediapipe.python.solutions import holistic` (`models/model_utils.py`) and `mp.solutions.holistic` (`src/backend/app.py`, and the notebook). Nowhere in the repo is the newer Tasks `HolisticLandmarker` API used.
- **The exact 47 indices** (identical in the notebook, `app.py`, and `model_utils.py`):
  - Indices **0–20**: Left hand, all 21 MediaPipe hand landmarks, in landmark order.
  - Indices **21–41**: Right hand, all 21 MediaPipe hand landmarks, in landmark order (written at `frame_coords[i + 21]`).
  - Indices **42–46**: 5 selected **pose** landmarks, at `POSE_INDICES = [0, 11, 12, 13, 14]` → MediaPipe Pose's Nose, Left Shoulder, Right Shoulder, Left Elbow, Right Elbow (per the notebook's own comment: `# Nose, Shoulders, Elbows`).
  - Each of the 47 rows is `(x, y, z)`; if a hand/pose isn't detected in a frame, that block stays all-zero (`np.zeros`) — there's no missing-data flag, just zeros indistinguishable from "landmark at origin."
- **Frame sampling:** the whole video is read into memory with `cv2.VideoCapture` (all frames buffered as a Python list), then `SEQUENCE_LENGTH=32` frame indices are chosen via `np.linspace(0, len(frames)-1, 32, dtype=int)` — i.e. 32 evenly-spaced frames regardless of source video length or fps, then MediaPipe Holistic is run once per sampled frame (not once per raw frame). No temporal windowing, no fixed clip duration — a 2-second and a 10-second submission are both squashed to 32 samples.

---

## 4. Notebook train/val/test split — signer leakage

**File:** `notebook/KSL.ipynb`, Cell 6 (index 11 in the raw JSON).

```python
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
...
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
```

This is a plain **per-sample random split, stratified only by class label** (67 classes, 14–20 samples each → 1228 total). There is no grouping key at all.

**Signer identity is not tracked anywhere in the pipeline** — this is the headline finding for this section, not just "can't tell if it leaks":
- The feature-extraction loop (`process_all_videos`, Cell 5) walks `BASE_INPUT_DIR/<class_folder>/*.mp4` and derives the label with `class_label = int(os.path.basename(os.path.dirname(video_path)))` — **only the class-folder name is kept**. The video's own filename (which is where a signer ID would plausibly live, per the original KSL-77 dataset's directory convention) is discarded entirely; it's read for the `cv2.VideoCapture` call and then thrown away.
- The resulting `.pkl` (`features, labels`) has no per-sample metadata at all beyond the 67 class labels — no signer field, no filename, no video path survives serialization.
- **Consequence:** it is not just "unknown whether a signer appears in both train and test" — it is *structurally impossible* to check from this pipeline's output, and if (as is typical for this kind of dataset) each class's ~15–20 clips come from a handful of repeat signers, `train_test_split` with no grouping will near-certainly put the same signer in both train and test. That would inflate the reported 88.21% test accuracy relative to how the model would perform on a genuinely unseen signer.
- Compounding this: because the split test set only has **246 samples across 67 classes (3–4 per class)**, and the earlier validation split even fewer, the accuracy/F1 numbers are already very high-variance from sample size alone — signer leakage would be an additional, unmeasurable inflation on top of that.

---

## 5. Inference path: browser vs FastAPI, payload, latency

**Browser (React, `LearningCard.tsx`):**
- Live camera preview via `getUserMedia`, no MediaPipe/model code runs client-side at all.
- Recording: `MediaRecorder` captures a `video/webm` blob (or a user-uploaded file via `<input type="file">`).
- On Submit, `runInference()` builds a `FormData` with the blob under the field name `video`, filename `"sign_video.webm"`, and does:
  ```
  POST {VITE_API_URL || "http://34.239.230.9:8000"}/predict
  Content-Type: multipart/form-data  (video: <blob>)
  ```
  — note the **hardcoded production IP as a source-code fallback** (not just an example/placeholder — it's live code that will actually hit that IP if `VITE_API_URL` isn't set in the build env).
- Response is `{success, predicted_class, class_id}`; the browser compares `String(result.predicted_class)` to the expected folder-ID string from its own local `WORD_TO_ID_MAP` and sets `feedback` to `"correct"`/`"incorrect"` accordingly.

**FastAPI (`src/backend/app.py`), all inference happens here:**
1. Save the uploaded blob to a temp `.mp4` file (note: browsers send `.webm`, but it's written with `suffix=".mp4"` — works because `cv2.VideoCapture`/ffmpeg sniff the container rather than trust the extension, but it's a latent footgun).
2. `preprocess_video()`: read every frame into memory (`cv2.VideoCapture` loop), pick 32 evenly-spaced frame indices, run **MediaPipe Holistic once per sampled frame, sequentially, on CPU** (`device = torch.device("cpu")`; per `SETUP.md` the backend runs on an AWS Lightsail box with 2 vCPUs / 2 GB RAM — no GPU).
3. Stack into a `(1, 3, 32, 47)` tensor, run `PoseCNN_LSTM_Attn` forward pass (small model, negligible cost next to step 2).
4. Mask logits to the 31 `perfect_mapped_classes`, `argmax`, map through `reverse_label_map`.

**Where the ~5s comes from:** almost certainly **step 2** — 32 sequential synchronous MediaPipe Holistic `.process()` calls (each doing hand + pose + face-adjacent detection internally) on a 2-vCPU CPU-only box, with no batching/threading/GPU. This is the same architecture as the notebook's own preprocessing, which is why the notebook needed a retry/backoff loop just to open files reliably — Holistic-per-frame on CPU is the known slow part of this whole pipeline, and it happens once per prediction request in real time (not precomputed). The one-time `wakeUpBackend()` fetch to `/` on component mount is unrelated to this — Lightsail is an always-on VM, not a cold-start serverless platform, so that ping is not the source of the delay.

---

## 6. What the user sees

Frontend feedback states are exactly `"idle" | "processing" | "correct" | "incorrect"` (`LearningCard.tsx` — `FeedbackState` type). There is **no fourth state for "body/hands not detected."**

- **Correct:** green overlay, `CheckCircle`, "Perfect!"
- **Incorrect:** red overlay, `XCircle`, "Try again!"
- **"Body not detected":** does not exist as a distinct UX state. If MediaPipe fails to find hands/pose in some or all of the 32 sampled frames, those joint rows are silently zero-filled (§3) and fed to the model like any other input — the model still produces *some* class via `argmax` over the 31 allowed classes, and the user just sees a generic **"Try again!"** (incorrect) with no indication that the real problem was framing/lighting/camera rather than getting the sign wrong. The only way an error surfaces distinctly is if the fetch itself throws (network/CORS/5xx) — that's caught and *also* mapped straight to `"incorrect"` (`catch { setFeedback("incorrect") }`), so a backend outage looks identical to a wrong sign to the end user.

---

## 7. License / terms shipped with data vs. `ATTRIBUTION.md` claims

- `LICENSE` (repo root) is **MIT**, copyright Jonathan Reyes — this covers the *code*, not the data.
- `data/DATA_README.md` states the raw video data is **CC BY-NC 4.0** (non-commercial), citing the same Yang et al. MMSys/MMM 2020 paper.
- `ATTRIBUTION.md` §1.A states the same: KAIST/Samsung-affiliated dataset, **CC BY-NC 4.0**, with the required citation.
- These are consistent with each other. The thing worth flagging: **the MIT LICENSE at the repo root makes no mention of the CC BY-NC 4.0 data restriction**, and the shipped model weights (`src/backend/best_model.pt`) and the `.pkl` feature file are derived directly from that NC-licensed data. Anyone cloning the repo under the "MIT license" impression could reasonably miss that the trained model / dataset artifacts are NC-restricted and not MIT-covered — this is only caught by separately reading `DATA_README.md`/`ATTRIBUTION.md`. **Fixed in PR 1:** `LICENSE` and `README.md` now both carry an explicit note that the model weights, `.pkl`, and demo videos are CC BY-NC 4.0-derived and not MIT-covered, and the in-app attribution under the practice video now says "Clips trimmed from the original source videos" instead of just "Video source."
- `ATTRIBUTION.md` §5 also discloses Gemini 3 was used to draft frontend/backend code, the notebook preprocessing, and the docs themselves (self-disclosed, not something I could independently verify, but worth knowing when reading "why is this code shaped like this" — some of the inconsistencies below (§9) are plausibly explained by that provenance).

---

## 8. Most-confused class pairs

A confusion-matrix PNG exists (`images/confusion_matrix.png`) but reading exact off-diagonal cell values from a 67×67 heatmap image is unreliable, so instead of eyeballing it I **reproduced the notebook's exact evaluation from scratch**: loaded `data/KSL77_joint_stream_47pt.pkl`, rebuilt the same `train_test_split(..., random_state=42)` / label re-indexing the notebook uses, loaded `src/backend/best_model.pt` into `src/backend/model.py`'s architecture, and ran inference on the held-out test set. This reproduced **88.21% test accuracy exactly**, confirming `best_model.pt` is indeed the tuned model from the notebook and the test split is reproducible.

Top confused pairs by count (true → predicted, with English glosses via `class_label.p`; test set is tiny — 3–4 samples/class — so treat exact ranking past the top few as noisy):

| Count | True | Predicted |
|---|---|---|
| 2 | thank | finally |
| 2 | walk | ride |
| 1 | what | hobby |
| 1 | bi bim rice | recieve |
| 1 | bi bim rice | invite |
| 1 | movie | thank |
| 1 | face | good, nice |
| 1 | see | effort |
| 1 | thank | sorry |
| 1 | sorry | thank |

(17 more count-1 pairs exist; total 27 distinct confused pairs covering 29 of 246 test predictions — 88.21% accuracy checks out: 217/246 correct.) Notable: `thank`↔`sorry` and `sorry`↔`thank` both appear (symmetric confusion), and `bi bim rice` is confused with two different words, consistent with it being one of the harder/lower-support classes.

---

## 9. Extra finding not asked for directly, but blocking a clean refactor: `models/` is dead, broken code

Since the notebook-extraction plan below assumes `models/` as a starting point, flagging this now: **`models/` (`config.py`, `model_architecture.py`, `model_utils.py`) is never imported by anything that actually runs.** `grep` for `from models` / `import models` across the repo's Python files returns nothing; `src/backend/app.py` has its own copies of everything inline instead. And the two copies have diverged in a way that matters:

- `models/model_architecture.py`'s `PoseCNN_LSTM_Attn` uses 128-channel conv/LSTM layers (the *original*, pre-Optuna-tuning architecture from notebook Cell 7). Loading `best_model.pt` into it **fails with a shape-mismatch error** (verified directly — `conv2.weight` expects `[128,64,1,3]`, checkpoint has `[64,64,1,3]`, etc., 20+ mismatched tensors). The actually-trained/deployed model is the 64-channel "tuned" architecture in `src/backend/model.py`, which does load successfully and reproduces 88.21% test accuracy.
- `models/config.py`'s `perfect_mapped_classes`/`reverse_label_map` (31 entries) is a **different, incorrect** subset — 7 of its "perfect" classes don't actually have F1=1.00, and it's missing 7 that do. `src/backend/app.py`'s list is the correct one (verified: it's exactly the F1=1.00 set from re-running the eval).

Net effect: if a future refactor starts from `models/` as "the clean version to build scripts from," it will silently produce a broken model definition and a wrong prediction mask. The source of truth for architecture + masking is `src/backend/model.py` + `src/backend/app.py`, not `models/`.

---

## Proposed plan: extract `notebook/KSL.ipynb` into scripts (executed in PR 2, see below — kept as the original design record)

Goal: turn the notebook into `extract_landmarks.py`, `dataset.py`, `models/`, `train.py`, `evaluate.py`, `export_onnx.py`, using `src/backend/model.py` + `src/backend/app.py` as the source of truth (not the current `models/` package, which should be replaced, not extended).

1. **`extract_landmarks.py`** — Cells 1–5 (`BASE_INPUT_DIR` walk, `mp_holistic` setup, `extract_landmarks_from_frame`, `preprocess_video` with its retry/backoff, `process_all_videos`). Add a `--signer-regex`/manifest option so signer ID (from filename or a sidecar CSV) is captured and written alongside `features`/`labels` into the output `.pkl` or a companion `manifest.csv` — this is the prerequisite for ever fixing §4's leakage risk, since right now that identity is destroyed at exactly this step.
2. **`dataset.py`** — `PoseDataset` (Cell 6) plus the split logic, refactored to support **group-aware splitting** (e.g. `sklearn.model_selection.GroupShuffleSplit` keyed on signer ID once available) as an option alongside the current stratified-only split, so the two can be compared.
3. **`models/`** (fresh, replacing the current stale package) — one canonical `PoseCNN_LSTM_Attn` module built from `src/backend/model.py` (the verified-correct 64/256-channel architecture), parameterized like the notebook's `PoseCNN_LSTM_Attn_Tunable` (Cell 7) so Optuna tuning stays possible; delete the current broken `models/config.py` / `models/model_architecture.py` / `models/model_utils.py` rather than merging them in, since they don't match the deployed weights.
4. **`train.py`** — Cells 7–8 (`Optuna` objective + best-params training loop), parameterized via CLI/config instead of notebook globals (`study`, `best_params` living in kernel memory); write the label re-indexing map (`label_map`/`reverse_label_map`) out as a checked-in JSON artifact next to the checkpoint, so `app.py` and any retrain always derive it from the same source instead of three hand-copied literals (§1).
5. **`evaluate.py`** — Cell 9 (classification report, confusion matrix), taking a checkpoint + split as input so it can be re-run against both the current stratified split and a future grouped split for direct comparison, and re-deriving `perfect_mapped_classes` programmatically (F1==1.00 filter) rather than hand-copying it — this removes the exact class of bug found in §9/§1.
6. **`export_onnx.py`** — new, not in the notebook today: load a trained checkpoint via the shared `models/` module and `torch.onnx.export` it, as a first step toward moving inference off a synchronous per-request Python/CPU MediaPipe loop (§5) — e.g. onward to a faster runtime, batching, or moving MediaPipe extraction client-side/WASM to cut the ~5s round trip.

Suggested order if this is picked up: (1) → (3) first since everything else depends on a correct, shared model definition; (5)'s programmatic `perfect_mapped_classes` derivation is the highest-value/lowest-risk piece since it directly removes a live correctness bug (§9) without touching training.

---

## PR 1 — hygiene and honest error states (applied)

Small, reviewable fixes made after this audit, addressing the findings above directly:

1. Renamed `src/frontend/public/videos/how many_example2.MP4` → `how many_example2.mp4` (§2). Added `src/frontend/scripts/check-videos.mjs`, wired as `npm run check:videos` and `npm run prebuild`, which parses `WORD_LIST` out of `Learn.tsx` and fails if any practiceable word's `_example1`/`_example2` video is missing or case-mismatched in `public/videos`.
2. Removed the hardcoded fallback IP (`34.239.230.9`) from `LearningCard.tsx`. `getApiUrl()` now throws immediately (not just logged/swallowed) if `VITE_API_URL` isn't set, instead of silently falling back to a stale production IP that didn't even match the real prod backend (`.env.production` points at `koala-sign-learn.onrender.com`, not that IP).
3. `src/backend/app.py`: `preprocess_video` now counts, per sampled frame, whether pose / left hand / right hand were detected, and returns those counts. `/predict` short-circuits to `{"success": false, "reason": "not_detected", "detection": {...}}` (no model call) when fewer than 60% of the 32 sampled frames have both a detected pose and at least one detected hand. Successful responses now also include `top3`: the top-3 classes (from the same `perfect_mapped_classes`-masked logits) with softmax probabilities.
4. `LearningCard.tsx`: `FeedbackState` gained `"not_detected"` and `"error"`. `not_detected` shows framing guidance ("Make sure both hands and shoulders are in frame..."); `error` (network failure, non-2xx response, or any other `success: false`) shows "Couldn't reach the server — this isn't your signing." Neither path calls `onFeedback(...)`, so neither is recorded as a wrong attempt by the SM-2 spaced-repetition logic in `Learn.tsx`.
5. Deleted the `models/` package entirely (`config.py`, `model_architecture.py`, `model_utils.py`, and its byte-identical duplicate `best_model.pt`) per §9 — confirmed dead (unimported) and inconsistent with the deployed weights before removal. `src/backend/best_model.pt` (the real, deployed weight file) is untouched.
6. `README.md` updated: now states 67 trained classes (with 77 as the raw KSL folder-ID space, not a second class count), 31 practiceable words in the current app, and an explicit caveat on the 88.21% test accuracy figure (random non-signer-grouped split, 3–4 test samples/class — treat as an upper bound).
7. Added a "this does not cover the data" note to both `LICENSE` and `README.md`'s License section, naming the specific CC BY-NC 4.0-derived artifacts (`best_model.pt`, the `.pkl`, the demo videos). Changed the in-app attribution line under the practice video from "Video source: ..." to "Clips trimmed from the original source videos: ...".
8. Confirmed and listed the exact set of unreachable demo-video words in §2 above: **18** words (not deleted yet), not 19 — `care`, `effort`, `experience`, `finally`, `fine`, `good`, `introduction`, `invite`, `receive`, `see`, `sorry`, `special`, `test`, `thank`, `time`, `want`, `who`, `yesterday`.

**Not done in this pass** (left for a follow-up PR, on purpose — out of scope for "hygiene and honest error states"): the `models/` replacement package + `extract_landmarks.py`/`dataset.py`/`train.py`/`evaluate.py`/`export_onnx.py` extraction, group-aware (signer-grouped) re-splitting, and deleting the 18 unreachable demo videos.

---

## PR 2 — extract `notebook/KSL.ipynb` into scripts with signer-aware evaluation (applied)

Adds `extract_landmarks.py`, `models/` (fresh), `dataset.py`, `train.py`, `evaluate.py`, `RESULTS.md`, `scripts/smoke_test_colab.py`, root-level `requirements.txt`, and a pointer note at the top of `notebook/KSL.ipynb`. Real KSL-77 raw video and Google Drive access aren't reachable from this environment, so **no MediaPipe extraction was actually run against real footage** — everything below was verified as far as it can be without that access; see "What's been verified so far" in `RESULTS.md` for the exact list.

- **`extract_landmarks.py`**: uses the MediaPipe **Tasks API**, preferring `HolisticLandmarker` (confirmed present in the locally-installed `mediapipe` package's `tasks.python.vision` module) with a `--backend hand_pose` fallback (`HandLandmarker(num_hands=2)` + `PoseLandmarker`). Keeps the 47-point layout (left hand 0–20, right hand 21–41, pose `[0,11,12,13,14]` at 42–46) and 32-frame `np.linspace` sampling, but samples frames by seeking (`cv2.CAP_PROP_POS_FRAMES`) instead of reading the whole video into memory first. Adds a genuine per-joint detection mask (`mask.npy`, 1/0) instead of the old silent zero-fill. All result-field access is isolated in one function, `result_to_47pt()`, specifically so a MediaPipe version difference only needs a fix there.
  - Signer id + class id come from `<NN>_<class_id>.mp4` (case-insensitive), matched against the parent folder name; a mismatch is logged to `mismatches.csv` and the file is skipped by default, or raises immediately under `--strict`.
  - Resumable via a per-video cache (`out/cache/*.npz`, keyed by class/signer/filename): a second run against the same `--out` does zero MediaPipe work for already-processed videos (verified — see below).
  - `--device-check` prints platform + mediapipe version and warns explicitly if run on macOS.
  - **Local MediaPipe Tasks execution is broken on this development machine independent of this script**: `HolisticLandmarker.detect()` (and standalone `HandLandmarker`/`PoseLandmarker`) abort with `F... Check failed: service_ Service is unavailable` from a Metal-backed calculator (`DrishtiMetalHelper`), reproduced with and without `BaseOptions.Delegate.CPU` explicitly set, on a single blank still frame. This is a known class of MediaPipe-on-macOS issue, not a bug in this repo. Everything not dependent on that native call *was* verified directly: `discover_videos()`'s regex/mismatch-logging (including a synthetic mismatch), the wrist-proximity hand-assignment heuristic in `assign_hands_by_wrist()` (proven to override a deliberately-wrong `handedness` label when pose is available, and to fall back to the label correctly when it isn't), and the full `main()` pipeline's caching/resume/manifest/`.npy` assembly against real (tiny, synthetic) `.mp4` files with a monkeypatched fake detector standing in for MediaPipe. `scripts/smoke_test_colab.py` exists to run the real thing on Colab before committing to a full extraction.
- **`models/`**: rebuilt as a small package (`models/pose_cnn_lstm_attn.py` + `models/__init__.py`) with the tunable 64-channel/256-hidden `PoseCNN_LSTM_Attn` from `src/backend/model.py` (not the deleted, broken 128-channel version). Verified to load `src/backend/best_model.pt` with no shape mismatch and reproduce its forward pass.
- **`dataset.py`**: `random` mode reproduces `notebook/KSL.ipynb`'s exact split (verified: `dataset.random_split(seed=42)` + `src/backend/best_model.pt` reproduces 88.21% test accuracy exactly, bit-for-bit the same as AUDIT.md §8's earlier reproduction). `signer_kfold` mode groups by `signer_id` with `GroupKFold(n_splits=5)`, then carves a validation *signer* subset out of each fold's training signers with `GroupShuffleSplit` — verified on a synthetic 20-signer/10-class dataset that every signer lands in exactly one test fold and train/val/test signer sets never overlap within a fold, both via the sanity check passing on a correct partition and correctly raising `AssertionError` when a violation was deliberately injected.
- **`train.py`** / **`evaluate.py`**: CLI-driven, save `label_map.json` + `config.json` per fold, report top-1/top-3/macro-F1/per-class-F1/per-signer accuracy, and mean±std across folds for `signer_kfold`. `evaluate.py` re-derives each fold's split independently from `config.json` (not just trusting train.py's saved indices) and re-runs `check_signer_partition()` itself before reporting anything, per spec. Both were run end-to-end: `train.py`+`evaluate.py` in `random` mode against the real `data/KSL77_joint_stream_47pt.pkl`, and in `signer_kfold` mode (5 folds) against a synthetic class-separable manifest dataset — which caught one real bug, now fixed: `BatchNorm1d` in the classifier head raised on a trailing training batch of size 1 (fixed via `drop_last=True` on the training `DataLoader`, with a clear error if that would leave zero batches for a very small fold).
- **`RESULTS.md`**: table comparing (a) old checkpoint / random split — filled in with real, reproduced numbers (88.21% top-1, 95.53% top-3, 0.8806 macro-F1) — against (b) retrained / random split and (c) signer-out 5-fold, both left explicitly `*pending*` with the exact commands to fill them in on Colab. No number is described as an improvement anywhere, since (c) — the only row that actually answers the signer-generalization question from §4 — hasn't been run.
- `notebook/KSL.ipynb` is kept, with a new markdown cell at the top pointing to the scripts above and explaining which cells each one replaces.

**Not done in this pass**: actually running extraction/training against the real KSL-77 raw videos (needs Colab + Drive access — see `RESULTS.md`'s pending rows), and `export_onnx.py` (was in the original proposed plan above but wasn't requested for PR 2).

### PR 2 follow-up: Colab smoke test results + fixes

`scripts/smoke_test_colab.py` has since been run for real on Colab (mediapipe 0.10.35, Python 3.13) against real KSL-77 clips — the thing this repo's local macOS environment can't do (see above). It passed: `HolisticLandmarker` loaded, the result exposed exactly the fields `result_to_47pt()` was written against (`left_hand_landmarks`, `right_hand_landmarks`, `pose_landmarks`, etc.), and features/mask came out as `(3, 32, 47)` / `(32, 47)` with no NaNs. Per-clip hand detection ranged 69–97%, pose 100%. Follow-up fixes made in response:

- `requirements.txt` pinned to the version actually verified, `mediapipe==0.10.35` (not the notebook's older `0.10.21`), with a note that PR 3's browser-side `@mediapipe/tasks-vision` must track the same version/model or the features the model sees at inference (browser) won't match training (this file).
- `Detector` now has `close()` + `__enter__`/`__exit__`, and `extract_landmarks.main()` / `smoke_test_colab.py` both use it (try/finally or `with`) instead of relying on MediaPipe Tasks objects' `__del__` at interpreter shutdown, which raised a spurious `TypeError` on exit.
- `extract_single_video()`'s stats now include `frame_width`, `frame_height`, and separate `frac_frames_with_left_hand`/`frac_frames_with_right_hand` (previously only a combined "any hand" fraction) — printed per-video by `smoke_test_colab.py`.
- The Colab run also surfaced a MediaPipe warning on non-square clips: `Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI.` Investigated as far as possible without a real MediaPipe run: `mp.Image(...)` is built directly from each frame's own array (so its real, possibly non-square, width/height are already attached at the top level — this isn't us handing MediaPipe a mis-shaped image), and a synthetic-video test in this environment confirmed our own pixel-mapping code (`x * frame_width`, `y * frame_height`) places points correctly on a 100×60 non-square frame with no distortion. Whether MediaPipe's own internal ROI-cropping calculator (the thing actually emitting the warning) still produces correctly-aligned landmarks on a non-square frame is a question only a real video can answer, so `smoke_test_colab.py` gained `--save-overlay DIR`: it renders our `(3, 32, 47)` array (not the raw MediaPipe result — so it also checks our own left/right/pose index layout) back onto frames 0/10/20/31 of each test clip, so this can be checked visually before committing to a full extraction run.

### PR 2 follow-up: corrupted video handling

The full Colab extraction run hit a corrupted raw video ("moov atom not found" — decodes zero frames). `extract_landmarks.py` correctly recorded it as `n_sampled_ok=0`/`frac_frames_with_pose=0` in `manifest.csv`, but nothing was dropping that row before training/evaluation, so it would have been used as a real (all-zero) training and test example. Fixed:

- `manifest.csv` now also includes `n_sampled_ok`, `frac_frames_with_left_hand`, `frac_frames_with_right_hand` (previously computed internally but not written out).
- `dataset.py` gained `drop_undetected_samples()`, called automatically from `load_manifest_dataset()`: drops any row with `n_sampled_ok == 0` or `frac_frames_with_pose == 0` (and, opt-in via `--min-hand-frac`, rows below a hand-detection threshold), filtering `features`/`mask`/`manifest` together so row alignment is preserved, and printing exactly which `video_path`s were dropped and why. Verified against a synthetic manifest with a deliberately corrupted row and a deliberately pose-less row — both dropped, arrays stayed aligned.
- `train.py` now has `--min-hand-frac` (default `0.0`) and records it in each fold's `config.json`; `evaluate.py` reads it back automatically (rather than requiring the user to remember to pass a matching flag) so it reconstructs the exact same filtered dataset the checkpoint's split was built from, and refuses to run if an explicitly-passed `--min-hand-frac` doesn't match — verified both the automatic-match and the refusal-on-mismatch paths. See `RESULTS.md`'s "Known real-world data issue" section for details.

### PR 2 follow-up: real Colab results, per-signer diagnosis, and 4 new feature transforms

Real numbers came back from Colab: (a) old checkpoint/random split 88.21% top-1, (b) retrained on new Tasks features/random split 83.74%, **(c) signer-out 5-fold 78.22% ± 7.56%** — now the headline metric in `RESULTS.md`, since it's the only row that actually measures the signer-leakage risk flagged in §4 (the (b)→(c) gap, ~5.5 points, is that risk now measured rather than theoretical). Per-signer diagnosis identified three outlier signers (08: opposite-handed/mirrored; 14: framing/nose position; 18: farthest-from-camera + unusually short, tightly-trimmed clips) — excluding signer 08 alone brings signer-out top-1 to ~81%.

Added `transforms.py` (new module, not part of the original PR 2 scope) with `mirror_clip`, `normalize_body`, `trim_idle` — pure-numpy, unit-tested (`test_transforms.py`, 9 tests) — directly targeting those three failure modes, wired into `train.py`/`evaluate.py` as `--mirror-aug`/`--tta-mirror`, `--normalize-body`, `--trim-idle`/`--trim-idle-motion-threshold`. The two deterministic transforms (`--normalize-body`, `--trim-idle`) are recorded in `config.json` and auto-resolved by `evaluate.py` the same way `--min-hand-frac` already was (refusing on a mismatched override) — verified end-to-end, including the legacy-`.pkl` fallback path (no real mask, so a joint's "detected" state is inferred as "not exactly (0,0,0)"). `evaluate.py` also gained a per-signer accuracy table marking which fold each signer was tested in. New `scripts/compare_hand_detection.py` (numpy-only) compares legacy-vs-Tasks-API frame-level hand/pose detection rates — run against the real legacy `.pkl`: 44.75% left hand, 58.97% right hand, 66.29% any hand, 99.86% pose, as the baseline to compare a real Colab `mask.npy` against. `RESULTS.md` gained placeholder ablation rows (b')/(c') for these transforms, with the same "only counts as an improvement if it improves (c'), not (b')" rule applied.

### PR 2 follow-up: ablation results, production recipe, and PR 3 handoff (ONNX + preprocessing spec + golden vectors)

Real ablation numbers came back from Colab across 6 configs (baseline, TTA-only, `--mirror-aug`, `--normalize-body`, `--mirror-aug --normalize-body`, and that plus `--trim-idle`), plus a 3-seed check on the best one. **`RESULTS.md` now has a real headline: config f (`--mirror-aug --normalize-body` at train, `--tta-mirror` at eval), 3-seed mean 83.2% top-1 / 94.1% top-3 on unseen signers** — up from baseline (c)'s 78.22%. `--trim-idle` was measured and dropped (no gain beyond noise, made signer 14 worse, extra porting cost for PR 3 with no payoff) — all reasoning recorded in `RESULTS.md`'s new "Decisions" section, including the honest caveat that config f was chosen by looking at all 6 configs' signer-out numbers, so 83.2% carries some selection optimism.

To get a real deployment checkpoint out of this: `train.py` gained `--split-mode full` (train on all 20 signers, no validation split, no early stopping) and `--stop-epoch N` (the cosine scheduler is still built with `T_max=--epochs`, matching the CV runs' LR schedule, but training stops after `N` epochs — so `--stop-epoch` picks a point on the *same* schedule the CV folds used, it doesn't reshape it). `--stop-epoch 34` was chosen as the median best-epoch (33, zero-indexed) across the 15 config-f CV folds (3 seeds × 5 folds). Every mode's `config.json` now also records `best_epoch` (previously only `best_val_acc` was saved — you couldn't tell *when* the best epoch happened without this). Verified with a dedicated test file, `test_train_full_and_export.py`: `--split-mode full` requires `--stop-epoch`, trains exactly that many epochs while `config.json`'s `scheduler_t_max` still equals `--epochs` (not `--stop-epoch`), and the resulting checkpoint loads back into `PoseCNN_LSTM_Attn` with no shape mismatch; `best_epoch` was also confirmed to land correctly for the pre-existing `random`/`signer_kfold` paths.

New `export_onnx.py`: exports a checkpoint to ONNX (opset 17, `(N, 3, 32, 47)` float32 input with a dynamic batch axis, raw logits output — not softmax), then always runs a PyTorch-vs-onnxruntime parity check (random noise, `[0,1]`-uniform "real-shaped" input, at three different batch sizes to exercise the dynamic axis) and deletes the exported file if any input exceeds `1e-4` max-abs-diff rather than leaving a silently-wrong `.onnx` in place. Verified end-to-end (including the ambiguous-multi-fold-checkpoint-dir error path) in `test_train_full_and_export.py` — all parity checks passed with max abs diff on the order of `1e-7`–`1e-8`, well under tolerance.

New `preprocessing_spec.json` (machine-readable) + `docs/PREPROCESSING.md` (human-readable) specify exactly what PR 3's browser port must do before calling the ONNX model: the 47-point layout/index order, the exact frame-sampling formula (`floor`, not round — verified against `numpy.linspace(..., dtype=int)`'s actual truncation behavior, not assumed), mask semantics, `normalize_body`'s exact math (reference frames, mid-shoulder point, shoulder width, the missing-joints-stay-exactly-zero rule), and mirror TTA (the reflection/swap transform, its fallback chain, and — the easy mistake to port wrong — that the two forward passes are combined by averaging *softmax probabilities*, not raw logits). Explicitly documents that `trim_idle` is **not** part of the production recipe, so PR 3 doesn't need to port it. New `scripts/generate_golden_vectors.py --checkpoint-dir DIR` produces 3 real clips' raw landmarks/mask → normalized landmarks → logits (original + mirrored) → TTA-averaged probabilities, computed with the real trained model, for the TypeScript port to diff against — verified to run correctly against a toy checkpoint (both the legacy-`.pkl`-with-inferred-mask path and the manifest-with-real-mask path), though real golden numbers require the actual config-f production checkpoint, which hasn't been trained yet in this environment (see Colab commands below).
