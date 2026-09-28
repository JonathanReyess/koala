# PERF.md — in-browser inference (PR 3)

## Versions

| Piece | Version | Notes |
|---|---|---|
| Python extraction (training) | `mediapipe==0.10.35` | `requirements.txt` |
| `@mediapipe/tasks-vision` | **0.10.35** (exact pin) | npm has an exact `0.10.35`, so no "closest" approximation was needed |
| HolisticLandmarker model | `holistic_landmarker.task` float16, sha256 `e2dab611…50c3f8` | Same "latest" URL `extract_landmarks.py` downloads; committed to `public/models/` so it is served same-origin |
| `onnxruntime-web` | **1.19.2** (exact pin) | Matches the `onnxruntime==1.19.2` used to verify `model.onnx` in Python |
| Model | `public/models/ksl_f/model.onnx`, opset 17, input `input` (N,3,32,47) f32, output `logits` (N,67) | names verified with onnxruntime |

Both runtimes are single-threaded WASM (no COOP/COEP headers needed); WebGPU is opt-in via `?webgpu=1`.
MediaPipe uses the **GPU delegate by default with automatic CPU (XNNPACK) fallback** (creation failure or a failed
warm-up probe → CPU). Python extraction used CPU. Override with `?delegate=cpu` / `?delegate=gpu`; the delegate
actually in use is logged in the `[koala] models loaded` line.

## Measured numbers

**Setup:** MacBook (Apple M4, 16 GB), Chrome 154, `vite build` + `vite preview` on localhost (no network
latency, no compression) for load times; dev server for live runs. Driven with puppeteer-core in headless
Chrome using Chrome's fake camera fed with a real signing clip (`public/videos/me_example1.mp4` as y4m).
**These are not from a hand-held, interactive Chrome session with a webcam** — see "Reproduce" below to get
your own numbers (the fps badge shows in dev, or with `?perf=1`).

| Metric | Result |
|---|---|
| Model + landmarker load, **cold** (HTTP cache off; 48.7 MB transferred, uncompressed localhost) | ONNX session ≈ 1.8 s, landmarker ≈ 1.9 s (in parallel) → **≈ 1.9 s** to ready (2.2 s to interactive page) |
| Model + landmarker load, **warm** (HTTP cache) | ONNX ≈ 0.6 s, landmarker ≈ 0.7 s → **≈ 0.7 s** (0.8 s to interactive page) |
| Re-entering Practice in the same session | 0 s (promises are memoised; nothing is refetched) |
| Landmarker fps, live camera, person in frame — see delegate table below | CPU **≈ 11.6 fps**, GPU **≈ 17.9 fps** (camera delivers 30 fps; ~30 fps when nobody is in frame) |
| Stop → result (live recording; 32-frame select + normalize + batch-2 ONNX + softmax/grade) | **≈ 110–135 ms** (ONNX batch-2 inference itself 23–44 ms) |
| Stop → result, uploaded clip (VIDEO mode over every frame, ~4 s clip) | ≈ 7 s (seek + detect every frame at 30 fps assumed; dominated by landmark extraction, not the model) |

Because landmarks are computed live while recording, Stop only has to do the cheap part, which is why
Stop→result is ~0.1 s. The cost of the heavy part is the ~10–12 fps landmarker (≈ 60–75 recorded frames per
6 s clip, of which 32 are sampled per the spec).

Uploads are slower than live because every frame is decoded, seeked and detected sequentially. Clips much
longer than ~10 s will take proportionally longer.

Download size (first visit): `model.onnx` 15 MB, `holistic_landmarker.task` 13.7 MB, MediaPipe WASM 11 MB,
ORT WASM 11 MB, JS ≈ 1 MB.

## MediaPipe delegate: GPU vs CPU

Headed Chrome 154 on the same Mac, fake camera fed with a signing clip, live `HolisticLandmarker` in VIDEO
mode, fps badge sampled once a second for 12 s (`?delegate=cpu|gpu`):

| Delegate | fps median (min–max) | Landmarker init | Stop → result (3 recordings) |
|---|---|---|---|
| CPU (XNNPACK) | **11.6** (10.7–11.7) | ≈ 1.2 s | 131 / 112 / 114 ms |
| GPU (WebGL) | **17.9** (15.7–20.1) | **≈ 1.4–1.5 s** once the GPU shader cache is warm; **17.6 s the very first time on this machine** | 112 / 97 / 71 ms |

- GPU is ≈ 55% faster live. Grading pipeline is unchanged (same landmarks → same preprocessing → same model).
- **Predictions unchanged:** running the 98 example clips through the GPU delegate gives the same top-1 as CPU on
  98/98 clips (both 98/98 correct in VIDEO mode). Not bit-identical landmarks, so a borderline clip could still
  flip; it did not on this set.
- **First-run cost:** the first GPU init on a machine compiles shaders (17.6 s here, then cached by the OS/browser).
  New visitors will see the "Getting the sign checker ready…" state for that long once. If that matters more than
  the fps, set `landmarkerDelegate: "CPU"` in `src/lib/inference/config.ts`. I could not reproduce a truly cold
  cache on demand, so treat 17.6 s as one observation, not a distribution.
- Fallback: if GPU creation (or a warm-up probe) throws, it falls back to CPU with a console warning. I could not
  test a machine without WebGL2 here, so the fallback path is exercised only by code review.

## IMAGE vs VIDEO mode parity (`/debug/parity`, dev server only)

Python extraction ran HolisticLandmarker in **IMAGE** mode on the 32 sampled frames; the browser uses
**VIDEO** mode (tracking). I ran both modes on all 98 bundled example clips (2 per practice word):

| | top-1 = expected word |
|---|---|
| VIDEO mode (live path) | **98 / 98** |
| IMAGE mode (browser, seek + detect per sampled frame) | 81 / 98 |

Landmarks differ substantially between modes: mean per-joint difference after `normalize_body`
(units = shoulder widths) has median **0.39** across clips (mean over all joints, both modes detecting), and the two modes agree on
whether a joint is detected in ≈ 86% of sampled frame×joint pairs on average.

Takeaways:
- Mode **does** matter for the raw landmarks, but on these clips the VIDEO-mode landmarks give *better*
  predictions than IMAGE-mode ones, so the live path is not a regression risk on this evidence.
- Uploaded clips therefore use the same VIDEO-mode extraction as the live camera (one pipeline).
- Caveat: these example clips are trimmed from the KSL-77 source videos the model was trained on, so this is
  not a held-out accuracy number; and the browser's IMAGE mode is not guaranteed identical to Python's IMAGE
  mode (different build/delegate), so the 81/98 says nothing certain about the Python features.

## Reproduce

```bash
cd src/frontend
npm install && npm run dev            # http://localhost:8080/learn  (fps badge bottom-left of camera view)
# open DevTools console: "[koala] models loaded {...}" and "[koala:perf] Stop→result … ms" lines
# parity tool: http://localhost:8080/debug/parity  (dev only) — upload a clip
```
