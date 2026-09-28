# Browser preprocessing spec (for PR 3)

This is the human-readable version of [`preprocessing_spec.json`](../preprocessing_spec.json)
(the machine-readable source of truth) — what a TypeScript/browser port must
do, in order, before calling the exported ONNX model
(`export_onnx.py`). Every step here has a Python reference implementation
you can diff against: [`extract_landmarks.py`](../extract_landmarks.py)
(landmark extraction, frame sampling) and [`transforms.py`](../transforms.py)
(`normalize_body`, `mirror_clip`). Test your TypeScript port against
`scripts/generate_golden_vectors.py`'s output — see the bottom of this doc.

**Production recipe is "config f"** (see `RESULTS.md`): `normalize_body`
always applied, `mirror_aug` was a *training-time-only* augmentation (the
browser never needs to reproduce it), and `--tta-mirror` (average softmax
over the original clip and its mirror) applied at inference. `trim_idle` is
**not** part of the production recipe — don't implement it for PR 3.

## 1. The 47-point layout

| Indices | Group | Notes |
|---|---|---|
| 0–20 | Left hand | MediaPipe Hand landmarks, MediaPipe's own order. Index 0 (within this block) = wrist. |
| 21–41 | Right hand | Same, for the right hand. Index 0 (within this block, i.e. absolute index 21) = wrist. |
| 42 | Nose | BlazePose landmark 0 |
| 43 | Left shoulder | BlazePose landmark 11 |
| 44 | Right shoulder | BlazePose landmark 12 |
| 45 | Left elbow | BlazePose landmark 13 |
| 46 | Right elbow | BlazePose landmark 14 |

Coordinates are MediaPipe's own normalized landmark coordinates: x, y in
`[0, 1]` relative to the input frame's width/height; z is MediaPipe's own
relative-depth convention. These are **not** rescaled to any fixed physical
unit before `normalize_body` (§3) runs.

## 2. Frame sampling

32 frames, evenly spaced across the whole clip, **including the first and
last frame**:

```
index[i] = floor( i * (total_frames - 1) / 31 )   for i = 0 .. 31
```

**The rounding is truncation toward zero, not round-to-nearest.** This
matches `numpy.linspace(0, total_frames - 1, 32, dtype=int)`, which computes
the float linspace and then truncates on cast — e.g.
`np.linspace(0, 2, 5, dtype=int) == [0, 0, 1, 1, 2]`, not `[0, 1, 1, 2, 2]`.
If you implement this with `Math.round` instead of `Math.floor` in
TypeScript, you will pick a different (adjacent) frame on roughly half of
the 32 samples for a typical clip length — check against the golden vectors.

## 3. `normalize_body` (always applied, in the production recipe)

Applied once per clip, to the raw (pre-sampling-adjacent, but post-frame-
sampling — i.e., to the 32 already-sampled frames) landmarks, **before**
anything else (before mirroring, before feeding the model).

1. **Reference frames**: only the sampled frames where *both* shoulders
   (indices 43 and 44) are detected (mask == 1 for both) count toward the
   two statistics below. If no frame in the clip has both shoulders
   detected, `normalize_body` is a no-op — the clip's landmarks pass through
   unchanged (there's no body frame to normalize into).
2. **Mid-shoulder point**: the mean, over reference frames only, of
   `(left_shoulder_xyz + right_shoulder_xyz) / 2` — one `(x, y, z)` 3-vector
   for the whole clip (not per-frame). Subtract this from every joint's
   `(x, y, z)` in **every** frame of the clip (not just reference frames).
3. **Shoulder width**: the mean, over reference frames only, of the 3D
   Euclidean distance between the two shoulders in that frame — one scalar
   for the whole clip. Divide every joint's `(x, y, z)` (after the
   subtraction above) by this scalar, with `max(shoulder_width, 1e-6)` to
   guard divide-by-zero. All three axes are divided by the *same* scalar —
   this is not independently normalizing the x/y/z ranges.
4. **Missing joints stay exactly zero.** After the above, force any
   `(frame, joint)` with `mask == 0` back to exactly `(0.0, 0.0, 0.0)` — do
   not leave it at `-mid_shoulder / shoulder_width`, which would make a
   missing joint look like a real joint sitting at the body center. The mask
   array itself is unchanged by this step.

## 4. Mirror TTA (inference-time only)

At inference, after `normalize_body` (§3), run the model **twice** — once on
the clip as-is, once on a mirrored copy — and combine the two predictions.
Mirroring is never applied to training data the browser sees; it's purely an
inference-time technique.

**The mirror transform** (`transforms.mirror_clip`):

1. For each frame, compute a reflection point on the x-axis:
   - If both shoulders are detected in that frame: `(left_shoulder_x + right_shoulder_x) / 2` for that frame.
   - Else: the clip's mean of that same value, over whichever frames *do* have both shoulders detected.
   - Else (no frame in the clip has both shoulders detected at all): `0.5` — reflect about the frame's own horizontal center. (This is in normalized-body coordinate space at this point, since mirroring runs after §3, not in raw `[0,1]` pixel space.)
2. Reflect: `mirrored_x = 2 * reflect_point_x - original_x` (y and z unchanged), per-frame, using that frame's own reflect point from step 1.
3. After reflecting, swap slots (coordinates *and* mask together):
   - The whole left-hand block (0–20) ↔ the whole right-hand block (21–41).
   - Left shoulder (43) ↔ right shoulder (44).
   - Left elbow (45) ↔ right elbow (46).
   - The nose (42) is untouched by swapping (only reflected in x — it's a single joint, not a pair).

Mirroring twice returns the exact original clip — a good self-check when
porting (`test_transforms.py` verifies this in Python across three
shoulder-detection scenarios: all-detected, partially-detected, and
none-detected).

**Combining the two predictions**: take `softmax` of each forward pass's
logits *separately*, then **average the two probability vectors** (not the
raw logits), then `argmax`/top-k on the averaged probabilities. Averaging
raw logits from two different forward passes has no clean interpretation;
averaging softmax probabilities does (it's the two-view mixture prediction).

## 5. Model I/O

- Input: `"input"`, `float32`, shape `(batch, 3, 32, 47)`, channel order `[x, y, z]` (channel-major, not the frame-major `(32, 47, 3)` layout used internally during preprocessing — transpose before calling the model).
- Output: `"logits"`, `float32`, shape `(batch, num_classes)` — **raw logits**, not softmax. Apply softmax yourself (needed anyway for the TTA averaging in §4).
- `num_classes` and the dense-index ↔ original-KSL-class-id mapping come from that specific checkpoint's `label_map.json`, not from this spec.
- Opset 17 (see `export_onnx.py`).

## 6. Verifying your port: golden vectors

`scripts/generate_golden_vectors.py --checkpoint-dir <trained checkpoint>`
produces `golden_vectors.json`: 3 real clips, each with the raw landmarks +
mask (the input to §2/§3), the normalized landmarks (expected output of
§3), and the expected logits for both the original and mirrored input (§4)
plus the final TTA-averaged probabilities. Run your TypeScript
implementation on the same 3 raw-landmark/mask inputs and diff against
these — if your normalized landmarks or logits disagree beyond floating-point
noise, something in the port doesn't match this spec.
