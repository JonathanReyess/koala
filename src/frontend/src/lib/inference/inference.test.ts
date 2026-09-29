import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as ort from "onnxruntime-web";
import {
  Clip,
  sampleFrameIndices,
  normalizeBody,
  mirrorClip,
  writeModelInput,
  buildBatchInput,
  J,
} from "./preprocess";
import { runTta, softmax } from "./model";
import { gradePrediction } from "./grading";
import { checkFraming, detectionFractions, resultTo47 } from "./landmarks";
import { feedbackMessage } from "./messages";
import { attemptCount, attemptsToJson, buildAttemptRecord, clearAttempts, logAttempt } from "./attemptLog";
import { INFERENCE_CONFIG } from "./config";

const root = resolve(__dirname, "../../..");
const golden = JSON.parse(readFileSync(resolve(root, "src/test/fixtures/golden_vectors.json"), "utf8"));
const labels = JSON.parse(readFileSync(resolve(root, "public/models/ksl_f/label_map.json"), "utf8"));

const flat = (a: number[][][] | number[][]) => Float32Array.from((a as number[][]).flat(2));
const toClip = (v: { raw_landmarks: number[][][]; raw_mask: number[][] }): Clip => ({
  coords: flat(v.raw_landmarks),
  mask: Uint8Array.from(v.raw_mask.flat()),
  frames: 32,
});
const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

describe("frame sampling (np.linspace dtype=int truncation)", () => {
  it("matches numpy's truncation, not rounding", () => {
    expect(sampleFrameIndices(3, 5)).toEqual([0, 0, 1, 1, 2]); // np.linspace(0,2,5,dtype=int)
    expect(sampleFrameIndices(0)).toEqual([]);
    expect(sampleFrameIndices(1)).toEqual(Array(32).fill(0));
  });
  it("matches numpy reference values for typical clip lengths", () => {
    // Values produced with np.linspace(0, n-1, 32, dtype=int).tolist()
    expect(sampleFrameIndices(100)).toEqual([
      0, 3, 6, 9, 12, 15, 19, 22, 25, 28, 31, 35, 38, 41, 44, 47, 51, 54, 57, 60, 63, 67, 70, 73, 76, 79, 83, 86, 89,
      92, 95, 99,
    ]);
  });
  it("equals np.linspace(0, n-1, 32, dtype=int) for every n in 1..400", () => {
    const table = JSON.parse(readFileSync(resolve(root, "src/test/fixtures/linspace_indices.json"), "utf8"));
    for (const [n, expected] of Object.entries(table)) {
      expect(sampleFrameIndices(Number(n))).toEqual(expected);
    }
  });
  it("always includes first and last frame", () => {
    for (const n of [2, 31, 32, 33, 97, 250]) {
      const idx = sampleFrameIndices(n);
      expect(idx[0]).toBe(0);
      expect(idx[31]).toBe(n - 1);
    }
  });
});

describe("golden vectors: preprocessing", () => {
  golden.vectors.forEach((v: never, i: number) => {
    const g = v as {
      raw_landmarks: number[][][];
      raw_mask: number[][];
      normalized_landmarks: number[][][];
    };
    it(`vector ${i}: normalize_body within 1e-5`, () => {
      const norm = normalizeBody(toClip(g));
      expect(maxAbsDiff(norm.coords, flat(g.normalized_landmarks))).toBeLessThan(1e-5);
    });
    it(`vector ${i}: mirror is idempotent`, () => {
      const norm = normalizeBody(toClip(g));
      const twice = mirrorClip(mirrorClip(norm));
      expect(maxAbsDiff(twice.coords, norm.coords)).toBeLessThan(1e-5);
      expect(Array.from(twice.mask)).toEqual(Array.from(norm.mask));
    });
  });
  it("normalize_body is a no-op with no shoulders, and zeroes missing joints", () => {
    const clip: Clip = { coords: new Float32Array(32 * J * 3).fill(0.5), mask: new Uint8Array(32 * J), frames: 32 };
    expect(Array.from(normalizeBody(clip).coords)).toEqual(Array.from(clip.coords));
    // shoulders in frame 0 only; joint 0 missing => stays exactly 0 elsewhere
    clip.mask[43] = 1;
    clip.mask[44] = 1;
    clip.coords[44 * 3] = 0.7;
    const out = normalizeBody(clip);
    expect(out.coords[0]).toBe(0);
    expect(out.coords[43 * 3]).not.toBe(0);
  });
});

describe("golden vectors: onnxruntime-web logits", () => {
  let session: ort.InferenceSession;
  beforeAll(async () => {
    ort.env.wasm.wasmPaths = resolve(root, "node_modules/onnxruntime-web/dist") + "/";
    ort.env.wasm.numThreads = 1;
    const bytes = readFileSync(resolve(root, "public/models/ksl_f/model.onnx"));
    session = await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ["wasm"] });
  });

  it("model I/O names match the spec", () => {
    expect(session.inputNames).toEqual(["input"]);
    expect(session.outputNames).toEqual(["logits"]);
  });

  golden.vectors.forEach((v: never, i: number) => {
    const g = v as {
      raw_landmarks: number[][][];
      raw_mask: number[][];
      logits_original: number[];
      logits_mirrored: number[];
      probs_tta_averaged: number[];
      predicted_class_original_id: number;
    };
    it(`vector ${i}: original + mirrored logits within 1e-3, TTA probs match`, async () => {
      const pred = await runTta(ort, session, toClip(g));
      expect(maxAbsDiff(pred.logitsOriginal, g.logits_original)).toBeLessThan(1e-3);
      expect(maxAbsDiff(pred.logitsMirrored, g.logits_mirrored)).toBeLessThan(1e-3);
      expect(maxAbsDiff(pred.probs, g.probs_tta_averaged)).toBeLessThan(1e-3);
      const top = pred.probs.indexOf(Math.max(...pred.probs));
      expect(labels.reverse_label_map[String(top)]).toBe(g.predicted_class_original_id);
    });
    it(`vector ${i}: batch layout is channel-major (3,32,47)`, () => {
      const { data, normalized } = buildBatchInput(toClip(g));
      const single = new Float32Array(3 * 32 * J);
      writeModelInput(normalized, single);
      expect(Array.from(data.slice(0, single.length))).toEqual(Array.from(single));
      // channel 1 (y), frame 2, joint 5
      expect(single[(1 * 32 + 2) * J + 5]).toBe(normalized.coords[(2 * J + 5) * 3 + 1]);
    });
  });
});

describe("landmark layout + grading", () => {
  it("resultTo47 places hands and the 5 pose joints in the training layout", () => {
    const pt = (v: number) => ({ x: v, y: v + 0.1, z: v + 0.2 });
    const pose = Array.from({ length: 33 }, (_, i) => pt(i / 100));
    const hand = (o: number) => Array.from({ length: 21 }, (_, i) => pt(o + i / 1000));
    const f = resultTo47({ poseLandmarks: [pose], leftHandLandmarks: [hand(1)], rightHandLandmarks: [] });
    expect(Array.from(f.mask.slice(0, 21)).every((m) => m === 1)).toBe(true);
    expect(Array.from(f.mask.slice(21, 42)).every((m) => m === 0)).toBe(true);
    expect(Array.from(f.mask.slice(42))).toEqual([1, 1, 1, 1, 1]);
    expect(f.coords[43 * 3]).toBeCloseTo(0.11); // left shoulder = BlazePose 11
    expect(f.coords[46 * 3]).toBeCloseTo(0.14); // right elbow = BlazePose 14
    expect(f.coords[21 * 3]).toBe(0); // missing right hand stays exactly 0
  });

  const d2o = (d: number) => d + 100;
  /** 10-class prob vector with the given (dense idx -> prob) entries; the rest share what's left. */
  const P = (entries: Record<number, number>) => {
    const p = new Float64Array(10);
    const used = Object.values(entries).reduce((a, b) => a + b, 0);
    const rest = 10 - Object.keys(entries).length;
    p.fill((1 - used) / rest);
    for (const [d, v] of Object.entries(entries)) p[Number(d)] = v;
    return p;
  };
  const ok = { pose: 1, anyHand: 1, leftHand: 1, rightHand: 1 };
  const G = INFERENCE_CONFIG.grading;
  /** Fixed thresholds so the boundary cases below don't move when the calibrated config changes. */
  const TEST_CFG = { ...G, CORRECT_MIN: 0.4, CLOSE_MIN: 0.15, CONFUSION_MIN: 0.6 };

  it("thresholds live in config with the calibrated values", () => {
    expect([G.CORRECT_MIN, G.CLOSE_MIN, G.CONFUSION_MIN]).toEqual([0.2, 0.1, 0.75]);
    expect(INFERENCE_CONFIG.landmarkerDelegate).toBe("CPU");
  });

  it("calibrated defaults: boundaries at 0.20 / 0.10 / 0.75", () => {
    const g = (e: Record<number, number>) => gradePrediction(P(e), d2o, 101, ok).status;
    expect(g({ 1: 0.2 })).toBe("correct");
    expect(g({ 1: 0.19 })).toBe("close"); // top-1 but under CORRECT_MIN, still >= CLOSE_MIN
    expect(g({ 1: 0.347 })).toBe("correct"); // learner-testing regressions: "when" 34.7%, "study" 41%
    expect(g({ 1: 0.41 })).toBe("correct");
    expect(g({ 2: 0.6, 3: 0.2, 1: 0.1 })).toBe("close");
    expect(g({ 2: 0.6, 3: 0.2, 4: 0.09, 1: 0.05 })).toBe("incorrect");
    expect(g({ 2: 0.75, 1: 0.02 })).toBe("confused");
  });

  it("not_detected when pose <60% or any-hand <30% (with the reason); not an attempt", () => {
    const pose = gradePrediction(P({ 1: 0.9 }), d2o, 101, { ...ok, pose: 0.59 }, TEST_CFG);
    expect(pose).toMatchObject({ status: "not_detected", reason: "pose", countsAsAttempt: false, countsAsMiss: false });
    const hands = gradePrediction(P({ 1: 0.9 }), d2o, 101, { ...ok, anyHand: 0.29 }, TEST_CFG);
    expect(hands).toMatchObject({ status: "not_detected", reason: "hands", countsAsAttempt: false });
    expect(gradePrediction(P({ 1: 0.9 }), d2o, 101, { ...ok, pose: 0.6, anyHand: 0.3 }, TEST_CFG).status).toBe("correct");
  });

  it("correct needs top-1 AND probability >= CORRECT_MIN", () => {
    expect(gradePrediction(P({ 1: 0.4 }), d2o, 101, ok, TEST_CFG).status).toBe("correct");
    // top-1 but under CORRECT_MIN (0.39): falls to close (>= CLOSE_MIN)
    expect(gradePrediction(P({ 1: 0.39 }), d2o, 101, ok, TEST_CFG).status).toBe("close");
  });

  it("close: target in top-3 and p >= CLOSE_MIN, not a miss", () => {
    const g = gradePrediction(P({ 2: 0.5, 3: 0.2, 1: 0.15 }), d2o, 101, ok, TEST_CFG);
    expect(g).toMatchObject({ status: "close", top1: 102, countsAsMiss: false, countsAsAttempt: true });
    // in top-3 but too improbable -> not close
    expect(gradePrediction(P({ 2: 0.5, 3: 0.3, 1: 0.14 }), d2o, 101, ok, TEST_CFG).status).not.toBe("close");
    // p >= CLOSE_MIN but outside the top-3 -> not close
    expect(gradePrediction(P({ 2: 0.3, 3: 0.25, 4: 0.2, 1: 0.16 }), d2o, 101, ok, TEST_CFG).status).not.toBe("close");
  });

  it("confused: wrong top-1 with p >= CONFUSION_MIN names the word and is a miss", () => {
    const g = gradePrediction(P({ 2: 0.6, 1: 0.05 }), d2o, 101, ok, TEST_CFG);
    expect(g).toMatchObject({ status: "confused", top1: 102, countsAsMiss: true });
    expect(gradePrediction(P({ 2: 0.59, 1: 0.05 }), d2o, 101, ok, TEST_CFG).status).toBe("incorrect");
  });

  it("incorrect otherwise: counts as a miss, no word named", () => {
    const g = gradePrediction(P({ 2: 0.3, 3: 0.2, 4: 0.15, 1: 0.01 }), d2o, 101, ok, TEST_CFG);
    expect(g).toMatchObject({ status: "incorrect", countsAsMiss: true, countsAsAttempt: true });
    const word = (id: number) => `word${id}`;
    expect(feedbackMessage(g, word)).toBe("Not quite. Watch the example and try again.");
    expect(feedbackMessage(g, word)).not.toMatch(/word/);
  });

  it("order of checks follows the spec: close beats confused when the target is in the top-3", () => {
    expect(gradePrediction(P({ 2: 0.7, 1: 0.2 }), d2o, 101, ok, TEST_CFG).status).toBe("close");
  });

  it("messages", () => {
    const word = (id: number) => `word${id}`;
    const msg = (e: Record<number, number>) => feedbackMessage(gradePrediction(P(e), d2o, 101, ok), word);
    expect(msg({ 2: 0.8, 1: 0.05 })).toBe("That looked like “word102”.");
    expect(msg({ 1: 0.9 })).toBe("Perfect!");
    expect(msg({ 2: 0.8, 1: 0.1 })).toBe("Almost! It looked a bit like “word102”."); // close AND confident top-1
    const g = (over: object) => ({ countsAsAttempt: true, countsAsMiss: false, ...over }) as never;
    expect(feedbackMessage(g({ status: "not_detected", reason: "hands" }), word)).toBe(
      "I couldn't see your hands much. Keep them in view while signing.",
    );
  });

  it("close never names a low-confidence top-1 (regression: 'looked a bit like subway' at 44.6%)", () => {
    const word = (id: number) => `word${id}`;
    const msg = (e: Record<number, number>) => {
      const grade = gradePrediction(P(e), d2o, 101, ok);
      return { grade, text: feedbackMessage(grade, word) };
    };
    // wrong top-1 at 44.6% (< CONFUSION_MIN 0.75); target 2nd at 30% -> close, generic message
    const low = msg({ 2: 0.446, 1: 0.3 });
    expect(low.grade.status).toBe("close");
    expect(low.grade.namesTop1).toBe(false);
    expect(low.text).toBe("Almost! That was close. Try once more.");
    expect(low.text).not.toMatch(/word/);
    // just under the confusion bar: still generic; at the bar: names the word
    expect(msg({ 2: 0.74, 1: 0.15 }).text).toBe("Almost! That was close. Try once more.");
    expect(msg({ 2: 0.75, 1: 0.15 }).text).toBe("Almost! It looked a bit like “word102”.");
    // target itself is top-1 but under CORRECT_MIN: generic, never names the target
    const self = msg({ 1: 0.15 });
    expect(self.grade.status).toBe("close");
    expect(self.text).toBe("Almost! That was close. Try once more.");
  });

  it("live framing never warns about hands", () => {
    const frame = (mask: number[], coords: Record<number, [number, number]>) => {
      const m = new Uint8Array(J);
      const c = new Float32Array(J * 3);
      mask.forEach((j) => (m[j] = 1));
      for (const [j, [x, y]] of Object.entries(coords)) {
        c[Number(j) * 3] = x;
        c[Number(j) * 3 + 1] = y;
      }
      return { coords: c, mask: m };
    };
    const body = { 42: [0.5, 0.3], 43: [0.6, 0.6], 44: [0.4, 0.6] } as Record<number, [number, number]>;
    // no hands at all -> fine
    expect(checkFraming(frame([42, 43, 44], body))).toEqual({ ok: true, issues: [] });
    // too far away (shoulders 0.05 apart)
    expect(checkFraming(frame([42, 43, 44], { ...body, 43: [0.525, 0.6], 44: [0.475, 0.6] })).issues).toEqual(["too_small"]);
    // face missing
    expect(checkFraming(frame([43, 44], body)).issues).toEqual(["face"]);
    // shoulders missing
    expect(checkFraming(frame([42], body)).issues).toContain("shoulders");
    // nothing
    expect(checkFraming(frame([], {})).issues).toEqual(["no_body"]);
    for (const f of [frame([42, 43, 44], body), frame([], {}), frame([43, 44], body)]) {
      expect(checkFraming(f).issues.join()).not.toMatch(/hand/);
    }
  });

  it("detectionFractions counts frames with pose / any hand / each hand", () => {
    const mask = new Uint8Array(32 * J);
    for (let t = 0; t < 16; t++) mask[t * J + 43] = 1;
    for (let t = 0; t < 8; t++) mask[t * J + 30] = 1; // right hand
    for (let t = 4; t < 12; t++) mask[t * J + 3] = 1; // left hand
    expect(detectionFractions(mask, 32)).toEqual({ pose: 0.5, anyHand: 12 / 32, leftHand: 0.25, rightHand: 0.25 });
    expect(softmax([0, 0]).reduce((a, b) => a + b)).toBeCloseTo(1);
  });
});

describe("debug attempt log", () => {
  it("records target, top-5, grade, fractions, delegate, fps and the raw 32x47x3 clip + mask", () => {
    const g = golden.vectors[0];
    const clip = toClip(g);
    const d2o = (d: number) => d + 100;
    const probs = new Float64Array(10).fill(0.02);
    probs[3] = 0.6;
    probs[1] = 0.12;
    const fractions = detectionFractions(clip.mask, clip.frames);
    const grade = gradePrediction(probs, d2o, 101, fractions);
    const top5 = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 5).map((d) => ({ classId: d2o(d), prob: probs[d] }));
    clearAttempts();
    logAttempt(
      buildAttemptRecord({
        source: "recorded", targetWord: "eat", targetClassId: 101, top5, grade, fractions,
        wordFor: (id) => `w${id}`, delegate: "CPU", backend: "wasm", liveFps: 11.5, clip,
        now: new Date("2026-01-02T03:04:05Z"),
      }),
    );
    expect(attemptCount()).toBe(1);
    const out = JSON.parse(attemptsToJson());
    const r = out.attempts[0];
    expect(r).toMatchObject({
      timestamp: "2026-01-02T03:04:05.000Z", source: "recorded", target_word: "eat", target_class_id: 101,
      delegate: "CPU", ort_backend: "wasm", live_fps: 11.5,
    });
    expect(r.top5).toHaveLength(5);
    expect(r.top5[0]).toMatchObject({ class_id: 103, word: "w103" });
    expect(r.grade).toMatchObject({ status: "close", top1_class_id: 103, names_top1: false });
    expect(r.thresholds).toMatchObject({ CORRECT_MIN: 0.2, CLOSE_MIN: 0.1, CONFUSION_MIN: 0.75 });
    expect(r.detection_fractions).toEqual(fractions);
    expect(r.raw_landmarks).toHaveLength(32);
    expect(r.raw_landmarks[0]).toHaveLength(47);
    expect(r.raw_landmarks[0][0]).toHaveLength(3);
    expect(r.raw_mask).toHaveLength(32);
    // Same nested layout/values as the golden fixture's raw input (float32-exact).
    expect(r.raw_landmarks[5][43]).toEqual(g.raw_landmarks[5][43].map(Math.fround));
    expect(r.raw_mask).toEqual(g.raw_mask);
    clearAttempts();
    expect(attemptCount()).toBe(0);
  });
});
