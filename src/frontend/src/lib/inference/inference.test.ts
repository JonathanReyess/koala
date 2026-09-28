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
import { detectionFractions, resultTo47 } from "./landmarks";

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
  const probs = (order: number[]) => {
    const p = new Float64Array(10).fill(0.01);
    order.forEach((d, r) => (p[d] = 0.5 - r * 0.1));
    return p;
  };
  const ok = { pose: 1, anyHand: 1 };
  it("not_detected when pose <60% or any-hand <30%; not an attempt", () => {
    for (const fr of [{ pose: 0.59, anyHand: 1 }, { pose: 1, anyHand: 0.29 }]) {
      const g = gradePrediction(probs([1, 2, 3]), d2o, 101, fr);
      expect(g.status).toBe("not_detected");
      expect(g.countsAsAttempt).toBe(false);
    }
    expect(gradePrediction(probs([1]), d2o, 101, { pose: 0.6, anyHand: 0.3 }).status).toBe("correct");
  });
  it("correct / close / incorrect, with no class masking", () => {
    expect(gradePrediction(probs([1, 2, 3]), d2o, 101, ok).status).toBe("correct");
    const close = gradePrediction(probs([2, 3, 1]), d2o, 101, ok);
    expect(close).toMatchObject({ status: "close", top1: 102, countsAsMiss: false });
    const bad = gradePrediction(probs([2, 3, 4, 1]), d2o, 101, ok);
    expect(bad).toMatchObject({ status: "incorrect", top1: 102, countsAsMiss: true });
  });
  it("detectionFractions counts frames with pose / any hand", () => {
    const mask = new Uint8Array(32 * J);
    for (let t = 0; t < 16; t++) mask[t * J + 43] = 1;
    for (let t = 0; t < 8; t++) mask[t * J + 30] = 1;
    expect(detectionFractions(mask, 32)).toEqual({ pose: 0.5, anyHand: 0.25 });
    expect(softmax([0, 0]).reduce((a, b) => a + b)).toBeCloseTo(1);
  });
});
