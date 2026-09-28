/**
 * TypeScript port of the preprocessing contract in preprocessing_spec.json /
 * docs/PREPROCESSING.md. Python references: extract_landmarks.sample_frame_indices,
 * transforms.normalize_body, transforms.mirror_clip.
 *
 * Internal layout is frame-major: coords Float32Array(T*J*3) indexed
 * ((t*J)+j)*3+c, mask Uint8Array(T*J). Model layout is channel-major (3,T,J).
 */
import { INFERENCE_CONFIG } from "./config";

export const J = INFERENCE_CONFIG.numJoints;
export const LEFT_HAND_START = 0;
export const RIGHT_HAND_START = 21;
export const HAND_LEN = 21;
export const NOSE = 42;
export const LEFT_SHOULDER = 43;
export const RIGHT_SHOULDER = 44;
export const LEFT_ELBOW = 45;
export const RIGHT_ELBOW = 46;

export interface Clip {
  /** (T, J, 3) frame-major, float32. */
  coords: Float32Array;
  /** (T, J), 0/1. */
  mask: Uint8Array;
  frames: number;
}

/** One recorded frame: (J*3) coords + (J) mask. */
export interface FrameLandmarks {
  coords: Float32Array;
  mask: Uint8Array;
}

/**
 * np.linspace(0, total-1, seqLen, dtype=int): float linspace, then TRUNCATE.
 * numpy computes start + i*step with step = (stop-start)/(num-1), and pins the
 * last element to `stop` exactly.
 */
export function sampleFrameIndices(totalFrames: number, seqLen: number = INFERENCE_CONFIG.sequenceLength): number[] {
  if (totalFrames <= 0) return [];
  const out: number[] = [];
  const step = (totalFrames - 1) / (seqLen - 1);
  for (let i = 0; i < seqLen; i++) {
    const v = i === seqLen - 1 ? totalFrames - 1 : i * step;
    out.push(Math.trunc(v));
  }
  return out;
}

/** Picks the 32 spec-sampled frames out of everything recorded. All-zero clip if nothing recorded. */
export function buildClip(frames: FrameLandmarks[], seqLen: number = INFERENCE_CONFIG.sequenceLength): Clip {
  const coords = new Float32Array(seqLen * J * 3);
  const mask = new Uint8Array(seqLen * J);
  const idx = sampleFrameIndices(frames.length, seqLen);
  idx.forEach((frameIdx, slot) => {
    coords.set(frames[frameIdx].coords, slot * J * 3);
    mask.set(frames[frameIdx].mask, slot * J);
  });
  return { coords, mask, frames: seqLen };
}

const c = (t: number, j: number, ax: number) => (t * J + j) * 3 + ax;

/** transforms.normalize_body. */
export function normalizeBody(clip: Clip, eps = 1e-6): Clip {
  const { coords, mask, frames: T } = clip;
  const refs: number[] = [];
  for (let t = 0; t < T; t++) {
    if (mask[t * J + LEFT_SHOULDER] && mask[t * J + RIGHT_SHOULDER]) refs.push(t);
  }
  if (refs.length === 0) return { coords: coords.slice(), mask: mask.slice(), frames: T };

  const mid = [0, 0, 0];
  let width = 0;
  for (const t of refs) {
    let d2 = 0;
    for (let ax = 0; ax < 3; ax++) {
      const l = coords[c(t, LEFT_SHOULDER, ax)];
      const r = coords[c(t, RIGHT_SHOULDER, ax)];
      mid[ax] += (l + r) / 2;
      d2 += (l - r) * (l - r);
    }
    width += Math.sqrt(d2);
  }
  for (let ax = 0; ax < 3; ax++) mid[ax] /= refs.length;
  width = Math.max(width / refs.length, eps);

  const out = new Float32Array(coords.length);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < J; j++) {
      if (!mask[t * J + j]) continue; // missing joints stay exactly 0
      for (let ax = 0; ax < 3; ax++) {
        out[c(t, j, ax)] = (coords[c(t, j, ax)] - mid[ax]) / width;
      }
    }
  }
  return { coords: out, mask: mask.slice(), frames: T };
}

/** transforms.mirror_clip (expects an already-normalized clip). */
export function mirrorClip(clip: Clip): Clip {
  const { coords, mask, frames: T } = clip;
  const midX = new Float64Array(T);
  const valid = new Uint8Array(T);
  let sum = 0;
  let n = 0;
  for (let t = 0; t < T; t++) {
    if (mask[t * J + LEFT_SHOULDER] && mask[t * J + RIGHT_SHOULDER]) {
      midX[t] = (coords[c(t, LEFT_SHOULDER, 0)] + coords[c(t, RIGHT_SHOULDER, 0)]) / 2;
      valid[t] = 1;
      sum += midX[t];
      n++;
    }
  }
  const fallback = n > 0 ? sum / n : 0.5;

  const oc = new Float32Array(coords.length);
  const om = mask.slice();
  for (let t = 0; t < T; t++) {
    const m = valid[t] ? midX[t] : fallback;
    for (let j = 0; j < J; j++) {
      oc[c(t, j, 0)] = 2 * m - coords[c(t, j, 0)];
      oc[c(t, j, 1)] = coords[c(t, j, 1)];
      oc[c(t, j, 2)] = coords[c(t, j, 2)];
    }
  }

  const swapJoint = (t: number, a: number, b: number) => {
    for (let ax = 0; ax < 3; ax++) {
      const tmp = oc[c(t, a, ax)];
      oc[c(t, a, ax)] = oc[c(t, b, ax)];
      oc[c(t, b, ax)] = tmp;
    }
    const tm = om[t * J + a];
    om[t * J + a] = om[t * J + b];
    om[t * J + b] = tm;
  };
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < HAND_LEN; k++) swapJoint(t, LEFT_HAND_START + k, RIGHT_HAND_START + k);
    swapJoint(t, LEFT_SHOULDER, RIGHT_SHOULDER);
    swapJoint(t, LEFT_ELBOW, RIGHT_ELBOW);
  }
  return { coords: oc, mask: om, frames: T };
}

/** (T,J,3) frame-major -> (3,T,J) channel-major, appended into `dst` at `offset`. */
export function writeModelInput(clip: Clip, dst: Float32Array, offset = 0): void {
  const T = clip.frames;
  for (let ax = 0; ax < 3; ax++) {
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < J; j++) {
        dst[offset + (ax * T + t) * J + j] = clip.coords[c(t, j, ax)];
      }
    }
  }
}

/** Convenience: raw sampled clip -> batch-2 model input [normalized, mirror(normalized)]. */
export function buildBatchInput(rawClip: Clip): { data: Float32Array; normalized: Clip; mirrored: Clip } {
  const normalized = normalizeBody(rawClip);
  const mirrored = mirrorClip(normalized);
  const per = 3 * rawClip.frames * J;
  const data = new Float32Array(2 * per);
  writeModelInput(normalized, data, 0);
  writeModelInput(mirrored, data, per);
  return { data, normalized, mirrored };
}
