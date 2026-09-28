/**
 * onnxruntime-web session + TTA inference.
 * Model I/O (verified with onnxruntime): input "input" float32 (N,3,32,47),
 * output "logits" float32 (N,67).
 */
import type { InferenceSession, Tensor as OrtTensor } from "onnxruntime-web";
import { INFERENCE_CONFIG } from "./config";
import { Clip, buildBatchInput, J } from "./preprocess";

export const MODEL_INPUT_NAME = "input";
export const MODEL_OUTPUT_NAME = "logits";

export interface LabelMap {
  label_map: Record<string, number>; // original id (string) -> dense
  reverse_label_map: Record<string, number>; // dense (string) -> original id
}

export function softmax(logits: ArrayLike<number>): Float64Array {
  const out = new Float64Array(logits.length);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/** Average of the two softmax vectors (NOT of the logits), per the TTA rule in the spec. */
export function averageProbs(logitsA: ArrayLike<number>, logitsB: ArrayLike<number>): Float64Array {
  const a = softmax(logitsA);
  const b = softmax(logitsB);
  return a.map((v, i) => (v + b[i]) / 2);
}

export interface Prediction {
  logitsOriginal: Float32Array;
  logitsMirrored: Float32Array;
  probs: Float64Array;
  normalized: Clip;
  mirrored: Clip;
}

/** ort namespace is injected so tests (node) and the app (browser) can configure it differently. */
export type OrtModule = typeof import("onnxruntime-web");

export async function runTta(
  ort: OrtModule,
  session: InferenceSession,
  rawClip: Clip,
): Promise<Prediction> {
  const { data, normalized, mirrored } = buildBatchInput(rawClip);
  const T = INFERENCE_CONFIG.sequenceLength;
  const input: OrtTensor = new ort.Tensor("float32", data, [2, 3, T, J]);
  const out = await session.run({ [MODEL_INPUT_NAME]: input });
  const logits = out[MODEL_OUTPUT_NAME].data as Float32Array;
  const n = logits.length / 2;
  const logitsOriginal = logits.slice(0, n);
  const logitsMirrored = logits.slice(n);
  return {
    logitsOriginal,
    logitsMirrored,
    probs: averageProbs(logitsOriginal, logitsMirrored),
    normalized,
    mirrored,
  };
}
