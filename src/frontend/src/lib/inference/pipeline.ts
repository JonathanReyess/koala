import type { LoadedModels } from "./loader";
import { Clip } from "./preprocess";
import { detectionFractions, DetectionFractions } from "./landmarks";
import { runTta, Prediction } from "./model";
import { Grade, gradePrediction } from "./grading";
import { denseToOriginal } from "./labels";

export interface PipelineResult {
  grade: Grade;
  fractions: DetectionFractions;
  prediction: Prediction | null;
  inferenceMs: number;
}

/** Sampled raw clip -> detection gate -> normalize + TTA inference -> grade. */
export async function gradeClip(models: LoadedModels, clip: Clip, targetClassId: number): Promise<PipelineResult> {
  const fractions = detectionFractions(clip.mask, clip.frames);
  const d2o = denseToOriginal(models.labels);
  const t0 = performance.now();
  // Cheap early-out: no point running the network when the clip fails the detection gate.
  const gated = gradePrediction(new Float64Array(1), d2o, targetClassId, fractions);
  if (gated.status === "not_detected") {
    return { grade: gated, fractions, prediction: null, inferenceMs: 0 };
  }
  const prediction = await runTta(models.ort, models.session, clip);
  const inferenceMs = performance.now() - t0;
  const grade = gradePrediction(prediction.probs, d2o, targetClassId, fractions);
  return { grade, fractions, prediction, inferenceMs };
}
