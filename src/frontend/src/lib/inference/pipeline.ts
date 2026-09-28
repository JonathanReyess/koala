import type { LoadedModels } from "./loader";
import { Clip } from "./preprocess";
import { detectionFractions, DetectionFractions } from "./landmarks";
import { runTta, Prediction } from "./model";
import { Grade, RankedGuess, detectionGate, gradePrediction, topKIndices } from "./grading";
import { denseToOriginal } from "./labels";

export interface PipelineResult {
  grade: Grade;
  fractions: DetectionFractions;
  prediction: Prediction | null;
  /** Top-5 guesses (only when the model ran). */
  top5: RankedGuess[];
  inferenceMs: number;
}

export interface GradeOptions {
  /** Run the model even if the clip fails the detection gate (debug view only; grade is unchanged). */
  alwaysPredict?: boolean;
}

/** Sampled raw clip -> detection gate -> normalize + TTA inference -> grade. */
export async function gradeClip(
  models: LoadedModels,
  clip: Clip,
  targetClassId: number,
  opts: GradeOptions = {},
): Promise<PipelineResult> {
  const fractions = detectionFractions(clip.mask, clip.frames);
  const d2o = denseToOriginal(models.labels);
  const t0 = performance.now();
  // Cheap early-out (unless debugging): no point running the network when the clip fails the detection gate.
  const reason = detectionGate(fractions);
  const gateGrade: Grade | null = reason
    ? { status: "not_detected", reason, countsAsAttempt: false, countsAsMiss: false }
    : null;
  if (gateGrade && !opts.alwaysPredict) {
    return { grade: gateGrade, fractions, prediction: null, top5: [], inferenceMs: 0 };
  }
  const prediction = await runTta(models.ort, models.session, clip);
  const inferenceMs = performance.now() - t0;
  const top5 = topKIndices(prediction.probs, 5).map((d) => ({ classId: d2o(d), prob: prediction.probs[d] }));
  const grade = gateGrade ?? gradePrediction(prediction.probs, d2o, targetClassId, fractions);
  return { grade, fractions, prediction, top5, inferenceMs };
}
