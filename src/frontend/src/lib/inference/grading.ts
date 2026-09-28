import { INFERENCE_CONFIG, GradingConfig } from "./config";
import type { DetectionFractions } from "./landmarks";

export type GradeStatus = "correct" | "close" | "incorrect" | "not_detected";

export interface Grade {
  status: GradeStatus;
  /** Original class id (folder id) of the top-1 guess, when a prediction exists. */
  top1?: number;
  /** Top-K original class ids, best first. */
  topK?: number[];
  /** Counts as an attempt in spaced repetition? not_detected: no; close: yes but not as a miss. */
  countsAsAttempt: boolean;
  /** Whether to record a *miss* for spaced repetition. */
  countsAsMiss: boolean;
}

export function topKIndices(probs: ArrayLike<number>, k: number): number[] {
  return Array.from({ length: probs.length }, (_, i) => i)
    .sort((a, b) => probs[b] - probs[a] || a - b)
    .slice(0, k);
}

/**
 * @param probs averaged softmax probabilities, dense-indexed
 * @param denseToOriginal dense index -> original class id
 * @param targetOriginalId original class id of the word being practised
 */
export function gradePrediction(
  probs: ArrayLike<number>,
  denseToOriginal: (dense: number) => number,
  targetOriginalId: number,
  fractions: DetectionFractions,
  cfg: GradingConfig = INFERENCE_CONFIG.grading,
): Grade {
  if (fractions.pose < cfg.minPoseFrac || fractions.anyHand < cfg.minAnyHandFrac) {
    return { status: "not_detected", countsAsAttempt: false, countsAsMiss: false };
  }
  const topK = topKIndices(probs, cfg.closeTopK).map(denseToOriginal);
  const top1 = topK[0];
  if (top1 === targetOriginalId) {
    return { status: "correct", top1, topK, countsAsAttempt: true, countsAsMiss: false };
  }
  if (topK.includes(targetOriginalId)) {
    return { status: "close", top1, topK, countsAsAttempt: true, countsAsMiss: false };
  }
  return { status: "incorrect", top1, topK, countsAsAttempt: true, countsAsMiss: true };
}
