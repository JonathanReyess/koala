import { INFERENCE_CONFIG, GradingConfig } from "./config";
import type { DetectionFractions } from "./landmarks";

/**
 * correct       target is top-1 and confident enough
 * close         target is in the top-K with at least CLOSE_MIN probability (not accepted as correct)
 * confused      wrong, but the top-1 guess is confident (>= CONFUSION_MIN): we may name that word
 * incorrect     wrong/unsure: never names a word
 * not_detected  couldn't see the signer; not an attempt
 */
export type GradeStatus = "correct" | "close" | "confused" | "incorrect" | "not_detected";

/** Why a clip was not_detected (drives the guidance message). */
export type NotDetectedReason = "pose" | "hands";

export interface RankedGuess {
  /** Original class id (folder id). */
  classId: number;
  prob: number;
}

export interface Grade {
  status: GradeStatus;
  reason?: NotDetectedReason;
  /** Original class id of the top-1 guess, when a prediction exists. */
  top1?: number;
  top1Prob?: number;
  targetProb?: number;
  /**
   * True when a message may name the top-1 word: top-1 is not the target AND its probability >= CONFUSION_MIN.
   * The one place that decides this, so no message can name a low-confidence guess.
   */
  namesTop1?: boolean;
  /** Top-5 (best first) for debugging. */
  ranked?: RankedGuess[];
  /** Counts as an attempt in spaced repetition? Only not_detected does not. */
  countsAsAttempt: boolean;
  /** Record a *miss* for spaced repetition (confused + incorrect). close is neutral. */
  countsAsMiss: boolean;
}

export function topKIndices(probs: ArrayLike<number>, k: number): number[] {
  return Array.from({ length: probs.length }, (_, i) => i)
    .sort((a, b) => probs[b] - probs[a] || a - b)
    .slice(0, k);
}

/** Detection gate: null when the clip is usable, else why it isn't. */
export function detectionGate(
  fractions: DetectionFractions,
  cfg: GradingConfig = INFERENCE_CONFIG.grading,
): NotDetectedReason | null {
  if (fractions.pose < cfg.minPoseFrac) return "pose";
  if (fractions.anyHand < cfg.minAnyHandFrac) return "hands";
  return null;
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
  const reason = detectionGate(fractions, cfg);
  if (reason) return { status: "not_detected", reason, countsAsAttempt: false, countsAsMiss: false };

  const order = topKIndices(probs, Math.max(5, cfg.closeTopK));
  const ranked = order.map((d) => ({ classId: denseToOriginal(d), prob: probs[d] }));
  const top1 = ranked[0];
  const topKIds = ranked.slice(0, cfg.closeTopK).map((r) => r.classId);
  const targetRank = ranked.findIndex((r) => r.classId === targetOriginalId);
  // Target may be outside the ranked window; look its probability up directly.
  let targetProb = targetRank >= 0 ? ranked[targetRank].prob : 0;
  if (targetRank < 0) {
    for (let d = 0; d < probs.length; d++) if (denseToOriginal(d) === targetOriginalId) targetProb = probs[d];
  }
  const namesTop1 = top1.classId !== targetOriginalId && top1.prob >= cfg.CONFUSION_MIN;
  const base = { top1: top1.classId, top1Prob: top1.prob, targetProb, ranked, namesTop1 };

  if (top1.classId === targetOriginalId && targetProb >= cfg.CORRECT_MIN) {
    return { status: "correct", ...base, countsAsAttempt: true, countsAsMiss: false };
  }
  if (topKIds.includes(targetOriginalId) && targetProb >= cfg.CLOSE_MIN) {
    return { status: "close", ...base, countsAsAttempt: true, countsAsMiss: false };
  }
  if (namesTop1) {
    return { status: "confused", ...base, countsAsAttempt: true, countsAsMiss: true };
  }
  return { status: "incorrect", ...base, countsAsAttempt: true, countsAsMiss: true };
}
