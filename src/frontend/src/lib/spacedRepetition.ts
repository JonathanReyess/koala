/**
 * SM-2-style spaced repetition, per word.
 *
 * - correct: ease factor +0.1 (capped at EASE_FACTOR_MAX, the starting value, so it only ever recovers what
 *   misses took away); interval 0 -> 1 -> 6 -> previous * ease (days).
 * - miss: ease factor -0.2 (floor 1.3), interval reset to 0.
 *
 * (Bug fixed: the old update was `ease + ease + delta`, doubling the ease factor on every correct answer, so
 * intervals exploded after a few successes.)
 */
export interface WordProgress {
  word: string;
  correctCount: number;
  incorrectCount: number;
  lastSeen: number;
  /** Days until the next review (0 = due now / after a miss). */
  interval: number;
  easeFactor: number;
}

export const INITIAL_EASE_FACTOR = 2.5;
export const EASE_FACTOR_MAX = 2.5;
export const MIN_EASE_FACTOR = 1.3;
export const EASE_FACTOR_INCREMENT = 0.1;
export const EASE_FACTOR_DECREMENT = 0.2;
/** Safety cap on intervals (also used to repair values corrupted by the old bug). */
export const MAX_INTERVAL_DAYS = 100;

export const newWordProgress = (word: string): WordProgress => ({
  word,
  correctCount: 0,
  incorrectCount: 0,
  lastSeen: 0,
  interval: 0,
  easeFactor: INITIAL_EASE_FACTOR,
});

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

export function calculateNextReview(current: WordProgress, correct: boolean, now: number = Date.now()): WordProgress {
  const updated = { ...current, lastSeen: now };

  if (correct) {
    updated.correctCount += 1;
    updated.easeFactor = clamp(current.easeFactor + EASE_FACTOR_INCREMENT, MIN_EASE_FACTOR, EASE_FACTOR_MAX);
    if (current.interval === 0) updated.interval = 1;
    else if (current.interval === 1) updated.interval = 6;
    else updated.interval = Math.min(MAX_INTERVAL_DAYS, Math.round(current.interval * updated.easeFactor));
  } else {
    updated.incorrectCount += 1;
    updated.interval = 0;
    updated.easeFactor = Math.max(MIN_EASE_FACTOR, current.easeFactor - EASE_FACTOR_DECREMENT);
  }
  return updated;
}

/**
 * Repairs progress saved by the buggy version (ease factors in the tens/hundreds, absurd intervals) and any
 * malformed values, without touching counts.
 */
export function sanitizeProgress(p: WordProgress): WordProgress {
  const num = (x: unknown, fallback: number) => (typeof x === "number" && Number.isFinite(x) ? x : fallback);
  return {
    ...p,
    correctCount: Math.max(0, num(p.correctCount, 0)),
    incorrectCount: Math.max(0, num(p.incorrectCount, 0)),
    lastSeen: num(p.lastSeen, 0),
    interval: clamp(num(p.interval, 0), 0, MAX_INTERVAL_DAYS),
    easeFactor: clamp(num(p.easeFactor, INITIAL_EASE_FACTOR), MIN_EASE_FACTOR, EASE_FACTOR_MAX),
  };
}
