import { describe, it, expect } from "vitest";
import {
  EASE_FACTOR_MAX,
  INITIAL_EASE_FACTOR,
  MAX_INTERVAL_DAYS,
  MIN_EASE_FACTOR,
  calculateNextReview,
  newWordProgress,
  sanitizeProgress,
} from "./spacedRepetition";

const answer = (results: boolean[]) => {
  let p = newWordProgress("word");
  const history = [p];
  results.forEach((ok, i) => {
    p = calculateNextReview(p, ok, 1000 + i);
    history.push(p);
  });
  return history;
};

describe("spaced repetition (SM-2 style)", () => {
  it("5 correct answers in a row give sensible, growing (not exploding) intervals", () => {
    const h = answer([true, true, true, true, true]);
    const intervals = h.slice(1).map((p) => p.interval);
    expect(intervals.slice(0, 3)).toEqual([1, 6, 15]); // 1 day, 6 days, then x ease (2.5)
    for (let i = 1; i < intervals.length; i++) expect(intervals[i]).toBeGreaterThan(intervals[i - 1]);
    expect(intervals[4]).toBeLessThanOrEqual(MAX_INTERVAL_DAYS); // ~3 months, not thousands of days
    for (const p of h) {
      expect(p.easeFactor).toBeLessThanOrEqual(EASE_FACTOR_MAX);
      expect(p.easeFactor).toBeGreaterThanOrEqual(MIN_EASE_FACTOR);
    }
    expect(h[5].correctCount).toBe(5);
  });

  it("the old bug (ease doubling every correct answer) is gone: ease never exceeds its cap", () => {
    const h = answer(Array(20).fill(true));
    expect(Math.max(...h.map((p) => p.easeFactor))).toBe(INITIAL_EASE_FACTOR);
    expect(Math.max(...h.map((p) => p.interval))).toBeLessThanOrEqual(MAX_INTERVAL_DAYS);
  });

  it("a miss decreases ease by a small step, resets the interval, and is floored at 1.3", () => {
    const [, one] = answer([false]);
    expect(one.easeFactor).toBeCloseTo(INITIAL_EASE_FACTOR - 0.2, 10);
    expect(one.interval).toBe(0);
    expect(one.incorrectCount).toBe(1);
    const many = answer(Array(20).fill(false));
    expect(many[20].easeFactor).toBe(MIN_EASE_FACTOR);
  });

  it("correct answers slowly recover ease lost to misses", () => {
    const h = answer([false, false, true, true]);
    expect(h[2].easeFactor).toBeCloseTo(2.1, 10);
    expect(h[3].easeFactor).toBeCloseTo(2.2, 10);
    expect(h[4].easeFactor).toBeCloseTo(2.3, 10);
    expect(h[4].interval).toBe(6); // after a miss the ladder restarts: 1 day, then 6
  });

  it("sanitizeProgress repairs values saved by the buggy version, keeping counts", () => {
    const corrupted = { ...newWordProgress("w"), correctCount: 5, easeFactor: 4321.5, interval: 987654 };
    const fixed = sanitizeProgress(corrupted);
    expect(fixed.correctCount).toBe(5);
    expect(fixed.easeFactor).toBe(EASE_FACTOR_MAX);
    expect(fixed.interval).toBe(MAX_INTERVAL_DAYS);
    const junk = sanitizeProgress({ ...newWordProgress("w"), easeFactor: NaN, interval: -3 } as never);
    expect(junk).toMatchObject({ easeFactor: INITIAL_EASE_FACTOR, interval: 0 });
  });
});
