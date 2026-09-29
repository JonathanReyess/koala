/**
 * In-memory log of graded attempts for `?debug=1` sessions - becomes a local learner test set.
 * Nothing is persisted or uploaded: entries live in this module until the page is reloaded, or are
 * exported explicitly via the "Download attempts (JSON)" button.
 *
 * `raw_landmarks` / `raw_mask` use the same nested layout as golden_vectors.json
 * ((32, 47, 3) and (32, 47), RAW = before normalize_body) so the same tooling can replay them.
 */
import { INFERENCE_CONFIG } from "./config";
import type { Grade } from "./grading";
import type { DetectionFractions } from "./landmarks";
import { Clip, J } from "./preprocess";

export interface AttemptRecord {
  format: "koala-attempt-v1";
  timestamp: string;
  source: "recorded" | "upload";
  target_word: string;
  target_class_id: number;
  top5: { class_id: number; word: string; prob: number }[];
  grade: {
    status: Grade["status"];
    reason: Grade["reason"] | null;
    top1_class_id: number | null;
    top1_prob: number | null;
    target_prob: number | null;
    names_top1: boolean | null;
  };
  thresholds: Record<string, number>;
  detection_fractions: DetectionFractions;
  delegate: string | null;
  ort_backend: string | null;
  /** Landmarker fps at the moment of Stop (live recordings only). */
  live_fps: number | null;
  raw_landmarks: number[][][];
  raw_mask: number[][];
}

export interface AttemptInput {
  source: AttemptRecord["source"];
  targetWord: string;
  targetClassId: number;
  top5: { classId: number; prob: number }[];
  grade: Grade;
  fractions: DetectionFractions;
  wordFor: (classId: number) => string;
  delegate?: string;
  backend?: string;
  liveFps?: number | null;
  clip: Clip;
  now?: Date;
}

export function buildAttemptRecord(a: AttemptInput): AttemptRecord {
  const raw_landmarks: number[][][] = [];
  const raw_mask: number[][] = [];
  for (let t = 0; t < a.clip.frames; t++) {
    const frame: number[][] = [];
    const mrow: number[] = [];
    for (let j = 0; j < J; j++) {
      const o = (t * J + j) * 3;
      frame.push([a.clip.coords[o], a.clip.coords[o + 1], a.clip.coords[o + 2]]);
      mrow.push(a.clip.mask[t * J + j]);
    }
    raw_landmarks.push(frame);
    raw_mask.push(mrow);
  }
  return {
    format: "koala-attempt-v1",
    timestamp: (a.now ?? new Date()).toISOString(),
    source: a.source,
    target_word: a.targetWord,
    target_class_id: a.targetClassId,
    top5: a.top5.map((r) => ({ class_id: r.classId, word: a.wordFor(r.classId), prob: r.prob })),
    grade: {
      status: a.grade.status,
      reason: a.grade.reason ?? null,
      top1_class_id: a.grade.top1 ?? null,
      top1_prob: a.grade.top1Prob ?? null,
      target_prob: a.grade.targetProb ?? null,
      names_top1: a.grade.namesTop1 ?? null,
    },
    thresholds: { ...INFERENCE_CONFIG.grading },
    detection_fractions: a.fractions,
    delegate: a.delegate ?? null,
    ort_backend: a.backend ?? null,
    live_fps: a.liveFps ?? null,
    raw_landmarks,
    raw_mask,
  };
}

const attempts: AttemptRecord[] = [];
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());

export function logAttempt(record: AttemptRecord): void {
  attempts.push(record);
  notify();
}
export function attemptCount(): number {
  return attempts.length;
}
export function clearAttempts(): void {
  attempts.length = 0;
  notify();
}
export function subscribeAttempts(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
export function attemptsToJson(): string {
  return JSON.stringify({ format: "koala-attempts-v1", exported_at: new Date().toISOString(), attempts });
}
