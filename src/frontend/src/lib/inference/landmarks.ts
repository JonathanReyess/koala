/**
 * MediaPipe HolisticLandmarker result -> 47-point layout used in training
 * (see extract_landmarks.result_to_47pt): left hand 0-20, right hand 21-41,
 * pose [0,11,12,13,14] at 42-46. Missing groups stay exactly 0 with mask 0.
 */
import { INFERENCE_CONFIG } from "./config";
import { FrameLandmarks, J, LEFT_SHOULDER, RIGHT_SHOULDER, NOSE } from "./preprocess";

interface LM {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/** Minimal structural view of HolisticLandmarkerResult (avoids importing the package in tests). */
export interface HolisticResultLike {
  poseLandmarks?: LM[][];
  leftHandLandmarks?: LM[][];
  rightHandLandmarks?: LM[][];
}

export function resultTo47(result: HolisticResultLike): FrameLandmarks {
  const coords = new Float32Array(J * 3);
  const mask = new Uint8Array(J);
  const pose = result.poseLandmarks?.[0];
  const left = result.leftHandLandmarks?.[0];
  const right = result.rightHandLandmarks?.[0];

  const put = (slot: number, lm: LM) => {
    coords[slot * 3] = lm.x;
    coords[slot * 3 + 1] = lm.y;
    coords[slot * 3 + 2] = lm.z;
    mask[slot] = 1;
  };
  if (left && left.length >= 21) for (let i = 0; i < 21; i++) put(i, left[i]);
  if (right && right.length >= 21) for (let i = 0; i < 21; i++) put(21 + i, right[i]);
  if (pose && pose.length >= 33) {
    INFERENCE_CONFIG.poseIndices.forEach((pi, i) => put(42 + i, pose[pi]));
  }
  return { coords, mask };
}

// --- Live framing checks -----------------------------------------------------

export type FramingIssue = "no_body" | "shoulders" | "no_hands" | "too_small";

export interface FramingStatus {
  ok: boolean;
  issues: FramingIssue[];
}

export const FRAMING_HINTS: Record<FramingIssue, string> = {
  no_body: "We can't see you yet — step into the frame.",
  shoulders: "Move back a little so both shoulders are in view.",
  no_hands: "Raise your hand(s) into the frame.",
  too_small: "Move a bit closer to the camera.",
};

/** Priority order: the first issue is the one shown as the hint. */
const ISSUE_ORDER: FramingIssue[] = ["no_body", "shoulders", "too_small", "no_hands"];

export function checkFraming(frame: FrameLandmarks): FramingStatus {
  const { minShoulderWidth, edgeMargin } = INFERENCE_CONFIG.framing;
  const issues = new Set<FramingIssue>();
  const has = (j: number) => frame.mask[j] === 1;
  const inFrame = (j: number) => {
    const x = frame.coords[j * 3];
    const y = frame.coords[j * 3 + 1];
    return x > edgeMargin && x < 1 - edgeMargin && y > edgeMargin && y < 1 - edgeMargin;
  };

  if (!has(NOSE) && !has(LEFT_SHOULDER) && !has(RIGHT_SHOULDER)) {
    issues.add("no_body");
  } else if (!(has(LEFT_SHOULDER) && has(RIGHT_SHOULDER) && inFrame(LEFT_SHOULDER) && inFrame(RIGHT_SHOULDER))) {
    issues.add("shoulders");
  } else {
    const dx = frame.coords[LEFT_SHOULDER * 3] - frame.coords[RIGHT_SHOULDER * 3];
    const dy = frame.coords[LEFT_SHOULDER * 3 + 1] - frame.coords[RIGHT_SHOULDER * 3 + 1];
    if (Math.hypot(dx, dy) < minShoulderWidth) issues.add("too_small");
  }
  let anyHand = false;
  for (let j = 0; j < 42; j++) if (frame.mask[j]) { anyHand = true; break; }
  if (!anyHand) issues.add("no_hands");

  const ordered = ISSUE_ORDER.filter((i) => issues.has(i));
  return { ok: ordered.length === 0, issues: ordered };
}

export interface DetectionFractions {
  pose: number;
  anyHand: number;
}

/** Fractions over the (already sampled) clip frames — same stats as manifest.csv. */
export function detectionFractions(mask: Uint8Array, frames: number): DetectionFractions {
  let pose = 0;
  let hand = 0;
  for (let t = 0; t < frames; t++) {
    let p = 0;
    let h = 0;
    for (let j = 0; j < 42; j++) if (mask[t * J + j]) { h = 1; break; }
    for (let j = 42; j < J; j++) if (mask[t * J + j]) { p = 1; break; }
    pose += p;
    hand += h;
  }
  return frames ? { pose: pose / frames, anyHand: hand / frames } : { pose: 0, anyHand: 0 };
}
