/**
 * Single place for every tunable in the in-browser inference pipeline.
 * Preprocessing constants mirror preprocessing_spec.json (do not change them
 * without regenerating the model); grading thresholds are product decisions.
 */

export const INFERENCE_CONFIG = {
  // --- Contract with the exported model / preprocessing_spec.json ----------
  sequenceLength: 32,
  numJoints: 47,
  /** BlazePose indices copied into slots 42..46 (nose, shoulders, elbows). */
  poseIndices: [0, 11, 12, 13, 14] as const,

  // --- Assets (served from /public, no network after first load) -----------
  modelUrl: "/models/ksl_f/model.onnx",
  labelMapUrl: "/models/ksl_f/label_map.json",
  holisticTaskUrl: "/models/holistic_landmarker.task",
  mediapipeWasmDir: "/wasm/mediapipe",
  ortWasmDir: "/wasm/ort/",

  // --- Landmarker options (extract_landmarks.py default is 0.5) -----------
  minDetectionConfidence: 0.5,

  // --- Grading -------------------------------------------------------------
  grading: {
    /** not_detected if pose is present in fewer than this fraction of the 32 frames. */
    minPoseFrac: 0.6,
    /** not_detected if at least one hand is present in fewer than this fraction of frames. */
    minAnyHandFrac: 0.3,
    /** "close" if the target is within the top-K but not top-1. */
    closeTopK: 3,
  },

  // --- Live framing checks (hints only; never block recording) ------------
  framing: {
    /** Shoulder-to-shoulder distance as a fraction of frame width. */
    minShoulderWidth: 0.12,
    /** A shoulder/nose must sit inside [margin, 1 - margin] to count as "in frame". */
    edgeMargin: 0.02,
    /** Hint is only shown after the problem persists this many ms (avoids flicker). */
    hintDebounceMs: 400,
  },

  // --- Live recording ------------------------------------------------------
  /** Safety cap on stored frames per recording (~2 min at 30 fps). */
  maxRecordedFrames: 3600,
  /** Frame rate assumed for uploaded clips (no reliable frame count in browsers). */
  uploadAssumedFps: 30,
} as const;

export type GradingConfig = typeof INFERENCE_CONFIG.grading;
