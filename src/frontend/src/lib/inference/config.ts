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
  /** Preferred MediaPipe delegate for the landmarker. CPU by default (GPU's first shader compile can take 10+ s); ?delegate=gpu opts in, and GPU falls back to CPU if unavailable. */
  landmarkerDelegate: "CPU" as "CPU" | "GPU",

  // --- Grading -------------------------------------------------------------
  grading: {
    /** not_detected if pose is present in fewer than this fraction of the 32 frames. */
    minPoseFrac: 0.6,
    /** not_detected if at least one hand is present in fewer than this fraction of frames. */
    minAnyHandFrac: 0.3,
    /** "close" if the target is within the top-K (and CLOSE_MIN is met) but not accepted as correct. */
    closeTopK: 3,
    // Confidence gates on the averaged (TTA) softmax probabilities. Calibrated on 3,684 held-out predictions
    // (15 config-f signer_kfold folds, TTA mirror) with scripts/calibrate_thresholds.py; sweep tables and reasoning
    // in RESULTS.md ("Grading thresholds"). Calibration data = fluent held-out signers only.
    //  - CORRECT_MIN 0.20: an earlier 0.50 (the script's suggestion) was rolled back after learner testing ("study"
    //    graded close at 41%, "when" at 34.7%). On the sweep 0.20 keeps 99.9% of correct top-1 predictions with
    //    0.3% false accepts.
    //  - CLOSE_MIN 0.10: script suggested 0.05; raised by hand (false-close 0.8% vs 1.2%) so random movement doesn't
    //    easily reach "Almost".
    //  - CONFUSION_MIN 0.75: lowest t with >= 95% naming precision. Also the bar for naming any word in a message.
    /** correct: target is top-1 AND its probability >= this. */
    CORRECT_MIN: 0.2,
    /** close: target in top-K AND its probability >= this. */
    CLOSE_MIN: 0.1,
    /** confident confusion: top-1 != target AND top-1 probability >= this -> we name the word. */
    CONFUSION_MIN: 0.75,
  },

  // --- Live framing checks (hints only; never block recording) ------------
  // Hands are deliberately NOT checked live: signers often start with hands out of frame.
  // Hand visibility is only judged after Stop (grading.minAnyHandFrac).
  framing: {
    /** Shoulder-to-shoulder distance as a fraction of frame width. */
    minShoulderWidth: 0.12,
    /** A shoulder/nose must sit inside [margin, 1 - margin] to count as "in frame". */
    edgeMargin: 0.02,
    /** A problem must persist this long (ms) before a hint is shown (avoids flicker). */
    hintDebounceMs: 1500,
  },

  // --- Live recording ------------------------------------------------------
  /** Safety cap on stored frames per recording (~2 min at 30 fps). */
  maxRecordedFrames: 3600,
  /** Frame rate assumed for uploaded clips (no reliable frame count in browsers). */
  uploadAssumedFps: 30,
} as const;

/** Widened to `number` so callers/tests can supply their own thresholds. */
export type GradingConfig = { [K in keyof typeof INFERENCE_CONFIG.grading]: number };
