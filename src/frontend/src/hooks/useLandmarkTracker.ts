import { RefObject, useCallback, useEffect, useRef, useState } from "react";
import type { HolisticLandmarker } from "@mediapipe/tasks-vision";
import { INFERENCE_CONFIG } from "@/lib/inference/config";
import { FRAMING_HINTS, FramingIssue, checkFraming, resultTo47 } from "@/lib/inference/landmarks";
import { FrameLandmarks } from "@/lib/inference/preprocess";
import { drawSkeleton } from "@/lib/inference/overlay";

/** Reads a design token (CSS custom property) so canvas drawing uses the same colours as the UI. */
const tokenColor = (name: string, fallback: string): string =>
  (typeof document !== "undefined" && getComputedStyle(document.documentElement).getPropertyValue(name).trim()) || fallback;

interface Options {
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  landmarker: HolisticLandmarker | null;
  /** Live tracking runs only while the camera is showing (not while replaying a clip). */
  active: boolean;
  /** Frames are appended to the recording buffer while this is true. */
  recording: boolean;
  /** Draw the skeleton overlay (tracking + framing checks run regardless). */
  drawOverlay: boolean;
}

export interface TrackerState {
  /** Friendly hint to show, or null when framing is fine. */
  hint: string | null;
  framingOk: boolean;
  fps: number;
}

/**
 * Runs HolisticLandmarker in VIDEO mode on the live camera, draws the skeleton
 * overlay, evaluates framing, and buffers per-frame 47-point landmarks while recording.
 */
export function useLandmarkTracker({ videoRef, canvasRef, landmarker, active, recording, drawOverlay }: Options) {
  const recordedRef = useRef<FrameLandmarks[]>([]);
  const recordingRef = useRef(recording);
  recordingRef.current = recording;
  const drawOverlayRef = useRef(drawOverlay);
  drawOverlayRef.current = drawOverlay;
  const [state, setState] = useState<TrackerState>({ hint: null, framingOk: false, fps: 0 });

  const startRecording = useCallback(() => {
    recordedRef.current = [];
  }, []);
  const takeRecording = useCallback(() => {
    const frames = recordedRef.current;
    recordedRef.current = [];
    return frames;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!landmarker || !active) {
      if (canvas) drawSkeleton(canvas, null, 0, 0, "");
      setState({ hint: null, framingOk: false, fps: 0 });
      return;
    }

    let raf = 0;
    let stopped = false;
    let lastMediaTime = -1;
    let lastTs = 0;
    let fpsEma = 0;
    let lastFrameAt = 0;
    let lastUiUpdate = 0;
    let issueSince: { issue: FramingIssue | null; at: number } = { issue: null, at: performance.now() };
    let shownHint: string | null = null;

    const tick = () => {
      if (stopped) return;
      raf = requestAnimationFrame(tick);
      const video = videoRef.current;
      if (!video || video.readyState < 2 || !video.videoWidth || !video.srcObject) return;
      if (video.currentTime === lastMediaTime) return; // no new camera frame yet
      lastMediaTime = video.currentTime;

      const now = performance.now();
      const ts = Math.max(now, lastTs + 1); // detectForVideo needs strictly increasing timestamps
      lastTs = ts;
      let frame: FrameLandmarks;
      try {
        frame = resultTo47(landmarker.detectForVideo(video, ts));
      } catch (e) {
        console.warn("[koala] landmarker frame failed", e);
        return;
      }

      if (recordingRef.current && recordedRef.current.length < INFERENCE_CONFIG.maxRecordedFrames) {
        recordedRef.current.push(frame);
      }

      if (lastFrameAt) {
        const inst = 1000 / (now - lastFrameAt);
        fpsEma = fpsEma ? fpsEma * 0.9 + inst * 0.1 : inst;
      }
      lastFrameAt = now;

      const framing = checkFraming(frame);
      const top = framing.issues[0] ?? null;
      if (top !== issueSince.issue) issueSince = { issue: top, at: now };
      // Show a hint only after it has persisted (and clear it immediately once fixed).
      if (top === null) shownHint = null;
      else if (now - issueSince.at >= INFERENCE_CONFIG.framing.hintDebounceMs) shownHint = FRAMING_HINTS[top];

      if (canvas) {
        drawSkeleton(
          canvas,
          drawOverlayRef.current ? frame : null,
          video.videoWidth,
          video.videoHeight,
          framing.ok ? tokenColor("--sage-200", "white") : "white",
        );
      }
      if (now - lastUiUpdate > 250) {
        lastUiUpdate = now;
        setState((s) =>
          s.hint === shownHint && s.framingOk === framing.ok && Math.abs(s.fps - fpsEma) < 0.5
            ? s
            : { hint: shownHint, framingOk: framing.ok, fps: fpsEma },
        );
      }
    };
    raf = requestAnimationFrame(tick);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      if (canvas) drawSkeleton(canvas, null, 0, 0, "");
    };
  }, [landmarker, active, videoRef, canvasRef]);

  return { ...state, startRecording, takeRecording };
}
