/**
 * Offline (file-based) landmark extraction for uploaded clips and the parity
 * debug page. Live camera capture lives in useLandmarkTracker.
 */
import type { HolisticLandmarker } from "@mediapipe/tasks-vision";
import { INFERENCE_CONFIG } from "./config";
import { resultTo47 } from "./landmarks";
import { Clip, FrameLandmarks, buildClip, sampleFrameIndices } from "./preprocess";

/** Loads a blob into an off-screen <video> and waits until its duration is known. */
export async function loadVideoElement(blob: Blob): Promise<{ video: HTMLVideoElement; revoke: () => void }> {
  const url = URL.createObjectURL(blob);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = url;
  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Could not read this video file."));
  });
  if (!Number.isFinite(video.duration)) {
    // MediaRecorder WebM files report Infinity until the end has been seen.
    await new Promise<void>((resolve) => {
      video.ontimeupdate = () => {
        video.ontimeupdate = null;
        resolve();
      };
      video.currentTime = 1e101;
    });
    video.currentTime = 0;
  }
  return { video, revoke: () => URL.revokeObjectURL(url) };
}

export function seekTo(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve, reject) => {
    if (Math.abs(video.currentTime - t) < 1e-6) return resolve();
    const done = () => {
      video.removeEventListener("seeked", done);
      resolve();
    };
    video.addEventListener("seeked", done);
    video.onerror = () => reject(new Error("Seek failed"));
    video.currentTime = t;
  });
}

/** Total frames assumed for a file (browsers don't expose a reliable frame count). */
export function assumedTotalFrames(video: HTMLVideoElement, fps = INFERENCE_CONFIG.uploadAssumedFps): number {
  return Math.max(1, Math.floor(video.duration * fps));
}

/**
 * IMAGE mode on the 32 spec-sampled frames - mirrors extract_landmarks.py
 * (seek to each sampled frame, detect independently, no tracking state).
 */
export async function extractImageMode(
  video: HTMLVideoElement,
  landmarker: HolisticLandmarker,
  fps = INFERENCE_CONFIG.uploadAssumedFps,
): Promise<Clip> {
  const total = assumedTotalFrames(video, fps);
  const idx = sampleFrameIndices(total);
  const frames: FrameLandmarks[] = [];
  for (const i of idx) {
    // Clamp just inside the end so the seek lands on a real frame.
    await seekTo(video, Math.min(i / fps, Math.max(0, video.duration - 1e-3)));
    frames.push(resultTo47(landmarker.detect(video)));
  }
  // frames is already the sampled set: 1:1 into the clip.
  return buildClipFromSampled(frames);
}

/**
 * VIDEO mode (tracking) over every frame in order, then spec-sample 32 of the
 * recorded frames - the same thing the live camera path does, so uploads and
 * live recordings go through one pipeline. Pass a FRESH VIDEO-mode landmarker
 * so no tracking state leaks in from earlier runs.
 */
export async function extractVideoMode(
  video: HTMLVideoElement,
  landmarker: HolisticLandmarker,
  fps = INFERENCE_CONFIG.uploadAssumedFps,
): Promise<Clip> {
  const total = assumedTotalFrames(video, fps);
  const frames: FrameLandmarks[] = [];
  for (let k = 0; k < total; k++) {
    await seekTo(video, Math.min(k / fps, Math.max(0, video.duration - 1e-3)));
    // +1: timestamps must be strictly increasing and a GPU warm-up probe may already have used 0.
    const ts = Math.round((k * 1000) / fps) + 1;
    frames.push(resultTo47(landmarker.detectForVideo(video, ts)));
  }
  return buildClip(frames);
}

function buildClipFromSampled(frames: FrameLandmarks[]): Clip {
  const T = INFERENCE_CONFIG.sequenceLength;
  const J = INFERENCE_CONFIG.numJoints;
  const coords = new Float32Array(T * J * 3);
  const mask = new Uint8Array(T * J);
  frames.forEach((f, t) => {
    coords.set(f.coords, t * J * 3);
    mask.set(f.mask, t * J);
  });
  return { coords, mask, frames: T };
}
