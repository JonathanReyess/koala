import type { FrameLandmarks } from "./preprocess";
import { LEFT_ELBOW, LEFT_SHOULDER, NOSE, RIGHT_ELBOW, RIGHT_SHOULDER } from "./preprocess";

// MediaPipe hand topology (21 points).
const HAND_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const POSE_EDGES: [number, number][] = [
  [LEFT_SHOULDER, RIGHT_SHOULDER],
  [LEFT_SHOULDER, LEFT_ELBOW],
  [RIGHT_SHOULDER, RIGHT_ELBOW],
];

/**
 * Draws the stored 47-point frame on a canvas that overlays an `object-cover`
 * <video>. Coordinates are normalized to the video frame, so they are mapped
 * through the same cover-crop the browser applies to the video.
 */
export function drawSkeleton(
  canvas: HTMLCanvasElement,
  frame: FrameLandmarks | null,
  videoW: number,
  videoH: number,
  color: string,
): void {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if (!frame || !videoW || !videoH) return;

  const scale = Math.max(cw / videoW, ch / videoH);
  const ox = (cw - videoW * scale) / 2;
  const oy = (ch - videoH * scale) / 2;
  const pt = (j: number) => [ox + frame.coords[j * 3] * videoW * scale, oy + frame.coords[j * 3 + 1] * videoH * scale];
  const has = (j: number) => frame.mask[j] === 1;

  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  for (const base of [0, 21]) {
    if (!has(base)) continue;
    ctx.beginPath();
    for (const [a, b] of HAND_EDGES) {
      const [x1, y1] = pt(base + a);
      const [x2, y2] = pt(base + b);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();
    for (let i = 0; i < 21; i++) {
      const [x, y] = pt(base + i);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.beginPath();
  for (const [a, b] of POSE_EDGES) {
    if (!has(a) || !has(b)) continue;
    const [x1, y1] = pt(a);
    const [x2, y2] = pt(b);
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
  }
  ctx.stroke();
  for (const j of [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW]) {
    if (!has(j)) continue;
    const [x, y] = pt(j);
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}
