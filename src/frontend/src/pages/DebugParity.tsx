/**
 * DEV-ONLY page (route /debug/parity, not registered in production builds).
 * Runs the same uploaded clip through both landmark paths and reports how far
 * they drift apart:
 *   IMAGE mode  = what Python extraction used (independent detection on the 32 sampled frames)
 *   VIDEO mode  = what the live camera uses (tracking across every frame, then the same 32 sampled)
 */
import { useState } from "react";
import { createHolisticLandmarker, loadImageLandmarker, loadModels } from "@/lib/inference/loader";
import { runTta } from "@/lib/inference/model";
import { denseToOriginal } from "@/lib/inference/labels";
import { wordForClassId } from "@/data/words";
import { topKIndices } from "@/lib/inference/grading";
import { extractImageMode, extractVideoMode, loadVideoElement } from "@/lib/inference/extract";
import { Clip, J, normalizeBody } from "@/lib/inference/preprocess";

interface Group {
  name: string;
  joints: number[];
}
const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i);
const GROUPS: Group[] = [
  { name: "left hand (0–20)", joints: range(0, 21) },
  { name: "right hand (21–41)", joints: range(21, 42) },
  { name: "nose (42)", joints: [42] },
  { name: "shoulders (43–44)", joints: [43, 44] },
  { name: "elbows (45–46)", joints: [45, 46] },
];

/** Mean Euclidean xyz distance per joint over frames where both modes detected it. */
function perJointDiff(a: Clip, b: Clip) {
  const mean = new Float64Array(J).fill(NaN);
  const both = new Uint32Array(J);
  const maskAgree = new Float64Array(J);
  for (let j = 0; j < J; j++) {
    let sum = 0;
    let n = 0;
    let agree = 0;
    for (let t = 0; t < a.frames; t++) {
      const ma = a.mask[t * J + j];
      const mb = b.mask[t * J + j];
      if (ma === mb) agree++;
      if (ma && mb) {
        const o = (t * J + j) * 3;
        sum += Math.hypot(a.coords[o] - b.coords[o], a.coords[o + 1] - b.coords[o + 1], a.coords[o + 2] - b.coords[o + 2]);
        n++;
      }
    }
    both[j] = n;
    maskAgree[j] = agree / a.frames;
    if (n) mean[j] = sum / n;
  }
  return { mean, both, maskAgree };
}

const avg = (xs: number[]) => {
  const v = xs.filter((x) => Number.isFinite(x));
  return v.length ? v.reduce((s, x) => s + x, 0) / v.length : NaN;
};
const fmt = (x: number, d = 4) => (Number.isFinite(x) ? x.toFixed(d) : "n/a");

interface Report {
  raw: ReturnType<typeof perJointDiff>;
  norm: ReturnType<typeof perJointDiff>;
  imageTop: string;
  videoTop: string;
  imageMs: number;
  videoMs: number;
  frames: number;
}

declare global {
  interface Window {
    __parityResult?: unknown;
  }
}

async function topPrediction(clip: Clip) {
  const m = await loadModels();
  const pred = await runTta(m.ort, m.session, clip);
  const d2o = denseToOriginal(m.labels);
  return topKIndices(pred.probs, 3).map((d) => `${wordForClassId(d2o(d))} ${(pred.probs[d] * 100).toFixed(0)}%`);
}

export default function DebugParity() {
  const [status, setStatus] = useState("Choose a clip (same kind of file you'd upload in Practice).");
  const [report, setReport] = useState<Report | null>(null);

  const run = async (file: File) => {
    setReport(null);
    try {
      setStatus("Loading landmarkers…");
      const { video, revoke } = await loadVideoElement(file);
      const imgLm = await loadImageLandmarker();
      const vidLm = await createHolisticLandmarker("VIDEO"); // fresh: no leftover tracking state
      try {
        setStatus("Running IMAGE mode on 32 sampled frames…");
        let t = performance.now();
        const imageClip = await extractImageMode(video, imgLm);
        const imageMs = performance.now() - t;
        setStatus(`Running VIDEO mode over all ~${Math.floor(video.duration * 30)} frames…`);
        t = performance.now();
        const videoClip = await extractVideoMode(video, vidLm);
        const videoMs = performance.now() - t;
        const [imgTop, vidTop] = [await topPrediction(imageClip), await topPrediction(videoClip)];
        const norm = perJointDiff(normalizeBody(imageClip), normalizeBody(videoClip));
        window.__parityResult = {
          imageTop3: imgTop,
          videoTop3: vidTop,
          meanNormDiff: avg(Array.from(norm.mean)),
          maskAgree: avg(Array.from(norm.maskAgree)),
        };
        setReport({
          imageTop: imgTop.join(", "),
          videoTop: vidTop.join(", "),
          raw: perJointDiff(imageClip, videoClip),
          norm: perJointDiff(normalizeBody(imageClip), normalizeBody(videoClip)),
          imageMs,
          videoMs,
          frames: imageClip.frames,
        });
        setStatus("Done.");
      } finally {
        vidLm.close();
        revoke();
      }
    } catch (e) {
      setStatus(`Failed: ${(e as Error).message}`);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-bold">Landmark parity: IMAGE vs VIDEO mode (dev only)</h1>
      <p className="text-sm text-gray-600">
        Python extraction ran HolisticLandmarker in IMAGE mode on 32 sampled frames; the browser runs VIDEO mode
        (tracking). This compares both on the same clip. “Normalized” distances are in shoulder-widths (the units the
        model sees), so ~0.02 means ~2% of shoulder width.
      </p>
      <input type="file" accept="video/*" onChange={(e) => e.target.files?.[0] && run(e.target.files[0])} />
      <p className="text-sm">{status}</p>
      {report && (
        <>
          <table className="w-full text-sm border">
            <thead>
              <tr className="bg-gray-100 text-left">
                <th className="p-2">Joint group</th>
                <th className="p-2">Mean diff (raw, frame-units)</th>
                <th className="p-2">Mean diff (normalized, shoulder-widths)</th>
                <th className="p-2">Detection agreement</th>
              </tr>
            </thead>
            <tbody>
              {GROUPS.map((g) => (
                <tr key={g.name} className="border-t">
                  <td className="p-2">{g.name}</td>
                  <td className="p-2">{fmt(avg(g.joints.map((j) => report.raw.mean[j])))}</td>
                  <td className="p-2">{fmt(avg(g.joints.map((j) => report.norm.mean[j])))}</td>
                  <td className="p-2">{fmt(avg(g.joints.map((j) => report.raw.maskAgree[j])) * 100, 1)}%</td>
                </tr>
              ))}
              <tr className="border-t font-semibold">
                <td className="p-2">All joints (mean of per-joint means)</td>
                <td className="p-2">{fmt(avg(Array.from(report.raw.mean)))}</td>
                <td className="p-2">{fmt(avg(Array.from(report.norm.mean)))}</td>
                <td className="p-2">{fmt(avg(Array.from(report.raw.maskAgree)) * 100, 1)}%</td>
              </tr>
            </tbody>
          </table>
          <p className="text-sm">
            <b>IMAGE top-3:</b> {report.imageTop}
            <br />
            <b>VIDEO top-3:</b> {report.videoTop}
          </p>
          <p className="text-xs text-gray-500">
            IMAGE {report.imageMs.toFixed(0)} ms (32 frames) · VIDEO {report.videoMs.toFixed(0)} ms (all frames).
            Joints only compared on frames where both modes detected them.
          </p>
          <details>
            <summary className="cursor-pointer text-sm">Per-joint (normalized)</summary>
            <pre className="text-xs">
              {Array.from(report.norm.mean)
                .map((m, j) => `${String(j).padStart(2)}: ${fmt(m)}  (n=${report.norm.both[j]})`)
                .join("\n")}
            </pre>
          </details>
        </>
      )}
    </div>
  );
}
