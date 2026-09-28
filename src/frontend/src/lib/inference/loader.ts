/**
 * Lazy, cached loading of the ONNX session, label map and HolisticLandmarker.
 * Everything is fetched from same-origin /public assets; after the first load
 * the practice flow makes zero network calls. Promises are memoised so
 * re-entering Practice reuses the loaded models.
 */
import { INFERENCE_CONFIG } from "./config";
import type { LabelMap, OrtModule } from "./model";
import type { HolisticLandmarker } from "@mediapipe/tasks-vision";

export interface LoadTimings {
  modelMs: number;
  landmarkerMs: number;
  totalMs: number;
  backend: string;
}

export interface LoadedModels {
  ort: OrtModule;
  session: import("onnxruntime-web").InferenceSession;
  labels: LabelMap;
  landmarker: HolisticLandmarker;
  timings: LoadTimings;
}

let cached: Promise<LoadedModels> | null = null;

async function loadModel() {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.wasmPaths = INFERENCE_CONFIG.ortWasmDir;
  // Single-threaded: multi-threaded WASM needs cross-origin isolation headers.
  ort.env.wasm.numThreads = 1;
  const [modelBytes, labels] = await Promise.all([
    fetch(INFERENCE_CONFIG.modelUrl).then((r) => {
      if (!r.ok) throw new Error(`model fetch failed: ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch(INFERENCE_CONFIG.labelMapUrl).then((r) => {
      if (!r.ok) throw new Error(`label map fetch failed: ${r.status}`);
      return r.json() as Promise<LabelMap>;
    }),
  ]);
  // WASM is the default. WebGPU is opt-in (?webgpu=1) and only if the browser exposes it.
  const wantGpu = new URLSearchParams(location.search).has("webgpu") && "gpu" in navigator;
  const providers = wantGpu ? ["webgpu", "wasm"] : ["wasm"];
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: providers });
  return { ort, session, labels, backend: providers[0] };
}

export async function createHolisticLandmarker(runningMode: "VIDEO" | "IMAGE") {
  const { FilesetResolver, HolisticLandmarker } = await import("@mediapipe/tasks-vision");
  const fileset = await FilesetResolver.forVisionTasks(INFERENCE_CONFIG.mediapipeWasmDir);
  const conf = INFERENCE_CONFIG.minDetectionConfidence;
  const make = (delegate: "GPU" | "CPU") =>
    HolisticLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: INFERENCE_CONFIG.holisticTaskUrl, delegate },
      runningMode,
      minPoseDetectionConfidence: conf,
      minHandLandmarksConfidence: conf,
    });
  // CPU (XNNPACK) delegate matches Python's Delegate.CPU; GPU only as a fallback if CPU init fails.
  try {
    return await make("CPU");
  } catch {
    return make("GPU");
  }
}

export function loadModels(): Promise<LoadedModels> {
  if (!cached) {
    cached = (async () => {
      const t0 = performance.now();
      const [m, landmarker] = await Promise.all([
        (async () => {
          const s = performance.now();
          const r = await loadModel();
          return { ...r, ms: performance.now() - s };
        })(),
        (async () => {
          const s = performance.now();
          const r = await createHolisticLandmarker("VIDEO");
          return { r, ms: performance.now() - s };
        })(),
      ]);
      const timings: LoadTimings = {
        modelMs: m.ms,
        landmarkerMs: landmarker.ms,
        totalMs: performance.now() - t0,
        backend: m.backend,
      };
      console.info("[koala] models loaded", timings);
      return { ort: m.ort, session: m.session, labels: m.labels, landmarker: landmarker.r, timings };
    })().catch((e) => {
      cached = null; // allow retry
      throw e;
    });
  }
  return cached;
}

let imageLandmarker: Promise<HolisticLandmarker> | null = null;

/** IMAGE-mode landmarker (same mode Python extraction used); lazily created, cached. Used for uploaded clips + the parity page. */
export function loadImageLandmarker(): Promise<HolisticLandmarker> {
  if (!imageLandmarker) {
    imageLandmarker = createHolisticLandmarker("IMAGE").catch((e) => {
      imageLandmarker = null;
      throw e;
    });
  }
  return imageLandmarker;
}
