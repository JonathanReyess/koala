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
  /** MediaPipe delegate actually in use for the live landmarker. */
  delegate?: string;
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

export type Delegate = "GPU" | "CPU";

/** `?delegate=gpu|cpu` overrides the config default (used for benchmarking). */
export function preferredDelegate(): Delegate {
  const q = new URLSearchParams(location.search).get("delegate")?.toLowerCase();
  if (q === "gpu") return "GPU";
  if (q === "cpu") return "CPU";
  return INFERENCE_CONFIG.landmarkerDelegate;
}

/**
 * Creates a HolisticLandmarker with the preferred delegate. If the GPU delegate is unavailable
 * (creation throws, or a warm-up detection on a blank frame throws) it falls back to CPU (XNNPACK, the
 * delegate Python extraction used). Returns which delegate is actually in use.
 */
export async function createHolisticLandmarker(
  runningMode: "VIDEO" | "IMAGE",
  delegate: Delegate = preferredDelegate(),
): Promise<HolisticLandmarker & { delegateUsed?: Delegate }> {
  const { FilesetResolver, HolisticLandmarker } = await import("@mediapipe/tasks-vision");
  const fileset = await FilesetResolver.forVisionTasks(INFERENCE_CONFIG.mediapipeWasmDir);
  const conf = INFERENCE_CONFIG.minDetectionConfidence;
  const make = async (d: Delegate) => {
    const lm: HolisticLandmarker & { delegateUsed?: Delegate } = await HolisticLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: INFERENCE_CONFIG.holisticTaskUrl, delegate: d },
      runningMode,
      minPoseDetectionConfidence: conf,
      minHandLandmarksConfidence: conf,
    });
    if (d === "GPU") {
      // Creation can "succeed" on machines where the first inference then fails; probe once.
      const probe = document.createElement("canvas");
      probe.width = probe.height = 64;
      try {
        if (runningMode === "VIDEO") lm.detectForVideo(probe, 0);
        else lm.detect(probe);
      } catch (e) {
        lm.close();
        throw e;
      }
    }
    lm.delegateUsed = d;
    return lm;
  };
  if (delegate === "CPU") return make("CPU");
  try {
    return await make("GPU");
  } catch (e) {
    console.warn("[koala] GPU delegate unavailable, falling back to CPU", e);
    return make("CPU");
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
        delegate: landmarker.r.delegateUsed,
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
