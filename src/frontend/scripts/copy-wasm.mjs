// Copies the WASM runtimes for @mediapipe/tasks-vision and onnxruntime-web
// into public/wasm/ so the app serves them same-origin (no CDN calls).
// public/wasm/ is git-ignored; this runs on postinstall, predev and prebuild.
import { cpSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const jobs = [
  ["node_modules/@mediapipe/tasks-vision/wasm", "public/wasm/mediapipe", (f) => f.startsWith("vision_wasm")],
  ["node_modules/onnxruntime-web/dist", "public/wasm/ort", (f) => /^ort-wasm-simd-threaded(\.jsep)?\.(mjs|wasm)$/.test(f)],
];
import { readdirSync } from "node:fs";
for (const [src, dst, keep] of jobs) {
  const from = join(root, src);
  if (!existsSync(from)) {
    console.warn(`copy-wasm: ${src} not found (run npm install first)`);
    continue;
  }
  mkdirSync(join(root, dst), { recursive: true });
  for (const f of readdirSync(from).filter(keep)) cpSync(join(from, f), join(root, dst, f));
}
console.log("copy-wasm: done");
