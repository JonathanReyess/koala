#!/usr/bin/env node
// Verifies public/videos against src/data/vocab.json:
//  - every clip is named <english>_example1.mp4 / _example2.mp4 for a known vocab word, with exact-case
//    names and a lowercase .mp4 (case-sensitive hosts like Vercel/Linux won't fall back the way a
//    case-insensitive dev machine does), and each word has BOTH clips;
//  - src/data/video-manifest.json is up to date (regenerate with scripts/build-video-manifest.mjs).
// Words without clips are fine (they're simply not offered yet); they're listed for information.
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const vocab = JSON.parse(readFileSync(path.join(root, "src", "data", "vocab.json"), "utf8")).words;
const manifest = JSON.parse(readFileSync(path.join(root, "src", "data", "video-manifest.json"), "utf8")).words;
// Provenance files written by scripts/make_demo_clips.py live next to the clips.
const METADATA_FILES = new Set(["ATTRIBUTION.txt", "demo_clips_manifest.csv"]);
const files = readdirSync(path.join(root, "public", "videos")).filter((f) => !f.startsWith(".") && !METADATA_FILES.has(f));
const fileSet = new Set(files);
const english = new Set(vocab.map((w) => w.english));
const problems = [];

for (const f of files) {
  const m = f.match(/^(.*)_example([12])\.mp4$/);
  if (!m) {
    const ci = f.toLowerCase().match(/^(.*)_example[12]\.mp4$/);
    problems.push(`${f}: not named <word>_example1/2.mp4${ci ? " (extension/case mismatch)" : ""}`);
    continue;
  }
  if (!english.has(m[1])) {
    const near = [...english].find((w) => w.toLowerCase() === m[1].toLowerCase());
    problems.push(`${f}: no vocab word "${m[1]}"${near ? ` (case mismatch with "${near}")` : ""}`);
  }
}
const stems = new Set(files.map((f) => f.match(/^(.*)_example[12]\.mp4$/)?.[1]).filter(Boolean));
for (const w of stems) {
  for (const n of [1, 2]) if (!fileSet.has(`${w}_example${n}.mp4`)) problems.push(`${w}_example${n}.mp4 is missing (only one clip found)`);
}
const expected = [...stems].filter((w) => fileSet.has(`${w}_example1.mp4`) && fileSet.has(`${w}_example2.mp4`)).sort();
if (JSON.stringify(expected) !== JSON.stringify(manifest)) {
  problems.push("src/data/video-manifest.json is stale, run: node scripts/build-video-manifest.mjs");
}

if (problems.length) {
  console.error(`check-videos: ${problems.length} problem(s):\n`);
  for (const p of problems) console.error(`  - ${p}`);
  process.exit(1);
}
const missing = vocab.filter((w) => !manifest.includes(w.english)).map((w) => w.english);
console.log(`check-videos: OK: ${manifest.length}/${vocab.length} words have both clips.`);
if (missing.length) console.log(`  not yet practiceable (no clips): ${missing.join(", ")}`);
