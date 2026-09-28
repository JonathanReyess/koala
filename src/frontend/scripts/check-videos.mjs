#!/usr/bin/env node
// Verifies every video VideoExampleCard/Learn.tsx can request actually exists
// in public/videos, with exact-case filenames (the frontend always requests
// lowercase `.mp4`, and case-sensitive hosts like Vercel/Linux won't fall
// back to a differently-cased file the way a case-insensitive dev machine does).
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));
const learnPath = path.join(root, "..", "src", "pages", "Learn.tsx");
const videosDir = path.join(root, "..", "public", "videos");

const learnSrc = readFileSync(learnPath, "utf8");
const wordListMatch = learnSrc.match(
  /const WORD_LIST = \[([\s\S]*?)\] as const;/,
);
if (!wordListMatch) {
  console.error("Could not find WORD_LIST in Learn.tsx");
  process.exit(1);
}

const englishWords = [
  ...wordListMatch[1].matchAll(/english:\s*"([^"]+)"/g),
].map((m) => m[1]);

if (englishWords.length === 0) {
  console.error("Parsed WORD_LIST but found no english words");
  process.exit(1);
}

const actualFiles = new Set(readdirSync(videosDir));
const missing = [];

for (const word of englishWords) {
  for (const example of [1, 2]) {
    const expected = `${word}_example${example}.mp4`;
    if (!actualFiles.has(expected)) {
      // Case-insensitive fallback lookup so we can report "wrong case" vs "missing".
      const caseInsensitiveHit = [...actualFiles].find(
        (f) => f.toLowerCase() === expected.toLowerCase(),
      );
      missing.push(
        caseInsensitiveHit
          ? `${expected}  (found as "${caseInsensitiveHit}" — case mismatch)`
          : `${expected}  (not found)`,
      );
    }
  }
}

if (missing.length > 0) {
  console.error(
    `check-videos: ${missing.length} practiceable word(s) reference a missing/mismatched video:\n`,
  );
  for (const m of missing) console.error(`  - ${m}`);
  process.exit(1);
}

console.log(
  `check-videos: OK — all ${englishWords.length} practiceable words have example1/example2 videos with matching case.`,
);
