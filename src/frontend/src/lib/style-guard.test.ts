import { describe, it, expect } from "vitest";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

/** Repo-wide style rules: no beige page colour and no em dashes anywhere in tracked text files. */
const repo = resolve(__dirname, "../../../..");
const tracked = execSync("git ls-files", { cwd: repo, encoding: "utf8" })
  .split("\n")
  .filter((f) => f && !/(package-lock\.json|bun\.lockb)$/.test(f) && !/\.(mp4|onnx|task|png|jpg|jpeg|ico|pkl|p|pt|wasm|npz|npy|heic|svg|webp)$/i.test(f));

const textOf = (f: string) => {
  try {
    const buf = readFileSync(resolve(repo, f));
    return buf.includes(0) ? "" : buf.toString("utf8");
  } catch {
    return ""; // deleted in the working tree
  }
};

describe("style guard", () => {
  const EM_DASH = String.fromCharCode(0x2014);
  it("contains no em dashes", () => {
    const offenders = tracked.filter((f) => textOf(f).includes(EM_DASH));
    expect(offenders).toEqual([]);
  });

  it("uses no beige page colour (#f6efe5 / hsl(37 47% 93%) or the cream tile tint)", () => {
    const banned = /f6efe5|f4efe6|37 47% 93%/i;
    const offenders = tracked.filter((f) => f !== "src/frontend/src/lib/style-guard.test.ts" && banned.test(textOf(f)));
    expect(offenders).toEqual([]);
  });
});
