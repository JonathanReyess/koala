import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(__dirname, "../../..");
const css = readFileSync(resolve(root, "src/index.css"), "utf8");
const tw = readFileSync(resolve(root, "tailwind.config.ts"), "utf8");
const read = (p: string) => readFileSync(resolve(root, p), "utf8");

describe("design tokens", () => {
  const value = (name: string) => css.match(new RegExp(`--${name}:\\s*([^;]+);`))?.[1].trim();

  it("defines every token from the spec (two clash-avoiding renames documented in index.css)", () => {
    const expected: Record<string, string> = {
      bg: "#FBFCFB", surface: "#FFFFFF", "card-radius": "20px", "media-radius": "16px", "media-inset": "16px",
      ink: "#1B2430", "text-muted": "#6B7280", danger: "#E24B4A", "btn-radius": "999px", "btn-height": "44px", "chip-radius": "999px",
    };
    for (const [k, v] of Object.entries(expected)) expect(value(k), k).toBe(v);
    expect(value("card-shadow")).toBe("0 1px 2px rgba(16, 24, 40, 0.04), 0 8px 24px rgba(16, 24, 40, 0.06)");
    expect(value("card-border")).toBe("1px solid rgba(16, 24, 40, 0.06)");
    for (const s of ["700", "600", "200", "50"]) expect(value(`sage-${s}`), `sage-${s}`).toMatch(/^#[0-9a-f]{6}$/i);
  });

  it("is sage-only: the page canvas is near-white, sage-200 is the existing pale --primary", () => {
    expect(value("background")).toMatch(/^120 14% 98\.6%/); // = #FBFCFB
    expect(value("sage-200")).toBe("#b1d2b2");
  });

  it("is exposed to Tailwind (sage, ink, danger, card/media/btn/chip radii, card shadow, btn height)", () => {
    for (const needle of ["sage:", "ink:", "danger:", "card: \"var(--card-radius)\"", "media: \"var(--media-radius)\"", "btn: \"var(--btn-radius)\"", "chip: \"var(--chip-radius)\"", "var(--card-shadow)", "var(--btn-height)"]) {
      expect(tw, needle).toContain(needle);
    }
  });

  it("white text on the primary button colour meets 4.5:1", () => {
    const lum = (hex: string) => {
      const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((x) => (x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
      return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    };
    const ratio = 1.05 / (lum(value("sage-600")!) + 0.05);
    expect(ratio).toBeGreaterThanOrEqual(4.5);
  });
});

describe("practice page uses tokens, not one-off colours", () => {
  const files = [
    ...readdirSync(resolve(root, "src/components/ds")).filter((f) => f.endsWith(".tsx") && !f.endsWith(".test.tsx")).map((f) => `src/components/ds/${f}`),
    "src/components/LearningCard.tsx",
    "src/components/VideoExampleCard.tsx",
    "src/components/CompareDialog.tsx",
    "src/pages/Learn.tsx",
    "src/hooks/useLandmarkTracker.ts",
  ];
  it("has no hex colour literals", () => {
    for (const f of files) expect(read(f).match(/#[0-9a-fA-F]{3,8}\b/g) ?? [], f).toEqual([]);
  });
  it("has no ad-hoc green/gray utility colours for brand surfaces (sage / ink tokens only)", () => {
    for (const f of files) expect(read(f).match(/\b(?:bg|text|border)-(?:green|emerald|teal|lime|amber|red|gray|slate|zinc)-\d{2,3}\b/g) ?? [], f).toEqual([]);
  });
  it("both media wells share one aspect ratio (no per-card override)", () => {
    for (const f of ["src/components/LearningCard.tsx", "src/components/VideoExampleCard.tsx"]) expect(read(f)).not.toMatch(/aspect=/);
  });
  it("uses the primitives: SurfaceCard + MediaWell + PillButton for the two video panels, AppDialog for modals", () => {
    for (const f of ["src/components/LearningCard.tsx", "src/components/VideoExampleCard.tsx"]) {
      const src = read(f);
      for (const p of ["SurfaceCard", "MediaWell", "PillButton"]) expect(src, `${f} ${p}`).toContain(`<${p}`);
      expect(src).not.toMatch(/from "@\/components\/ui\/(button|card)"/);
    }
    expect(read("src/pages/Learn.tsx")).toContain("<AppDialog");
    expect(read("src/components/CompareDialog.tsx")).toContain("<AppDialog");
    expect(read("src/pages/Learn.tsx")).not.toContain("AlertDialog");
  });
});
