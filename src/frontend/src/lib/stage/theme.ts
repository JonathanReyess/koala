/**
 * Layout/visual tokens for the Learn -> Practice stage, in one place.
 * Applied as CSS custom properties on the stage root (`stageCssVars`) and read via Tailwind arbitrary values
 * (e.g. `rounded-[var(--tile-radius)]`) or inline styles, so the whole stage can be re-skinned from here.
 */
export const STAGE_THEME = {
  /** Tile (video card) corner radius. */
  tileRadius: "24px",
  /** Clean white cards: a hairline border and a soft shadow instead of heavy outlines. */
  tileBorderWidth: "1px",
  tileBorderColor: "#e5e7eb",
  tileShadow: "0 10px 30px -12px rgba(15, 23, 42, 0.25)",
  /** Camera tile ring while recording (the label + timer carry the meaning too; colour is only reinforcement). */
  tileBorderRecording: "#b3261e",
  /** Primary call-to-action colour (white text on it is ~7.5:1) and its hover state. */
  ctaColor: "#2f5f52",
  ctaHoverColor: "#244a40",
  /** Minimum tap target (px) for every control; large controls (Play, Slow, Record, primary CTAs) use tapLarge. */
  tapMin: 48,
  tapLarge: 60,
  /** Column splits as CSS grid templates (md and up; phones stack). */
  splitLearn: "minmax(0,72fr) minmax(200px,28fr)", // example ~72% of the stage; camera picture-in-picture column ~28% (>= 22%, min 200px)
  splitPractice: "1fr 1fr", // side by side, 50/50
  /** Learn-mode camera picture-in-picture: >= 22% of stage width, never below 200px, inset 16-24px. */
  pipMinWidthPx: 200,
  pipWidthPct: 22,
  pipInset: "20px",
  /**
   * Vertical space the rest of the page uses (header, word title, tile bars, action bar). Video areas are capped
   * at (viewport height - this) so a whole stage fits a ~800px laptop screen without scrolling; when the cap binds
   * the video is cropped from the TOP only (object-position centre-bottom keeps torso and hands).
   */
  stageChrome: "424px",
  /** Never let the cap shrink a video area below this. */
  videoMinHeight: "240px",
  /** Vertical crop anchor for videos so heads, torsos AND hands stay in view when a tile crops. */
  videoObjectPosition: "center bottom",
} as const;

export type StageTheme = typeof STAGE_THEME;

export function stageCssVars(t: StageTheme = STAGE_THEME): Record<string, string> {
  return {
    "--tile-radius": t.tileRadius,
    "--tile-border": t.tileBorderWidth,
    "--tile-border-color": t.tileBorderColor,
    "--tile-shadow": t.tileShadow,
    "--tile-border-recording": t.tileBorderRecording,
    "--cta": t.ctaColor,
    "--cta-hover": t.ctaHoverColor,
    "--tap-min": `${t.tapMin}px`,
    "--tap-large": `${t.tapLarge}px`,
    "--split-learn": t.splitLearn,
    "--split-practice": t.splitPractice,
    "--pip-min-width": `${t.pipMinWidthPx}px`,
    "--pip-width": `${t.pipWidthPct}%`,
    "--pip-inset": t.pipInset,
    "--video-position": t.videoObjectPosition,
    "--video-max-h": `max(${t.videoMinHeight}, calc(100dvh - ${t.stageChrome}))`,
  };
}
