/**
 * Layout tokens for the Learn -> Practice stage. Colours, radii, shadows and button looks come from the
 * design system tokens in index.css (--sage-*, --ink, --card-*, --media-*, --btn-*) and the primitives in
 * components/ds; this file only holds stage layout. Applied as CSS custom properties on the stage root.
 */
export const STAGE_THEME = {
  /** Column splits as CSS grid templates (md and up; phones stack). */
  // Columns are sized to the cards (whose width follows the fitted video height) and the pair is centred, so a
  // shorter screen gives smaller cards that stay together instead of drifting apart.
  splitLearn: "minmax(0,var(--card-max-w)) minmax(200px,280px)", // example card + camera picture-in-picture column (>= 200px)
  splitPractice: "repeat(2,minmax(0,var(--card-max-w)))", // side by side, equal halves
  /** Learn-mode camera picture-in-picture: >= 22% of stage width, never below 200px, inset 16-24px. */
  pipMinWidthPx: 200,
  pipWidthPct: 22,
  pipInset: "20px",
  /**
   * FALLBACK estimate of the vertical space everything except the media wells takes, used only until
   * hooks/useFitToViewport measures the real value in the browser and sets --video-max-h exactly (no scrolling on
   * md+ screens). When the cap binds, the CARD gets narrower and the video keeps its 16:10 shape (no cropping).
   */
  stageChrome: "480px", // learn / practice / countdown / recording / grading (one button below the cards)
  stageChromeResult: "528px", // result (the result panel is taller than one button)
  /** Never let the cap shrink a media well below this. */
  videoMinHeight: "240px",
  /** Minimum tap targets in px (ds PillButton sizes: touch = 48, lg = 60). */
  tapMin: 48,
  tapLarge: 60,
} as const;

export type StageTheme = typeof STAGE_THEME;

export function stageCssVars(t: StageTheme = STAGE_THEME, mode?: string): Record<string, string> {
  const chrome = mode === "result" ? t.stageChromeResult : t.stageChrome;
  return {
    "--split-learn": t.splitLearn,
    "--split-practice": t.splitPractice,
    "--pip-min-width": `${t.pipMinWidthPx}px`,
    "--pip-width": `${t.pipWidthPct}%`,
    "--pip-inset": t.pipInset,
    "--video-max-h": `max(${t.videoMinHeight}, calc(100dvh - ${chrome}))`,
    // widest a card may be so its 16:10 well is no taller than --video-max-h (+ 2 x 16px inset + 2px border)
    "--card-max-w": "calc(var(--video-max-h) * 1.6 + 34px)",
  };
}
