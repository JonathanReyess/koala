/**
 * Layout tokens for the Learn -> Practice stage. Colours, radii, shadows and button looks come from the
 * design system tokens in index.css (--sage-*, --ink, --card-*, --media-*, --btn-*) and the primitives in
 * components/ds; this file only holds stage layout. Applied as CSS custom properties on the stage root.
 */
export const STAGE_THEME = {
  /** Column splits as CSS grid templates (md and up; phones stack). */
  splitLearn: "minmax(0,72fr) minmax(200px,28fr)", // example ~72% of the stage; camera picture-in-picture column ~28% (>= 22%, min 200px)
  splitPractice: "1fr 1fr", // side by side, 50/50
  /** Learn-mode camera picture-in-picture: >= 22% of stage width, never below 200px, inset 16-24px. */
  pipMinWidthPx: 200,
  pipWidthPct: 22,
  pipInset: "20px",
  /**
   * Vertical space the rest of the page uses (header, word title, card chrome, action bar). Media wells are capped
   * at (viewport height - this) so a whole stage fits a ~800px laptop screen without scrolling; when the cap binds
   * the video is cropped from the TOP only (object-position centre-bottom keeps torso and hands).
   */
  stageChrome: "500px",
  /** Never let the cap shrink a media well below this. */
  videoMinHeight: "240px",
  /** Minimum tap targets in px (ds PillButton sizes: touch = 48, lg = 60). */
  tapMin: 48,
  tapLarge: 60,
} as const;

export type StageTheme = typeof STAGE_THEME;

export function stageCssVars(t: StageTheme = STAGE_THEME): Record<string, string> {
  return {
    "--split-learn": t.splitLearn,
    "--split-practice": t.splitPractice,
    "--pip-min-width": `${t.pipMinWidthPx}px`,
    "--pip-width": `${t.pipWidthPct}%`,
    "--pip-inset": t.pipInset,
    "--video-max-h": `max(${t.videoMinHeight}, calc(100dvh - ${t.stageChrome}))`,
  };
}
