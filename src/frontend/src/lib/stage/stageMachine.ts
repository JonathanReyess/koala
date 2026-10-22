/**
 * State machine for the stage (per word). The same flow runs in both views (Practice and Quiz; the view only
 * decides what is on screen and lives in the page, not here):
 *
 *   ready -> countdown -> recording -> grading -> result -> next word | try again
 *
 * Pure reducer (no React, no DOM) so the flow is unit-testable. Invalid events are ignored (state unchanged).
 */
export type StageMode = "ready" | "countdown" | "recording" | "grading" | "result";
/** Practice: example video + camera side by side. Quiz: only the word (in the page header) and your camera. */
export type StageView = "practice" | "quiz";
export type ResultKind = "correct" | "close" | "confused" | "incorrect" | "not_detected" | "error";
export type PlaybackRate = 1 | 0.5 | 0.25;
export const PLAYBACK_RATES: readonly PlaybackRate[] = [1, 0.5, 0.25];

export interface StageState {
  mode: StageMode;
  word: string;
  /** Example video loops (on by default). */
  loop: boolean;
  rate: PlaybackRate;
  /** Which of the two example signers is showing. */
  example: 1 | 2;
  /** Set while mode === "result". */
  result: ResultKind | null;
}

export type StageEvent =
  | { type: "SET_WORD"; word: string } // header arrows / deck change / first render: fresh state on that word
  | { type: "RECORD" }
  | { type: "CANCEL_COUNTDOWN" }
  | { type: "COUNTDOWN_DONE" }
  | { type: "STOP" }
  | { type: "UPLOAD" } // an uploaded clip goes straight to grading
  | { type: "GRADED"; result: ResultKind }
  | { type: "TRY_AGAIN" }
  | { type: "NEXT"; word: string }
  | { type: "TOGGLE_LOOP" }
  | { type: "CYCLE_RATE" }
  | { type: "SET_EXAMPLE"; example: 1 | 2 };

export const initialStageState = (word: string): StageState => ({
  mode: "ready",
  word,
  loop: true,
  rate: 1,
  example: 1,
  result: null,
});

export function nextRate(rate: PlaybackRate): PlaybackRate {
  return PLAYBACK_RATES[(PLAYBACK_RATES.indexOf(rate) + 1) % PLAYBACK_RATES.length];
}

/** Fresh state for a (new) word: loop on, normal speed, first example. */
const freshFor = (word: string): StageState => initialStageState(word);

export function stageReducer(s: StageState, e: StageEvent): StageState {
  switch (e.type) {
    case "SET_WORD":
      return e.word === s.word && s.mode === "ready" ? s : freshFor(e.word);
    case "NEXT":
      return s.mode === "result" ? freshFor(e.word) : s;
    case "RECORD":
      return s.mode === "ready" || s.mode === "result" ? { ...s, mode: "countdown", result: null } : s;
    case "CANCEL_COUNTDOWN":
      return s.mode === "countdown" ? { ...s, mode: "ready" } : s;
    case "COUNTDOWN_DONE":
      return s.mode === "countdown" ? { ...s, mode: "recording" } : s;
    case "STOP":
      return s.mode === "recording" ? { ...s, mode: "grading" } : s;
    case "UPLOAD":
      return s.mode === "ready" || s.mode === "result" ? { ...s, mode: "grading", result: null } : s;
    case "GRADED":
      return s.mode === "grading" ? { ...s, mode: "result", result: e.result } : s;
    case "TRY_AGAIN":
      return s.mode === "result" ? { ...s, mode: "ready", result: null } : s;
    case "TOGGLE_LOOP":
      return { ...s, loop: !s.loop };
    case "CYCLE_RATE":
      return { ...s, rate: nextRate(s.rate) };
    case "SET_EXAMPLE":
      return { ...s, example: e.example };
  }
}

/** The one primary action of the result panel, by outcome. */
export interface PrimaryAction {
  label: string;
  event: "NEXT" | "TRY_AGAIN";
}

export function primaryActionFor(result: ResultKind): PrimaryAction {
  return result === "correct" ? { label: "Next sign", event: "NEXT" } : { label: "Try again", event: "TRY_AGAIN" };
}

/** Camera should be running while getting ready, counting down and recording. */
export function cameraWanted(s: Pick<StageState, "mode">): boolean {
  return s.mode === "ready" || s.mode === "countdown" || s.mode === "recording";
}
