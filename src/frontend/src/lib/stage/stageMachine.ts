/**
 * State machine for the Learn -> Practice stage (per word):
 *
 *   learn -> practice -> countdown -> recording -> grading -> result -> next word | try again | watch again
 *
 * Pure reducer (no React, no DOM) so the flow is unit-testable. Invalid events are ignored (state unchanged).
 */
export type StageMode = "learn" | "practice" | "countdown" | "recording" | "grading" | "result";
export type ResultKind = "correct" | "close" | "confused" | "incorrect" | "not_detected" | "error";
export type PlaybackRate = 1 | 0.5 | 0.25;
export const PLAYBACK_RATES: readonly PlaybackRate[] = [1, 0.5, 0.25];

export interface StageState {
  mode: StageMode;
  word: string;
  /** Example video loops (on by default; forced on by "Watch again"). */
  loop: boolean;
  rate: PlaybackRate;
  /** Which of the two example signers is showing. */
  example: 1 | 2;
  /** Learn mode only: show the camera picture-in-picture ("Show me"). */
  showMe: boolean;
  /** Set while mode === "result". */
  result: ResultKind | null;
}

export type StageEvent =
  | { type: "SET_WORD"; word: string } // header arrows / deck change / first render: back to Learn on that word
  | { type: "READY" } // "I'm ready to practice"
  | { type: "RECORD" }
  | { type: "CANCEL_COUNTDOWN" }
  | { type: "COUNTDOWN_DONE" }
  | { type: "STOP" }
  | { type: "UPLOAD" } // an uploaded clip goes straight to grading
  | { type: "GRADED"; result: ResultKind }
  | { type: "TRY_AGAIN" }
  | { type: "WATCH_AGAIN" }
  | { type: "NEXT"; word: string }
  | { type: "TOGGLE_LOOP" }
  | { type: "CYCLE_RATE" }
  | { type: "SET_EXAMPLE"; example: 1 | 2 }
  | { type: "TOGGLE_SHOW_ME" };

export const initialStageState = (word: string): StageState => ({
  mode: "learn",
  word,
  loop: true,
  rate: 1,
  example: 1,
  showMe: false,
  result: null,
});

export function nextRate(rate: PlaybackRate): PlaybackRate {
  return PLAYBACK_RATES[(PLAYBACK_RATES.indexOf(rate) + 1) % PLAYBACK_RATES.length];
}

/** Fresh Learn state for a (new) word: loop on, normal speed, first example, camera preview off. */
const learnFor = (word: string): StageState => initialStageState(word);

export function stageReducer(s: StageState, e: StageEvent): StageState {
  switch (e.type) {
    case "SET_WORD":
      return e.word === s.word && s.mode === "learn" ? s : learnFor(e.word);
    case "NEXT":
      return s.mode === "result" ? learnFor(e.word) : s;
    case "READY":
      return s.mode === "learn" ? { ...s, mode: "practice", showMe: false } : s;
    case "RECORD":
      return s.mode === "practice" || s.mode === "result" ? { ...s, mode: "countdown", result: null } : s;
    case "CANCEL_COUNTDOWN":
      return s.mode === "countdown" ? { ...s, mode: "practice" } : s;
    case "COUNTDOWN_DONE":
      return s.mode === "countdown" ? { ...s, mode: "recording" } : s;
    case "STOP":
      return s.mode === "recording" ? { ...s, mode: "grading" } : s;
    case "UPLOAD":
      return s.mode === "practice" || s.mode === "result" ? { ...s, mode: "grading", result: null } : s;
    case "GRADED":
      return s.mode === "grading" ? { ...s, mode: "result", result: e.result } : s;
    case "TRY_AGAIN":
      return s.mode === "result" ? { ...s, mode: "practice", result: null } : s;
    case "WATCH_AGAIN":
      // Back to Learn on the SAME word, with the loop forced on.
      return s.mode === "practice" || s.mode === "result" ? { ...s, mode: "learn", loop: true, result: null, showMe: false } : s;
    case "TOGGLE_LOOP":
      return { ...s, loop: !s.loop };
    case "CYCLE_RATE":
      return { ...s, rate: nextRate(s.rate) };
    case "SET_EXAMPLE":
      return { ...s, example: e.example };
    case "TOGGLE_SHOW_ME":
      return s.mode === "learn" ? { ...s, showMe: !s.showMe } : s;
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

/** Camera should be running in these situations (Learn only when "Show me" is on). */
export function cameraWanted(s: Pick<StageState, "mode" | "showMe">): boolean {
  return s.mode === "practice" || s.mode === "countdown" || s.mode === "recording" || (s.mode === "learn" && s.showMe);
}
