import { describe, it, expect } from "vitest";
import {
  PLAYBACK_RATES,
  cameraWanted,
  initialStageState,
  nextRate,
  primaryActionFor,
  stageReducer,
  type ResultKind,
  type StageEvent,
  type StageState,
} from "./stageMachine";
import { STAGE_THEME, stageCssVars } from "./theme";

const run = (events: StageEvent[], from: StageState = initialStageState("hi")) => events.reduce(stageReducer, from);
const toResult = (result: ResultKind, from = initialStageState("hi")) =>
  run([{ type: "RECORD" }, { type: "COUNTDOWN_DONE" }, { type: "STOP" }, { type: "GRADED", result }], from);

describe("stage state machine", () => {
  it("starts ready with loop on, normal speed, example 1", () => {
    expect(initialStageState("hi")).toEqual({ mode: "ready", word: "hi", loop: true, rate: 1, example: 1, result: null });
  });

  it("walks ready -> countdown -> recording -> grading -> result", () => {
    let s = initialStageState("hi");
    const modes: string[] = [s.mode];
    for (const e of [{ type: "RECORD" }, { type: "COUNTDOWN_DONE" }, { type: "STOP" }, { type: "GRADED", result: "correct" }] as StageEvent[]) {
      s = stageReducer(s, e);
      modes.push(s.mode);
    }
    expect(modes).toEqual(["ready", "countdown", "recording", "grading", "result"]);
    expect(s.result).toBe("correct");
  });

  it.each(["correct", "close", "confused", "incorrect", "not_detected", "error"] as ResultKind[])("result %s is reachable", (r) => {
    expect(toResult(r)).toMatchObject({ mode: "result", result: r });
  });

  it("ignores events that aren't valid in the current mode", () => {
    const ready = initialStageState("hi");
    for (const e of [{ type: "COUNTDOWN_DONE" }, { type: "STOP" }, { type: "GRADED", result: "correct" }, { type: "TRY_AGAIN" }] as StageEvent[]) {
      expect(stageReducer(ready, e)).toBe(ready);
    }
    const recording = run([{ type: "RECORD" }, { type: "COUNTDOWN_DONE" }]);
    expect(stageReducer(recording, { type: "RECORD" })).toBe(recording);
    expect(stageReducer(recording, { type: "UPLOAD" })).toBe(recording);
    expect(stageReducer(recording, { type: "TRY_AGAIN" })).toBe(recording); // can't leave mid-recording
    expect(stageReducer(recording, { type: "GRADED", result: "correct" })).toBe(recording); // only from grading
  });

  it("can cancel a countdown back to ready", () => {
    expect(run([{ type: "RECORD" }, { type: "CANCEL_COUNTDOWN" }]).mode).toBe("ready");
  });

  it("an uploaded clip goes ready -> grading -> result", () => {
    expect(run([{ type: "UPLOAD" }]).mode).toBe("grading");
  });

  it("try again returns to ready (and can record again straight from a result)", () => {
    expect(run([{ type: "TRY_AGAIN" }], toResult("incorrect"))).toMatchObject({ mode: "ready", result: null });
    expect(run([{ type: "RECORD" }], toResult("close"))).toMatchObject({ mode: "countdown", result: null });
  });

  it("Next (after a result) and SET_WORD start the new word fresh with defaults", () => {
    const next = stageReducer(toResult("correct"), { type: "NEXT", word: "eat" });
    expect(next).toEqual(initialStageState("eat"));
    expect(stageReducer(toResult("correct"), { type: "SET_WORD", word: "no" })).toEqual(initialStageState("no"));
    const same = initialStageState("hi");
    expect(stageReducer(same, { type: "SET_WORD", word: "hi" })).toBe(same);
    expect(stageReducer(initialStageState("hi"), { type: "NEXT", word: "eat" })).toMatchObject({ word: "hi" }); // NEXT only from a result
  });

  it("Slow cycles 1x -> 0.5x -> 0.25x -> 1x", () => {
    expect(PLAYBACK_RATES).toEqual([1, 0.5, 0.25]);
    expect([1, 0.5, 0.25].map((r) => nextRate(r as 1 | 0.5 | 0.25))).toEqual([0.5, 0.25, 1]);
    expect(run(Array(3).fill({ type: "CYCLE_RATE" })).rate).toBe(1);
  });

  it("the camera runs while ready, counting down and recording, and is released for grading and results", () => {
    expect(["ready", "countdown", "recording"].every((m) => cameraWanted({ mode: m as never }))).toBe(true);
    expect(["grading", "result"].some((m) => cameraWanted({ mode: m as never }))).toBe(false);
  });

  it("primary action by result: correct -> Next sign; everything else -> Try again", () => {
    expect(primaryActionFor("correct")).toEqual({ label: "Next sign", event: "NEXT" });
    for (const r of ["close", "confused", "incorrect", "not_detected", "error"] as ResultKind[]) {
      expect(primaryActionFor(r)).toEqual({ label: "Try again", event: "TRY_AGAIN" });
    }
  });
});

describe("stage layout tokens", () => {
  it("splits and tap targets", () => {
    expect(STAGE_THEME.tapMin).toBeGreaterThanOrEqual(48);
    expect(STAGE_THEME.tapLarge).toBeGreaterThanOrEqual(56);
    expect(STAGE_THEME.tapLarge).toBeLessThanOrEqual(64);
    expect(STAGE_THEME.splitPractice).toBe("repeat(2,minmax(0,var(--card-max-w)))"); // two equal columns
    expect(STAGE_THEME.splitQuiz).toContain("var(--card-max-w)"); // the lone camera card
    const vars = stageCssVars();
    expect(vars["--split-practice"]).toContain("repeat(2,");
    expect(vars["--video-max-h"]).toContain("100dvh");
    // Colours, radii, shadows and button looks are NOT defined here: they come from the design-system tokens.
    expect(Object.keys(vars).some((k) => /color|shadow|radius|border/.test(k))).toBe(false);
  });
});
