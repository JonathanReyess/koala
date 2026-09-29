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
  run([{ type: "READY" }, { type: "RECORD" }, { type: "COUNTDOWN_DONE" }, { type: "STOP" }, { type: "GRADED", result }], from);

describe("stage state machine", () => {
  it("starts in learn with loop on, normal speed, example 1, camera preview off", () => {
    expect(initialStageState("hi")).toEqual({ mode: "learn", word: "hi", loop: true, rate: 1, example: 1, showMe: false, result: null });
  });

  it("walks learn -> practice -> countdown -> recording -> grading -> result", () => {
    let s = initialStageState("hi");
    const modes: string[] = [s.mode];
    for (const e of [{ type: "READY" }, { type: "RECORD" }, { type: "COUNTDOWN_DONE" }, { type: "STOP" }, { type: "GRADED", result: "correct" }] as StageEvent[]) {
      s = stageReducer(s, e);
      modes.push(s.mode);
    }
    expect(modes).toEqual(["learn", "practice", "countdown", "recording", "grading", "result"]);
    expect(s.result).toBe("correct");
  });

  it.each(["correct", "close", "confused", "incorrect", "not_detected", "error"] as ResultKind[])("result %s is reachable", (r) => {
    expect(toResult(r)).toMatchObject({ mode: "result", result: r });
  });

  it("ignores events that aren't valid in the current mode", () => {
    const learn = initialStageState("hi");
    for (const e of [{ type: "RECORD" }, { type: "COUNTDOWN_DONE" }, { type: "STOP" }, { type: "GRADED", result: "correct" }, { type: "TRY_AGAIN" }, { type: "WATCH_AGAIN" }, { type: "UPLOAD" }] as StageEvent[]) {
      expect(stageReducer(learn, e)).toBe(learn);
    }
    const practice = run([{ type: "READY" }]);
    expect(stageReducer(practice, { type: "READY" })).toBe(practice);
    expect(stageReducer(practice, { type: "STOP" })).toBe(practice);
    const recording = run([{ type: "READY" }, { type: "RECORD" }, { type: "COUNTDOWN_DONE" }]);
    expect(stageReducer(recording, { type: "RECORD" })).toBe(recording);
    expect(stageReducer(recording, { type: "WATCH_AGAIN" })).toBe(recording); // can't leave mid-recording
    expect(stageReducer(recording, { type: "GRADED", result: "correct" })).toBe(recording); // only from grading
  });

  it("can cancel a countdown back to practice", () => {
    expect(run([{ type: "READY" }, { type: "RECORD" }, { type: "CANCEL_COUNTDOWN" }]).mode).toBe("practice");
  });

  it("an uploaded clip goes practice -> grading -> result", () => {
    expect(run([{ type: "READY" }, { type: "UPLOAD" }]).mode).toBe("grading");
  });

  it("try again returns to practice (and can record again straight from a result)", () => {
    expect(run([{ type: "TRY_AGAIN" }], toResult("incorrect"))).toMatchObject({ mode: "practice", result: null });
    expect(run([{ type: "RECORD" }], toResult("close"))).toMatchObject({ mode: "countdown", result: null });
  });

  it("Watch again returns to Learn on the SAME word with loop forced on", () => {
    let s = run([{ type: "TOGGLE_LOOP" }, { type: "CYCLE_RATE" }, { type: "SET_EXAMPLE", example: 2 }]);
    expect(s).toMatchObject({ loop: false, rate: 0.5, example: 2 });
    s = toResult("incorrect", s);
    const back = stageReducer(s, { type: "WATCH_AGAIN" });
    expect(back).toMatchObject({ mode: "learn", word: "hi", loop: true, result: null, showMe: false });
    // also from practice
    expect(stageReducer(run([{ type: "TOGGLE_LOOP" }, { type: "READY" }]), { type: "WATCH_AGAIN" })).toMatchObject({ mode: "learn", word: "hi", loop: true });
  });

  it("Next (after a result) and SET_WORD start the new word in Learn with defaults", () => {
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

  it("Show me only toggles in Learn; the camera runs in practice/countdown/recording and in Learn only with Show me", () => {
    const learn = initialStageState("hi");
    expect(cameraWanted(learn)).toBe(false);
    const withMe = stageReducer(learn, { type: "TOGGLE_SHOW_ME" });
    expect(cameraWanted(withMe)).toBe(true);
    const practice = stageReducer(withMe, { type: "READY" });
    expect(practice.showMe).toBe(false);
    expect(stageReducer(practice, { type: "TOGGLE_SHOW_ME" })).toBe(practice);
    expect(["practice", "countdown", "recording"].every((m) => cameraWanted({ mode: m as never, showMe: false }))).toBe(true);
    expect(["grading", "result"].some((m) => cameraWanted({ mode: m as never, showMe: false }))).toBe(false);
  });

  it("primary action by result: correct -> Next sign; everything else -> Try again", () => {
    expect(primaryActionFor("correct")).toEqual({ label: "Next sign", event: "NEXT" });
    for (const r of ["close", "confused", "incorrect", "not_detected", "error"] as ResultKind[]) {
      expect(primaryActionFor(r)).toEqual({ label: "Try again", event: "TRY_AGAIN" });
    }
  });
});

describe("theme tokens meet the design rules", () => {
  it("tap targets, picture-in-picture and splits", () => {
    expect(STAGE_THEME.tapMin).toBeGreaterThanOrEqual(48);
    expect(STAGE_THEME.tapLarge).toBeGreaterThanOrEqual(56);
    expect(STAGE_THEME.tapLarge).toBeLessThanOrEqual(64);
    expect(STAGE_THEME.pipWidthPct).toBeGreaterThanOrEqual(22);
    expect(STAGE_THEME.pipMinWidthPx).toBeGreaterThanOrEqual(200);
    const inset = parseInt(STAGE_THEME.pipInset, 10);
    expect(inset).toBeGreaterThanOrEqual(16);
    expect(inset).toBeLessThanOrEqual(24);
    expect(STAGE_THEME.splitPractice).toBe("1fr 1fr");
    expect(STAGE_THEME.splitLearn).toMatch(/72fr/);
    expect(parseInt(STAGE_THEME.tileBorderWidth, 10)).toBeGreaterThanOrEqual(3);
    const vars = stageCssVars();
    expect(vars["--tap-min"]).toBe("48px");
    expect(vars["--split-practice"]).toBe("1fr 1fr");
    expect(vars["--video-position"]).toBe("center bottom");
  });
});
