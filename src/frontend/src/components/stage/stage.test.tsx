// @vitest-environment jsdom
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import type { ResultKind } from "@/lib/stage/stageMachine";

// --- A controllable stand-in for the camera/model/grading session -------------------------------------------------
const session = vi.hoisted(() => ({
  nextResult: "incorrect" as string,
  /** When true, beginRecording stops at the countdown (to inspect the 3-2-1 overlay). */
  holdCountdown: false,
  beginRecording: vi.fn(),
  stopRecording: vi.fn(),
  cancelCountdown: vi.fn(),
}));

vi.mock("@/hooks/usePracticeSession", () => ({
  usePracticeSession: ({ dispatch }: { dispatch: (e: unknown) => void }) => ({
    videoRef: { current: null },
    canvasRef: { current: null },
    modelStatus: "ready",
    cameraOn: true,
    cameraError: false,
    tracker: { hint: null, framingOk: true, fps: 0 },
    showTracking: false,
    setShowTracking: vi.fn(),
    countdown: 3,
    elapsedMs: 0,
    grade: null,
    feedbackText: `feedback for ${session.nextResult}`,
    recordedUrl: null,
    beginRecording: () => {
      session.beginRecording();
      dispatch({ type: "RECORD" });
      if (!session.holdCountdown) dispatch({ type: "COUNTDOWN_DONE" }); // skip the real 3-2-1
    },
    cancelCountdown: session.cancelCountdown,
    stopRecording: () => {
      session.stopRecording();
      dispatch({ type: "STOP" });
      dispatch({ type: "GRADED", result: session.nextResult });
    },
    uploadFile: vi.fn(),
    debug: false,
    debugInfo: null,
    loggedAttempts: 0,
    downloadAttempts: vi.fn(),
    clearAttempts: vi.fn(),
  }),
}));

import { Stage } from "./Stage";
import { ResultPanel } from "./ResultPanel";

function memoryStorage(): Storage {
  const m = new Map<string, string>();
  return {
    get length() {
      return m.size;
    },
    clear: () => m.clear(),
    getItem: (k) => (m.has(k) ? m.get(k)! : null),
    key: (i) => [...m.keys()][i] ?? null,
    removeItem: (k) => void m.delete(k),
    setItem: (k, v) => void m.set(k, String(v)),
  };
}

beforeAll(() => {
  // Node 25 ships an experimental global localStorage that shadows jsdom's; use a plain in-memory one.
  Object.defineProperty(globalThis, "localStorage", { value: memoryStorage(), configurable: true });
  Object.defineProperty(window, "localStorage", { value: globalThis.localStorage, configurable: true });
  // jsdom lacks these; Radix switches and media elements need them.
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as never;
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(() => Promise.resolve());
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
});

beforeEach(() => {
  session.nextResult = "incorrect";
  session.holdCountdown = false;
  session.beginRecording.mockClear();
  session.stopRecording.mockClear();
  session.cancelCountdown.mockClear();
  localStorage.clear();
});
afterEach(cleanup);

const exampleVideo = () => screen.getByTestId("example-video") as HTMLVideoElement;
const cameraVideo = () => screen.getByTestId("camera-video") as HTMLVideoElement;
const openSettings = () => fireEvent.click(screen.getByTestId("settings-button"));
const mode = () => screen.getByTestId("stage").getAttribute("data-mode");
const goToResult = (result: ResultKind) => {
  session.nextResult = result;
  fireEvent.click(screen.getByTestId("record"));
  fireEvent.click(screen.getByTestId("stop"));
};

describe("Practice view", () => {
  it("shows the example with custom controls next to the camera, and no native video controls", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    expect(mode()).toBe("ready");
    expect(exampleVideo().hasAttribute("controls")).toBe(false);
    expect(exampleVideo().getAttribute("src")).toBe("/videos/hi_example1.mp4");
    for (const id of ["play-pause", "slow", "loop"]) expect(screen.getByTestId(id)).toBeTruthy();
    expect(screen.getByTestId("camera-tile")).toBeTruthy();
    expect(screen.getByTestId("record").textContent).toContain("Record"); // no separate learn step: you can record right away
    expect(screen.queryByTestId("ready")).toBeNull();
    expect(screen.queryByTestId("show-me")).toBeNull();
    expect(exampleVideo().loop).toBe(true); // loop on by default
  });

  it("toggles between example 1 and example 2", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Example 2" }));
    expect(exampleVideo().getAttribute("src")).toBe("/videos/hi_example2.mp4");
  });

  it("Slow sets the video's playbackRate: 1x -> 0.5x -> 0.25x -> 1x", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    expect(exampleVideo().playbackRate).toBe(1);
    fireEvent.click(screen.getByTestId("slow"));
    expect(exampleVideo().playbackRate).toBe(0.5);
    expect(screen.getByTestId("slow").textContent).toContain("0.5×");
    fireEvent.click(screen.getByTestId("slow"));
    expect(exampleVideo().playbackRate).toBe(0.25);
    fireEvent.click(screen.getByTestId("slow"));
    expect(exampleVideo().playbackRate).toBe(1);
  });

  it("Loop toggles the video's loop attribute; there is no separate Replay button", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("loop"));
    expect(exampleVideo().loop).toBe(false);
    expect(screen.queryByTestId("replay")).toBeNull();
  });

  it("no auto-advance: stays ready until Record is pressed", () => {
    vi.useFakeTimers();
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    vi.advanceTimersByTime(60_000);
    expect(mode()).toBe("ready");
    vi.useRealTimers();
  });
});

describe("Record flow", () => {
  it("Ready -> Recording (Stop) -> Result", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    expect(mode()).toBe("ready");
    expect(screen.getByTestId("record").textContent).toContain("Record");
    fireEvent.click(screen.getByTestId("record"));
    expect(session.beginRecording).toHaveBeenCalled();
    expect(mode()).toBe("recording");
    expect(screen.getByTestId("stop").textContent).toContain("Stop");
    fireEvent.click(screen.getByTestId("stop"));
    expect(mode()).toBe("result");
    expect(screen.getByTestId("result-panel").getAttribute("data-result")).toBe("incorrect");
  });

  it("camera tile mirror: default mirrored; 'Match the example' unmirrors ONLY the camera tile, and persists", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    expect(cameraVideo().className).toContain("scale-x-[-1]");
    expect(screen.getByTestId("camera-canvas").className).toContain("scale-x-[-1]");
    const exampleClassBefore = exampleVideo().className;
    openSettings();
    fireEvent.click(screen.getByRole("switch", { name: "Match the example" }));
    expect(cameraVideo().className).not.toContain("scale-x-[-1]");
    expect(screen.getByTestId("camera-canvas").className).not.toContain("scale-x-[-1]");
    expect(exampleVideo().className).toBe(exampleClassBefore); // example untouched
    expect(exampleVideo().className).not.toContain("scale-x-[-1]");
    expect(localStorage.getItem("koala.mirrorCamera")).toBe("0");
    cleanup();
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />); // remembered
    expect(cameraVideo().className).not.toContain("scale-x-[-1]");
  });

  it("the tracking/mirror options are behind a gear in the media well's TOP-RIGHT, not loose chips over the video", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    const tile = screen.getByTestId("camera-tile");
    const overlay = cameraVideo().parentElement!.querySelector("div.absolute.inset-0")!;
    const [topRow, bottomRow] = Array.from(overlay.children) as HTMLElement[];
    expect(topRow.contains(within(tile).getByTestId("settings-button"))).toBe(true);
    expect(within(tile).queryByRole("switch")).toBeNull(); // nothing taking up video space until the gear is opened
    openSettings();
    expect(screen.getByRole("switch", { name: "Show tracking" })).toBeTruthy();
    expect(screen.getByRole("switch", { name: "Match the example" })).toBeTruthy();
    expect(screen.getByTestId("settings-popover").closest("section")).toBeNull(); // floats above the card, not inside the video
  });

  it("the status chip sits in the TOP row and has no star icon", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    const overlay = cameraVideo().parentElement!.querySelector("div.absolute.inset-0")!;
    const [topRow, bottomRow] = Array.from(overlay.children) as HTMLElement[];
    const chip = within(topRow).getByRole("status");
    expect(chip.textContent).toContain("You're all set. Press Record.");
    expect(chip.querySelector("svg")).toBeNull();
    // nothing sits in the bottom row until recording starts
    expect(bottomRow?.textContent ?? "").toBe("");
  });

  it("recording state is conveyed with text, not colour alone (Recording label + timer)", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("record"));
    const tile = screen.getByTestId("camera-tile");
    expect(tile.textContent).toContain("Recording");
    expect(tile.textContent).toMatch(/\d:\d\d/);
  });
});

describe("Result panel", () => {
  const primaryLabel = (r: ResultKind) => {
    render(<ResultPanel result={r} message="msg" onNext={vi.fn()} onTryAgain={vi.fn()} />);
    const label = screen.getByTestId("primary-action").textContent;
    cleanup();
    return label;
  };

  it("primary action per grade: correct -> Next sign; close/confused/incorrect/not_detected/error -> Try again", () => {
    expect(primaryLabel("correct")).toBe("Next sign");
    for (const r of ["close", "confused", "incorrect", "not_detected", "error"] as ResultKind[]) expect(primaryLabel(r)).toBe("Try again");
  });

  it("wires the primary action correctly; the quieter secondary only appears when given", () => {
    const onNext = vi.fn(), onTryAgain = vi.fn(), onSecondary = vi.fn();
    const { rerender } = render(<ResultPanel result="correct" message="Perfect!" onNext={onNext} onTryAgain={onTryAgain} />);
    expect(screen.queryByTestId("secondary-action")).toBeNull();
    fireEvent.click(screen.getByTestId("primary-action"));
    expect(onNext).toHaveBeenCalledTimes(1);
    rerender(<ResultPanel result="close" message="Almost" onNext={onNext} onTryAgain={onTryAgain} secondary={{ label: "Show me the sign", onClick: onSecondary }} />);
    fireEvent.click(screen.getByTestId("secondary-action"));
    expect(onSecondary).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByTestId("primary-action"));
    expect(onTryAgain).toHaveBeenCalledTimes(1);
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it("shows the feedback message and leaves an empty, marked slot for future per-part feedback", () => {
    render(<ResultPanel result="confused" message="That looked like “eat”." onNext={vi.fn()} onTryAgain={vi.fn()} />);
    expect(screen.getByTestId("result-message").textContent).toBe("That looked like “eat”.");
    const slot = document.querySelector('[data-slot="pose-coach-feedback"]')!;
    expect(slot).toBeTruthy();
    expect(slot.childElementCount).toBe(0);
    expect(slot.textContent).toBe("");
  });

  it("is rendered below the tiles, not inside a video area", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    goToResult("close");
    const panel = screen.getByTestId("result-panel");
    for (const v of [exampleVideo(), cameraVideo()]) expect(v.parentElement!.contains(panel)).toBe(false);
    expect(screen.getByTestId("stage").getAttribute("data-mode")).toBe("result");
    expect(screen.getByTestId("result-message").textContent).toBe("feedback for close");
  });

  it.each(["correct", "close", "confused", "incorrect", "not_detected"] as ResultKind[])("Stage shows the right primary action for %s", (r) => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    goToResult(r);
    expect(screen.getByTestId("primary-action").textContent).toBe(r === "correct" ? "Next sign" : "Try again");
  });

  it("'Next sign' calls onNext; 'Try again' returns to ready", () => {
    const onNext = vi.fn();
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={onNext} />);
    goToResult("correct");
    fireEvent.click(screen.getByTestId("primary-action"));
    expect(onNext).toHaveBeenCalledTimes(1);
    cleanup();
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    goToResult("incorrect");
    fireEvent.click(screen.getByTestId("primary-action"));
    expect(mode()).toBe("ready");
  });

  it("Practice has no 'Watch again' (the example is already on screen)", () => {
    render(<Stage word="eat" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    goToResult("incorrect");
    expect(screen.queryByTestId("watch-again")).toBeNull();
    expect(screen.queryByTestId("secondary-action")).toBeNull();
  });
});

describe("Practice | Quiz switch", () => {
  it("sits centred below the cards, marks the current view, and asks the page to change it", () => {
    const onViewChange = vi.fn();
    render(<Stage word="hi" view="practice" onViewChange={onViewChange} onNext={vi.fn()} />);
    const sw = screen.getByTestId("view-switch");
    expect(screen.getByTestId("action-bar").contains(sw)).toBe(true);
    expect(within(sw).getByRole("button", { name: "Practice" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(within(sw).getByRole("button", { name: "Quiz" }));
    expect(onViewChange).toHaveBeenCalledWith("quiz");
  });

  it("is only offered while ready (not mid-take or on the result)", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    goToResult("close");
    expect(screen.queryByTestId("view-switch")).toBeNull();
  });
});

describe("Quiz view", () => {
  const renderQuiz = (onViewChange = vi.fn()) => render(<Stage word="hi" view="quiz" onViewChange={onViewChange} onNext={vi.fn()} />);

  it("shows only your camera: no example video or its controls", () => {
    renderQuiz();
    expect(screen.getByTestId("stage").getAttribute("data-view")).toBe("quiz");
    expect(screen.queryByTestId("example-tile")).toBeNull();
    expect(screen.queryByTestId("example-video")).toBeNull();
    for (const id of ["play-pause", "slow", "loop"]) expect(screen.queryByTestId(id)).toBeNull();
    expect(screen.getByTestId("camera-tile")).toBeTruthy();
    expect(screen.getByTestId("record")).toBeTruthy();
  });

  it("runs the same record -> result flow", () => {
    renderQuiz();
    session.nextResult = "correct";
    fireEvent.click(screen.getByTestId("record"));
    fireEvent.click(screen.getByTestId("stop"));
    expect(mode()).toBe("result");
    expect(screen.getByTestId("primary-action").textContent).toBe("Next sign");
  });

  it("the result offers 'Show me the sign', which asks the page to switch to Practice", () => {
    const onViewChange = vi.fn();
    renderQuiz(onViewChange);
    goToResult("incorrect");
    fireEvent.click(screen.getByTestId("secondary-action"));
    expect(screen.getByTestId("secondary-action").textContent).toBe("Show me the sign");
    expect(onViewChange).toHaveBeenCalledWith("practice");
  });
});

describe("Keyboard shortcuts", () => {
  it("S cycles slow, Space plays/pauses, R records then stops, N goes next", () => {
    const onNext = vi.fn();
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={onNext} />);
    fireEvent.keyDown(window, { key: "s" });
    expect(exampleVideo().playbackRate).toBe(0.5);
    fireEvent.keyDown(window, { key: " " });
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
    fireEvent.keyDown(window, { key: "r" });
    expect(mode()).toBe("recording");
    fireEvent.keyDown(window, { key: "R" });
    expect(session.stopRecording).toHaveBeenCalled();
    expect(mode()).toBe("result");
    fireEvent.keyDown(window, { key: "n" });
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it("ignores shortcuts while typing in a field", () => {
    render(
      <>
        <input aria-label="notes" />
        <Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />
      </>,
    );
    fireEvent.keyDown(screen.getByLabelText("notes"), { key: "s" });
    expect(exampleVideo().playbackRate).toBe(1);
  });

  it("all stage controls have accessible names", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    for (const b of screen.getAllByRole("button")) {
      expect((b.getAttribute("aria-label") ?? b.textContent ?? "").trim().length, b.outerHTML).toBeGreaterThan(0);
    }
    openSettings();
    for (const sw of screen.getAllByRole("switch")) expect((sw.getAttribute("aria-label") ?? sw.textContent ?? "").trim().length).toBeGreaterThan(0);
  });
});

describe("layout rules from review", () => {
  it("the three example controls are exactly the same size", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    const sizes = ["play-pause", "slow", "loop"].map((id) =>
      screen.getByTestId(id).className.split(" ").filter((c) => /^(h|min-h)-/.test(c)).sort().join(" "),
    );
    expect(new Set(sizes).size).toBe(1);
    expect(sizes[0]).toContain("h-14");
  });

  it("Upload + Record sit inside the camera card under the video; only the Practice | Quiz switch is below the cards", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    const tile = screen.getByTestId("camera-tile");
    expect(tile.contains(screen.getByTestId("upload"))).toBe(true);
    expect(tile.contains(screen.getByTestId("record"))).toBe(true);
    const bar = screen.getByTestId("action-bar");
    expect(within(bar).getAllByRole("button").map((b) => b.textContent)).toEqual(["Practice", "Quiz"]);
  });

  it("the camera card has no duplicate framing tip under the video (the status chip on the video is the only one)", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    expect(screen.getByTestId("camera-tile").textContent).not.toContain("Keep your head");
    expect(within(screen.getByTestId("camera-tile")).getAllByRole("status")[0].textContent).toContain("You're all set");
  });

  it("the 3-2-1 countdown is an overlay ON the camera video, with Cancel under it", () => {
    session.holdCountdown = true;
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("record"));
    expect(mode()).toBe("countdown");
    const countdown = screen.getByTestId("countdown");
    expect(cameraVideo().parentElement!.contains(countdown)).toBe(true); // inside the media well
    expect(countdown.textContent).toContain("Get ready");
    expect(countdown.textContent).toContain("3");
    expect(screen.getByTestId("camera-tile").contains(screen.getByTestId("cancel-countdown"))).toBe(true);
    fireEvent.click(screen.getByTestId("cancel-countdown"));
    expect(session.cancelCountdown).toHaveBeenCalled();
  });

  it("Stop is inside the camera card while recording", () => {
    render(<Stage word="hi" view="practice" onViewChange={vi.fn()} onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("record"));
    expect(screen.getByTestId("camera-tile").contains(screen.getByTestId("stop"))).toBe(true);
  });
});

