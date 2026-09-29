// @vitest-environment jsdom
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import type { ResultKind } from "@/lib/stage/stageMachine";

// --- A controllable stand-in for the camera/model/grading session -------------------------------------------------
const session = vi.hoisted(() => ({
  nextResult: "incorrect" as string,
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
      dispatch({ type: "COUNTDOWN_DONE" }); // skip the real 3-2-1
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
  session.beginRecording.mockClear();
  session.stopRecording.mockClear();
  session.cancelCountdown.mockClear();
  localStorage.clear();
});
afterEach(cleanup);

const exampleVideo = () => screen.getByTestId("example-video") as HTMLVideoElement;
const cameraVideo = () => screen.getByTestId("camera-video") as HTMLVideoElement;
const mode = () => screen.getByTestId("stage").getAttribute("data-mode");
const goToResult = (result: ResultKind) => {
  session.nextResult = result;
  fireEvent.click(screen.getByTestId("ready"));
  fireEvent.click(screen.getByTestId("record"));
  fireEvent.click(screen.getByTestId("stop"));
};

describe("Learn mode", () => {
  it("shows the example with custom controls, one primary CTA, and no native video controls", () => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    expect(mode()).toBe("learn");
    expect(exampleVideo().hasAttribute("controls")).toBe(false);
    expect(exampleVideo().getAttribute("src")).toBe("/videos/hi_example1.mp4");
    for (const id of ["play-pause", "slow", "replay", "loop"]) expect(screen.getByTestId(id)).toBeTruthy();
    expect(screen.getByTestId("ready").textContent).toContain("I'm ready to practice");
    expect(screen.queryByTestId("record")).toBeNull();
    expect(exampleVideo().loop).toBe(true); // loop on by default
  });

  it("toggles between example 1 and example 2", () => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Example 2" }));
    expect(exampleVideo().getAttribute("src")).toBe("/videos/hi_example2.mp4");
  });

  it("Slow sets the video's playbackRate: 1x -> 0.5x -> 0.25x -> 1x", () => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    expect(exampleVideo().playbackRate).toBe(1);
    fireEvent.click(screen.getByTestId("slow"));
    expect(exampleVideo().playbackRate).toBe(0.5);
    expect(screen.getByTestId("slow").textContent).toContain("0.5×");
    fireEvent.click(screen.getByTestId("slow"));
    expect(exampleVideo().playbackRate).toBe(0.25);
    fireEvent.click(screen.getByTestId("slow"));
    expect(exampleVideo().playbackRate).toBe(1);
  });

  it("Loop toggles the video's loop attribute; Replay restarts and plays", () => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("loop"));
    expect(exampleVideo().loop).toBe(false);
    exampleVideo().currentTime = 5;
    fireEvent.click(screen.getByTestId("replay"));
    expect(exampleVideo().currentTime).toBe(0);
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
  });

  it("'Show me' turns the camera picture-in-picture on (compact tile with a Hide button); off by default", () => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    expect(screen.getByTestId("camera-tile").classList.contains("hidden")).toBe(true);
    fireEvent.click(screen.getByTestId("show-me"));
    expect(screen.getByTestId("camera-tile").classList.contains("hidden")).toBe(false);
    expect(screen.getByRole("button", { name: "Hide my camera" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Hide my camera" }));
    expect(screen.getByTestId("camera-tile").classList.contains("hidden")).toBe(true);
  });

  it("no auto-advance: stays in Learn until the CTA is pressed", () => {
    vi.useFakeTimers();
    render(<Stage word="hi" onNext={vi.fn()} />);
    vi.advanceTimersByTime(60_000);
    expect(mode()).toBe("learn");
    vi.useRealTimers();
  });
});

describe("Practice flow", () => {
  it("Learn -> Practice -> Recording (Stop) -> Result", () => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("ready"));
    expect(mode()).toBe("practice");
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
    render(<Stage word="hi" onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("ready"));
    expect(cameraVideo().className).toContain("scale-x-[-1]");
    expect(screen.getByTestId("camera-canvas").className).toContain("scale-x-[-1]");
    const exampleClassBefore = exampleVideo().className;
    fireEvent.click(screen.getByRole("switch", { name: "Match the example" }));
    expect(cameraVideo().className).not.toContain("scale-x-[-1]");
    expect(screen.getByTestId("camera-canvas").className).not.toContain("scale-x-[-1]");
    expect(exampleVideo().className).toBe(exampleClassBefore); // example untouched
    expect(exampleVideo().className).not.toContain("scale-x-[-1]");
    expect(localStorage.getItem("koala.mirrorCamera")).toBe("0");
    cleanup();
    render(<Stage word="hi" onNext={vi.fn()} />); // remembered
    fireEvent.click(screen.getByTestId("ready"));
    expect(cameraVideo().className).not.toContain("scale-x-[-1]");
  });

  it("framing hints and toggles are in the camera tile's top bar (not over the video)", () => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("ready"));
    const tile = screen.getByTestId("camera-tile");
    const videoArea = cameraVideo().parentElement!;
    expect(videoArea.contains(screen.getByRole("switch", { name: "Show tracking" }))).toBe(false);
    expect(videoArea.querySelectorAll("button, [role=switch], [role=status]").length).toBe(0);
    expect(within(tile).getByRole("switch", { name: "Show tracking" })).toBeTruthy();
    expect(within(tile).getAllByRole("status")[0].textContent).toContain("You're all set");
  });

  it("recording state is conveyed with text, not colour alone (Recording label + timer)", () => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("ready"));
    fireEvent.click(screen.getByTestId("record"));
    const tile = screen.getByTestId("camera-tile");
    expect(tile.textContent).toContain("Recording");
    expect(tile.textContent).toMatch(/\d:\d\d/);
  });
});

describe("Result panel", () => {
  const primaryLabel = (r: ResultKind) => {
    render(<ResultPanel result={r} message="msg" onNext={vi.fn()} onTryAgain={vi.fn()} onWatchAgain={vi.fn()} />);
    const label = screen.getByTestId("primary-action").textContent;
    cleanup();
    return label;
  };

  it("primary action per grade: correct -> Next sign; close/confused/incorrect/not_detected/error -> Try again", () => {
    expect(primaryLabel("correct")).toBe("Next sign");
    for (const r of ["close", "confused", "incorrect", "not_detected", "error"] as ResultKind[]) expect(primaryLabel(r)).toBe("Try again");
  });

  it("always offers 'Watch again' as the quieter secondary, and wires the primary correctly", () => {
    const onNext = vi.fn(), onTryAgain = vi.fn(), onWatchAgain = vi.fn();
    const { rerender } = render(<ResultPanel result="correct" message="Perfect!" onNext={onNext} onTryAgain={onTryAgain} onWatchAgain={onWatchAgain} />);
    fireEvent.click(screen.getByTestId("primary-action"));
    expect(onNext).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByTestId("watch-again"));
    expect(onWatchAgain).toHaveBeenCalledTimes(1);
    rerender(<ResultPanel result="close" message="Almost" onNext={onNext} onTryAgain={onTryAgain} onWatchAgain={onWatchAgain} />);
    fireEvent.click(screen.getByTestId("primary-action"));
    expect(onTryAgain).toHaveBeenCalledTimes(1);
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it("shows the feedback message and leaves an empty, marked slot for future per-part feedback", () => {
    render(<ResultPanel result="confused" message="That looked like “eat”." onNext={vi.fn()} onTryAgain={vi.fn()} onWatchAgain={vi.fn()} />);
    expect(screen.getByTestId("result-message").textContent).toBe("That looked like “eat”.");
    const slot = document.querySelector('[data-slot="pose-coach-feedback"]')!;
    expect(slot).toBeTruthy();
    expect(slot.childElementCount).toBe(0);
    expect(slot.textContent).toBe("");
  });

  it("is rendered below the tiles, not inside a video area", () => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    goToResult("close");
    const panel = screen.getByTestId("result-panel");
    for (const v of [exampleVideo(), cameraVideo()]) expect(v.parentElement!.contains(panel)).toBe(false);
    expect(screen.getByTestId("stage").getAttribute("data-mode")).toBe("result");
    expect(screen.getByTestId("result-message").textContent).toBe("feedback for close");
  });

  it.each(["correct", "close", "confused", "incorrect", "not_detected"] as ResultKind[])("Stage shows the right primary action for %s", (r) => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    goToResult(r);
    expect(screen.getByTestId("primary-action").textContent).toBe(r === "correct" ? "Next sign" : "Try again");
  });

  it("'Next sign' calls onNext; 'Try again' returns to Practice", () => {
    const onNext = vi.fn();
    render(<Stage word="hi" onNext={onNext} />);
    goToResult("correct");
    fireEvent.click(screen.getByTestId("primary-action"));
    expect(onNext).toHaveBeenCalledTimes(1);
    cleanup();
    render(<Stage word="hi" onNext={vi.fn()} />);
    goToResult("incorrect");
    fireEvent.click(screen.getByTestId("primary-action"));
    expect(mode()).toBe("practice");
  });

  it("Watch again returns to Learn on the SAME word with loop on (even if it was turned off)", () => {
    render(<Stage word="eat" onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("loop")); // loop off
    expect(exampleVideo().loop).toBe(false);
    fireEvent.click(screen.getByTestId("slow")); // 0.5x
    goToResult("incorrect");
    fireEvent.click(screen.getByTestId("watch-again"));
    expect(mode()).toBe("learn");
    expect(exampleVideo().getAttribute("src")).toContain("eat_example");
    expect(exampleVideo().loop).toBe(true);
    expect(screen.getByTestId("loop").getAttribute("aria-label")).toBe("Loop on");
    expect(screen.getByTestId("ready")).toBeTruthy();
  });
});

describe("Keyboard shortcuts", () => {
  it("S cycles slow, Space plays/pauses, R records then stops, N goes next", () => {
    const onNext = vi.fn();
    render(<Stage word="hi" onNext={onNext} />);
    fireEvent.keyDown(window, { key: "s" });
    expect(exampleVideo().playbackRate).toBe(0.5);
    fireEvent.keyDown(window, { key: " " });
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
    fireEvent.keyDown(window, { key: "r" }); // ignored in Learn
    expect(mode()).toBe("learn");
    fireEvent.click(screen.getByTestId("ready"));
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
        <Stage word="hi" onNext={vi.fn()} />
      </>,
    );
    fireEvent.keyDown(screen.getByLabelText("notes"), { key: "s" });
    expect(exampleVideo().playbackRate).toBe(1);
  });

  it("all stage controls have accessible names", () => {
    render(<Stage word="hi" onNext={vi.fn()} />);
    fireEvent.click(screen.getByTestId("ready"));
    for (const b of screen.getAllByRole("button")) {
      expect((b.getAttribute("aria-label") ?? b.textContent ?? "").trim().length, b.outerHTML).toBeGreaterThan(0);
    }
    for (const sw of screen.getAllByRole("switch")) expect(sw.getAttribute("aria-label")).toBeTruthy();
  });
});
