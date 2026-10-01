import { describe, it, expect, vi } from "vitest";
import { applyPlayback, exampleSrc, rateLabel, replay, type PlayableVideo } from "./player";

const stub = (): PlayableVideo => ({ playbackRate: 1, defaultPlaybackRate: 1, loop: false, currentTime: 12, play: vi.fn() });

describe("example player helpers", () => {
  it("applyPlayback sets playbackRate (and default rate, which survives source changes) and loop", () => {
    const v = stub();
    applyPlayback(v, { rate: 0.5, loop: true });
    expect(v).toMatchObject({ playbackRate: 0.5, defaultPlaybackRate: 0.5, loop: true });
    applyPlayback(v, { rate: 0.25, loop: false });
    expect(v).toMatchObject({ playbackRate: 0.25, loop: false });
  });
  it("replay restarts from 0 and plays", () => {
    const v = stub();
    replay(v);
    expect(v.currentTime).toBe(0);
    expect(v.play).toHaveBeenCalled();
  });
  it("labels and urls (words with spaces are encoded; the files keep their names)", () => {
    expect(rateLabel(0.25)).toBe("0.25×");
    expect(exampleSrc("how many", 2)).toBe("/videos/how%20many_example2.mp4");
    expect(decodeURIComponent(exampleSrc("Seoul", 1))).toBe("/videos/Seoul_example1.mp4");
  });
});
