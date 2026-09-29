import type { PlaybackRate } from "./stageMachine";

/** The subset of HTMLVideoElement the example player drives (lets tests use a plain stub). */
export interface PlayableVideo {
  playbackRate: number;
  defaultPlaybackRate: number;
  loop: boolean;
  currentTime: number;
  play: () => Promise<void> | void;
}

/** Applies speed + loop to the element. defaultPlaybackRate too: changing `src` resets playbackRate to it. */
export function applyPlayback(video: PlayableVideo, { rate, loop }: { rate: PlaybackRate; loop: boolean }): void {
  video.defaultPlaybackRate = rate;
  video.playbackRate = rate;
  video.loop = loop;
}

export function replay(video: PlayableVideo): void {
  video.currentTime = 0;
  void video.play();
}

export const rateLabel = (rate: PlaybackRate): string => `${rate}×`;

/** URL of a demo clip (public/videos/<word>_example<n>.mp4). */
export const exampleSrc = (word: string, example: 1 | 2): string => `/videos/${encodeURIComponent(word)}_example${example}.mp4`;
