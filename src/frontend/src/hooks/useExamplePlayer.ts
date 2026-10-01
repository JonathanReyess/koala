import { useCallback, useEffect, useRef, useState } from "react";
import { applyPlayback, replay as replayVideo } from "@/lib/stage/player";
import type { PlaybackRate } from "@/lib/stage/stageMachine";

/** Drives the example <video>: applies speed/loop from stage state, tracks play state, exposes Play/Replay. */
export function useExamplePlayer(word: string, example: 1 | 2, rate: PlaybackRate, loop: boolean) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(true); // autoplays muted

  const apply = useCallback(() => {
    if (videoRef.current) applyPlayback(videoRef.current, { rate, loop });
  }, [rate, loop]);

  useEffect(apply, [apply, word, example]);
  useEffect(() => setPlaying(true), [word, example]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  }, []);
  const replay = useCallback(() => {
    if (videoRef.current) replayVideo(videoRef.current);
  }, []);

  return { videoRef, playing, setPlaying, togglePlay, replay, onVideoReady: apply };
}
