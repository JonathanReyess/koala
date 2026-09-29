import { Check, Pause, Play, Repeat, RotateCcw, Turtle } from "lucide-react";
import { Tile, TileBar, TileVideoArea } from "./tile";
import { QuietButton, PrimaryButton } from "./buttons";
import { exampleSrc, rateLabel } from "@/lib/stage/player";
import type { PlaybackRate } from "@/lib/stage/stageMachine";
import type { RefObject } from "react";

interface Props {
  word: string;
  example: 1 | 2;
  rate: PlaybackRate;
  loop: boolean;
  playing: boolean;
  videoRef: RefObject<HTMLVideoElement>;
  onExample: (n: 1 | 2) => void;
  onTogglePlay: () => void;
  onCycleRate: () => void;
  onReplay: () => void;
  onToggleLoop: () => void;
  /** Called on every metadata load / source change so speed + loop are re-applied. */
  onVideoReady: () => void;
  onPlayState: (playing: boolean) => void;
  className?: string;
}

/** The example video with big custom controls (no native controls): Play/Pause, Slow, Replay, Loop. */
export const ExampleTile = ({
  word, example, rate, loop, playing, videoRef, onExample, onTogglePlay, onCycleRate, onReplay, onToggleLoop, onVideoReady, onPlayState, className,
}: Props) => (
  <Tile aria-label={`Example of ${word}`} data-testid="example-tile" className={className}>
    <TileBar>
      <span className="text-base font-bold text-gray-900">Watch</span>
      <div role="group" aria-label="Choose an example signer" className="flex gap-2">
        {([1, 2] as const).map((n) => (
          <QuietButton key={n} pressed={example === n} onClick={() => onExample(n)} aria-label={`Example ${n}`}>
            Example {n}
          </QuietButton>
        ))}
      </div>
    </TileBar>

    <TileVideoArea>
      <video
        key={`${word}-${example}`}
        ref={videoRef}
        data-testid="example-video"
        src={exampleSrc(word, example)}
        className="absolute inset-0 w-full h-full object-cover"
        style={{ objectPosition: "var(--video-position)" }}
        autoPlay
        muted
        playsInline
        loop={loop}
        onLoadedMetadata={onVideoReady}
        onPlay={() => onPlayState(true)}
        onPause={() => onPlayState(false)}
        aria-label={`Demonstration of the sign for ${word}, example ${example}`}
      />
    </TileVideoArea>

    <TileBar className="justify-center gap-1.5 mt-auto">
      <PrimaryButton onClick={onTogglePlay} aria-label={playing ? "Pause" : "Play"} className="px-3.5 text-lg gap-2" data-testid="play-pause">
        {playing ? <Pause className="h-7 w-7" aria-hidden="true" /> : <Play className="h-7 w-7" aria-hidden="true" />}
        {playing ? "Pause" : "Play"}
      </PrimaryButton>
      <QuietButton
        large
        onClick={onCycleRate}
        aria-label={`Playback speed ${rateLabel(rate)}. Press to change.`}
        pressed={rate !== 1}
        className="px-3"
        data-testid="slow"
      >
        <Turtle className="h-6 w-6" aria-hidden="true" />
        Slow <span className="tabular-nums">{rateLabel(rate)}</span>
      </QuietButton>
      <QuietButton large onClick={onReplay} aria-label="Replay from the start" className="px-3" data-testid="replay">
        <RotateCcw className="h-5 w-5" aria-hidden="true" />
        Replay
      </QuietButton>
      <QuietButton onClick={onToggleLoop} pressed={loop} aria-label={`Loop ${loop ? "on" : "off"}`} className="px-3" data-testid="loop">
        <Repeat className="h-5 w-5" aria-hidden="true" />
        Loop
        {loop && <Check className="h-4 w-4" aria-hidden="true" />}
      </QuietButton>
    </TileBar>
  </Tile>
);
