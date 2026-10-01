import { Check, Pause, Play, Repeat, RotateCcw, Turtle } from "lucide-react";
import type { RefObject } from "react";
import { MediaWell, OverlayChip, PillButton, SurfaceCard } from "@/components/ds";
import { exampleSrc, rateLabel } from "@/lib/stage/player";
import type { PlaybackRate } from "@/lib/stage/stageMachine";
import { cn } from "@/lib/utils";

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
  /** Practice mode: the card is half width, so controls sit in a 2x2 grid instead of one row. */
  compact?: boolean;
  className?: string;
}

/** The example video (inset media well) with big custom controls in the card footer: Play/Pause, Slow, Replay, Loop. */
export const ExampleTile = ({
  word, example, rate, loop, playing, videoRef, onExample, onTogglePlay, onCycleRate, onReplay, onToggleLoop, onVideoReady, onPlayState, compact, className,
}: Props) => (
  <SurfaceCard
    aria-label={`Example of ${word}`}
    data-testid="example-tile"
    className={cn("h-full", className)}
    footer={
      <div className={cn("grid grid-cols-2 gap-3 [&>*]:min-w-0", !compact && "md:grid-cols-4")}>
        <PillButton
          size="lg"
          onClick={onTogglePlay}
          aria-label={playing ? "Pause" : "Play"}
          icon={playing ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6" />}
          data-testid="play-pause"
        >
          {playing ? "Pause" : "Play"}
        </PillButton>
        <PillButton
          size="lg"
          variant={rate !== 1 ? "primary" : "secondary"}
          onClick={onCycleRate}
          aria-label={`Playback speed ${rateLabel(rate)}. Press to change.`}
          icon={<Turtle className="h-6 w-6" />}
          data-testid="slow"
        >
          Slow <span className="tabular-nums">{rateLabel(rate)}</span>
        </PillButton>
        <PillButton size="touch" variant="secondary" onClick={onReplay} aria-label="Replay from the start" icon={<RotateCcw className="h-5 w-5" />} data-testid="replay">
          Replay
        </PillButton>
        <PillButton
          size="touch"
          variant={loop ? "secondary" : "ghost"}
          onClick={onToggleLoop}
          aria-label={`Loop ${loop ? "on" : "off"}`}
          icon={<Repeat className="h-5 w-5" />}
          data-testid="loop"
        >
          Loop
          {loop && <Check className="h-4 w-4" aria-hidden="true" />}
        </PillButton>
      </div>
    }
  >
    <MediaWell
      style={{ maxHeight: "var(--video-max-h)" }}
      topLeft={
        <OverlayChip className="p-1" data-testid="example-switch">
          <span role="group" aria-label="Choose an example signer" className="flex gap-1">
            {([1, 2] as const).map((n) => (
              <button
                key={n}
                type="button"
                aria-pressed={example === n}
                aria-label={`Example ${n}`}
                onClick={() => onExample(n)}
                className={cn(
                  "min-h-8 rounded-chip px-3 text-sm font-semibold transition-colors",
                  "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage-600",
                  example === n ? "bg-sage-600 text-white" : "text-ink hover:bg-sage-50",
                )}
              >
                Example {n}
              </button>
            ))}
          </span>
        </OverlayChip>
      }
    >
      <video
        key={`${word}-${example}`}
        ref={videoRef}
        data-testid="example-video"
        src={exampleSrc(word, example)}
        autoPlay
        muted
        playsInline
        loop={loop}
        onLoadedMetadata={onVideoReady}
        onPlay={() => onPlayState(true)}
        onPause={() => onPlayState(false)}
        aria-label={`Demonstration of the sign for ${word}, example ${example}`}
      />
    </MediaWell>
  </SurfaceCard>
);
