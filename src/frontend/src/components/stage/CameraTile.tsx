import { Lightbulb, Loader, RotateCcw, Settings } from "lucide-react";
import type { ReactNode, RefObject } from "react";
import { MediaScrim, MediaWell, OverlayChip, OverlayIconButton, OverlayToggle, PillButton, SurfaceCard } from "@/components/ds";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import type { StageMode } from "@/lib/stage/stageMachine";

interface Props {
  mode: StageMode;
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  /** Display-only: flips ONLY this tile's camera video + skeleton canvas (never the frames MediaPipe/MediaRecorder read). */
  mirrored: boolean;
  onMirroredChange: (mirrored: boolean) => void;
  showTracking: boolean;
  onShowTracking: (on: boolean) => void;
  hint: string | null;
  framingOk: boolean;
  modelStatus: "loading" | "ready" | "error";
  cameraError: boolean;
  elapsedMs: number;
  /** The take (or uploaded clip) shown for review in grading/result. */
  recordedUrl: string | null;
  /** 3-2-1 before recording: shown as an overlay on the video. */
  countdown?: number | null;
  /** Primary actions for the card footer, right under the video (Upload / Record, Cancel, Stop, ...). */
  actions?: ReactNode;
  className?: string;
}

export const formatTimer = (ms: number): string => {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

/**
 * Live camera card. The status/hint is an overlay chip along the TOP-left edge of the media well (headroom, never
 * the signing space); the tracking / mirror options live behind a gear button at the top right so they take no
 * video space; the Recording label + timer is a chip in the bottom-left corner.
 */
export const CameraTile = ({
  mode, videoRef, canvasRef, mirrored, onMirroredChange, showTracking, onShowTracking, hint, framingOk, modelStatus,
  cameraError, elapsedMs, recordedUrl, countdown, actions, className,
}: Props) => {
  const live = mode === "ready" || mode === "countdown" || mode === "recording";
  const reviewing = (mode === "grading" || mode === "result") && !!recordedUrl;
  const recording = mode === "recording";
  const grading = mode === "grading";
  const counting = mode === "countdown";

  const status =
    modelStatus === "loading" ? "Getting the sign checker ready…"
    : modelStatus === "error" ? "Couldn't load the sign checker. Check your connection and refresh."
    : cameraError ? "Camera is off. Allow camera access in your browser."
    : mode === "countdown" ? "Get ready…"
    : mode === "grading" || mode === "result" ? "Your attempt"
    : hint ?? (framingOk ? (recording ? "Looking good, keep signing." : "You're all set. Press Record.") : "Getting a good look at you…");
  const statusIsHint = live && !!hint && modelStatus === "ready" && !cameraError;

  const settings = live && (
    <Popover>
      <PopoverTrigger asChild>
        <OverlayIconButton label="Camera settings" data-testid="settings-button">
          <Settings className="h-5 w-5" aria-hidden="true" />
        </OverlayIconButton>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        sideOffset={8}
        data-testid="settings-popover"
        className="flex w-auto flex-col items-stretch gap-2 rounded-card bg-surface p-3 [border:var(--card-border)] shadow-lg"
      >
        <OverlayToggle label="Show tracking" checked={showTracking} onCheckedChange={onShowTracking} />
        <OverlayToggle label="Match the example" checked={!mirrored} onCheckedChange={(match) => onMirroredChange(!match)} />
      </PopoverContent>
    </Popover>
  );

  const footer = (
    <div className="space-y-3">
      {actions ??
        (reviewing && !grading ? (
          <PillButton
            size="lg"
            variant="secondary"
            block
            aria-label="Replay my attempt"
            icon={<RotateCcw className="h-5 w-5" />}
            onClick={(e) => {
              const v = (e.currentTarget.closest("section") as HTMLElement | null)?.querySelector<HTMLVideoElement>('[data-testid="playback-video"]');
              if (v) {
                v.currentTime = 0;
                void v.play();
              }
            }}
          >
            Replay my take
          </PillButton>
        ) : null)}
    </div>
  );

  const scrim = counting ? (
    <MediaScrim passive tone="ink" label="Get ready" data-testid="countdown">
      <span aria-live="assertive" className="text-8xl font-black leading-none tabular-nums">
        {countdown ?? ""}
      </span>
    </MediaScrim>
  ) : grading ? (
    <MediaScrim passive icon={<Loader className="h-12 w-12 animate-spin" />} label="Analyzing your sign…" data-testid="analyzing" />
  ) : null;

  return (
    <SurfaceCard aria-label="Your camera" data-testid="camera-tile" data-recording={recording || undefined} className={cn("mx-auto w-full max-w-[var(--card-max-w)]", recording && "ring-2 ring-sage-600", className)} footer={footer}>
      <MediaWell
        scrim={scrim}
        topLeft={
          !counting &&
          !grading && (
            <OverlayChip
              role="status"
              aria-live="polite"
              tone={statusIsHint ? "default" : "sage"}
              icon={statusIsHint ? <Lightbulb className="h-4 w-4" /> : undefined}
            >
              {status}
            </OverlayChip>
          )
        }
        topRight={settings}
        bottomLeft={
          recording && (
            <OverlayChip
              role="status"
              className="font-semibold text-ink"
              icon={<span className="inline-block h-2.5 w-2.5 rounded-full bg-sage-600 animate-pulse" />}
            >
              Recording{" "}
              <time className="tabular-nums" aria-label={`${Math.floor(elapsedMs / 1000)} seconds`}>
                {formatTimer(elapsedMs)}
              </time>
            </OverlayChip>
          )
        }
      >
        <video
          ref={videoRef}
          data-testid="camera-video"
          className={cn(mirrored && "scale-x-[-1]", reviewing && "hidden")}
          autoPlay
          muted
          playsInline
          aria-label="Your camera"
        />
        <canvas ref={canvasRef} data-testid="camera-canvas" className={cn(mirrored && "scale-x-[-1]", reviewing && "hidden")} />
        {reviewing && (
          <video
            key={recordedUrl}
            data-testid="playback-video"
            src={recordedUrl!}
            autoPlay
            muted
            playsInline
            aria-label="Replay of your attempt"
            onEnded={(e) => e.currentTarget.pause()}
          />
        )}
      </MediaWell>
    </SurfaceCard>
  );
};
