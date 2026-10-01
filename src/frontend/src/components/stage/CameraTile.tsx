import { EyeOff, Lightbulb, RotateCcw, Sparkles } from "lucide-react";
import type { RefObject } from "react";
import { MediaWell, OverlayChip, OverlayToggle, PillButton, SurfaceCard } from "@/components/ds";
import { useMediaQuery } from "@/hooks/useMediaQuery";
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
  /** Learn-mode picture-in-picture: video only, plus a Hide button. */
  compact?: boolean;
  onHide?: () => void;
  className?: string;
}

export const formatTimer = (ms: number): string => {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

/**
 * Live camera card. Status/hints and the tracking / mirror toggles are overlay chips along the TOP edge of the
 * media well (headroom, never the signing space); the Recording label + timer is a chip in the bottom-left corner.
 */
export const CameraTile = ({
  mode, videoRef, canvasRef, mirrored, onMirroredChange, showTracking, onShowTracking, hint, framingOk, modelStatus,
  cameraError, elapsedMs, recordedUrl, compact, onHide, className,
}: Props) => {
  // On phones the media well is small: option toggles move to the card footer so they never cover the signer.
  const narrow = useMediaQuery("(max-width: 639px)");
  const live = mode === "practice" || mode === "countdown" || mode === "recording" || mode === "learn";
  const reviewing = (mode === "grading" || mode === "result") && !!recordedUrl;
  const recording = mode === "recording";

  const status =
    modelStatus === "loading" ? "Getting the sign checker ready…"
    : modelStatus === "error" ? "Couldn't load the sign checker. Check your connection and refresh."
    : cameraError ? "Camera is off. Allow camera access in your browser."
    : mode === "countdown" ? "Get ready…"
    : mode === "grading" || mode === "result" ? "Your attempt"
    : hint ?? (framingOk ? (recording ? "Looking good, keep signing." : "You're all set. Press Record.") : "Getting a good look at you…");
  const statusIsHint = live && !!hint && modelStatus === "ready" && !cameraError;

  const toggles = !compact && live && (
    <>
      <OverlayToggle label="Show tracking" checked={showTracking} onCheckedChange={onShowTracking} />
      <OverlayToggle label="Match the example" checked={!mirrored} onCheckedChange={(match) => onMirroredChange(!match)} />
    </>
  );

  const footer = compact ? (
    <PillButton size="sm" variant="secondary" block onClick={onHide} aria-label="Hide my camera" icon={<EyeOff className="h-4 w-4" />}>
      Hide me
    </PillButton>
  ) : reviewing ? (
    <PillButton
      size="touch"
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
  ) : narrow && toggles ? (
    <div className="flex flex-wrap items-center justify-center gap-2 min-h-12">{toggles}</div>
  ) : (
    <p className="text-center text-sm text-ink-muted min-h-12 flex items-center justify-center">Keep your head, body and both hands in view.</p>
  );

  return (
    <SurfaceCard aria-label="Your camera" data-testid="camera-tile" data-recording={recording || undefined} className={cn("h-full", recording && "ring-2 ring-sage-600", className)} footer={footer}>
      <MediaWell
        inset={compact ? "sm" : "md"}
        style={{ maxHeight: "var(--video-max-h)" }}
        topLeft={
          !compact && (
            <OverlayChip
              role="status"
              aria-live="polite"
              tone={statusIsHint ? "default" : "sage"}
              icon={statusIsHint ? <Lightbulb className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
            >
              {status}
            </OverlayChip>
          )
        }
        topRight={!narrow && toggles}
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
