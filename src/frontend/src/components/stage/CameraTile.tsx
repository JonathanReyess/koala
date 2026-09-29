import { Lightbulb, RotateCcw, Sparkles, EyeOff } from "lucide-react";
import type { RefObject } from "react";
import { Switch } from "@/components/ui/switch";
import { Tile, TileBar, TileVideoArea } from "./tile";
import { QuietButton } from "./buttons";
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

/** Live camera tile. Framing hints and the tracking / mirror toggles sit in the TOP bar, never over the signing space. */
export const CameraTile = ({
  mode, videoRef, canvasRef, mirrored, onMirroredChange, showTracking, onShowTracking, hint, framingOk, modelStatus,
  cameraError, elapsedMs, recordedUrl, compact, onHide, className,
}: Props) => {
  const live = mode === "practice" || mode === "countdown" || mode === "recording" || mode === "learn";
  const reviewing = (mode === "grading" || mode === "result") && !!recordedUrl;
  const recording = mode === "recording";

  const status =
    modelStatus === "loading" ? "Getting the sign checker ready…"
    : modelStatus === "error" ? "Couldn't load the sign checker — check your connection and refresh."
    : cameraError ? "Camera is off — allow camera access in your browser."
    : mode === "countdown" ? "Get ready…"
    : mode === "grading" || mode === "result" ? "Your attempt"
    : hint ?? (framingOk ? (recording ? "Looking good — keep signing." : "You're all set — press Record.") : "Getting a good look at you…");
  const statusIsHint = live && !!hint && modelStatus === "ready" && !cameraError;

  return (
    <Tile aria-label="Your camera" data-testid="camera-tile" recording={recording} className={className}>
      {!compact && (
        <TileBar>
          <span
            role="status"
            aria-live="polite"
            className={cn(
              "inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-semibold",
              statusIsHint ? "bg-amber-100 text-amber-900" : "bg-white text-gray-800",
            )}
          >
            {statusIsHint ? <Lightbulb className="h-4 w-4" aria-hidden="true" /> : <Sparkles className="h-4 w-4" aria-hidden="true" />}
            {status}
          </span>
          {live && (
            <div className="flex flex-wrap items-center gap-x-4">
              <label className="flex items-center gap-2 min-h-[var(--tap-min)] text-sm font-semibold text-gray-900 cursor-pointer">
                <Switch checked={showTracking} onCheckedChange={onShowTracking} aria-label="Show tracking" />
                Show tracking
              </label>
              <label className="flex items-center gap-2 min-h-[var(--tap-min)] text-sm font-semibold text-gray-900 cursor-pointer">
                <Switch checked={!mirrored} onCheckedChange={(match) => onMirroredChange(!match)} aria-label="Match the example" />
                Match the example
              </label>
            </div>
          )}
        </TileBar>
      )}

      <TileVideoArea>
        <video
          ref={videoRef}
          data-testid="camera-video"
          className={cn("absolute inset-0 w-full h-full object-cover", mirrored && "scale-x-[-1]", reviewing && "hidden")}
          style={{ objectPosition: "var(--video-position)" }}
          autoPlay
          muted
          playsInline
          aria-label="Your camera"
        />
        <canvas
          ref={canvasRef}
          data-testid="camera-canvas"
          className={cn("absolute inset-0 w-full h-full pointer-events-none", mirrored && "scale-x-[-1]", reviewing && "hidden")}
        />
        {reviewing && (
          <video
            key={recordedUrl}
            data-testid="playback-video"
            src={recordedUrl!}
            className="absolute inset-0 w-full h-full object-cover"
            style={{ objectPosition: "var(--video-position)" }}
            autoPlay
            muted
            playsInline
            aria-label="Replay of your attempt"
            onEnded={(e) => e.currentTarget.pause()}
          />
        )}
      </TileVideoArea>

      <TileBar className="justify-center mt-auto">
        {compact ? (
          <QuietButton onClick={onHide} aria-label="Hide my camera">
            <EyeOff className="h-4 w-4" aria-hidden="true" />
            Hide me
          </QuietButton>
        ) : recording ? (
          <span role="status" className="inline-flex items-center gap-3 text-lg font-extrabold text-[color:var(--tile-border-recording)]">
            <span aria-hidden="true" className="inline-block h-4 w-4 rounded-full bg-[color:var(--tile-border-recording)] animate-pulse" />
            Recording
            <time className="tabular-nums" aria-label={`${Math.floor(elapsedMs / 1000)} seconds`}>{formatTimer(elapsedMs)}</time>
          </span>
        ) : reviewing ? (
          <QuietButton
            onClick={(e) => {
              const v = (e.currentTarget.closest("section") as HTMLElement | null)?.querySelector<HTMLVideoElement>('[data-testid="playback-video"]');
              if (v) {
                v.currentTime = 0;
                void v.play();
              }
            }}
            aria-label="Replay my attempt"
          >
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
            Replay my take
          </QuietButton>
        ) : (
          <span className="text-sm font-medium text-gray-700">Keep your head, body and both hands in view.</span>
        )}
      </TileBar>
    </Tile>
  );
};
