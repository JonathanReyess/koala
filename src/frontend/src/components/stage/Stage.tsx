import { useCallback, useEffect, useReducer, useRef, useState, type CSSProperties } from "react";
import { Camera, Square, Upload } from "lucide-react";
import { ExampleTile } from "./ExampleTile";
import { CameraTile } from "./CameraTile";
import { ResultPanel } from "./ResultPanel";
import { ViewSwitch } from "./ViewSwitch";
import { PillButton, PillButtonRow } from "@/components/ds";
import { usePracticeSession } from "@/hooks/usePracticeSession";
import { useExamplePlayer } from "@/hooks/useExamplePlayer";
import { useFitToViewport } from "@/hooks/useFitToViewport";
import { initialStageState, stageReducer, type ResultKind, type StageView } from "@/lib/stage/stageMachine";
import { stageCssVars } from "@/lib/stage/theme";
import { cn } from "@/lib/utils";

interface StageProps {
  word: string;
  /** Practice: example video + camera. Quiz: only the word (shown by the page) and your camera. */
  view: StageView;
  /** Ask the page to change view (Quiz result: "Show me the sign"). */
  onViewChange: (view: StageView) => void;
  /** Advance to the next word (parent owns the queue/deck). */
  onNext: () => void;
  onFeedback?: (word: string, correct: boolean) => void;
}

const MIRROR_KEY = "koala.mirrorCamera";
const loadMirrored = (): boolean => {
  try {
    return localStorage.getItem(MIRROR_KEY) !== "0"; // mirrored (selfie view) unless the learner chose "Match the example"
  } catch {
    return true;
  }
};

const isTypingTarget = (t: EventTarget | null) =>
  t instanceof HTMLElement && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));

/**
 * The stage for one word. Practice view: example and camera side by side. Quiz view: just your camera, the word is
 * the prompt. Both run the same record -> grade -> result flow (state machine: lib/stage/stageMachine.ts).
 */
export const Stage = ({ word, view, onViewChange, onNext, onFeedback }: StageProps) => {
  const [state, dispatch] = useReducer(stageReducer, word, initialStageState);
  useEffect(() => dispatch({ type: "SET_WORD", word }), [word]);

  const [mirrored, setMirroredState] = useState(loadMirrored);
  const setMirrored = (m: boolean) => {
    setMirroredState(m);
    try {
      localStorage.setItem(MIRROR_KEY, m ? "1" : "0");
    } catch {
      /* preference just isn't remembered */
    }
  };

  const player = useExamplePlayer(word, state.example, state.rate, state.loop);
  const session = usePracticeSession({ word, mode: state.mode, dispatch, onFeedback });
  const fileRef = useRef<HTMLInputElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  const goNext = useCallback(() => {
    dispatch({ type: "NEXT", word });
    onNext();
  }, [word, onNext]);

  const recordOrStop = useCallback(() => {
    switch (state.mode) {
      case "ready":
      case "result":
        if (session.modelStatus === "ready") void session.beginRecording();
        break;
      case "countdown":
        session.cancelCountdown();
        break;
      case "recording":
        session.stopRecording();
        break;
    }
  }, [state.mode, session]);

  // Desktop shortcuts: Space = play/pause, S = slow, R = record/stop, N = next.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || isTypingTarget(e.target)) return;
      const onControl = e.target instanceof HTMLElement && !!e.target.closest("button, a, [role=switch], [role=checkbox]");
      switch (e.key.toLowerCase()) {
        case " ":
          if (onControl) return; // let a focused button handle its own Space
          e.preventDefault();
          player.togglePlay();
          break;
        case "s":
          dispatch({ type: "CYCLE_RATE" });
          break;
        case "r":
          recordOrStop();
          break;
        case "n":
          goNext();
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [player, recordOrStop, goNext]);

  const quiz = view === "quiz";
  // Size the videos so the page never scrolls on laptop/desktop screens (re-measured per mode / view / word / result).
  useFitToViewport(rootRef, [state.mode, state.result, view, word, session.modelStatus]);
  const ready = session.modelStatus === "ready";
  const result: ResultKind | null = state.result;

  // Actions that live in the camera card, right under the video.
  const cameraActions =
    state.mode === "ready" ? (
      <PillButtonRow className="flex-col-reverse sm:flex-row">
        <PillButton size="lg" variant="secondary" onClick={() => fileRef.current?.click()} disabled={!ready} icon={<Upload className="h-5 w-5" />} data-testid="upload">
          Upload a video
        </PillButton>
        <PillButton size="lg" onClick={recordOrStop} disabled={!ready || session.cameraError} icon={<Camera className="h-5 w-5" />} data-testid="record" aria-keyshortcuts="R">
          Record
        </PillButton>
      </PillButtonRow>
    ) : state.mode === "countdown" ? (
      <PillButton size="lg" variant="secondary" block onClick={recordOrStop} data-testid="cancel-countdown">
        Cancel
      </PillButton>
    ) : state.mode === "recording" ? (
      <PillButton size="lg" block onClick={recordOrStop} icon={<Square className="h-5 w-5 fill-current" />} data-testid="stop" aria-keyshortcuts="R">
        Stop
      </PillButton>
    ) : state.mode === "grading" ? (
      <PillButton size="lg" block disabled>
        Analyzing…
      </PillButton>
    ) : undefined;

  return (
    <div
      ref={rootRef}
      data-testid="stage"
      data-mode={state.mode}
      data-view={view}
      style={
        {
          ...stageCssVars(undefined, state.mode),
          "--split": quiz ? "var(--split-quiz)" : "var(--split-practice)",
        } as CSSProperties
      }
      className="w-full space-y-7"
    >
      <div className="grid grid-cols-1 gap-5 md:justify-center md:items-start md:grid-cols-[var(--split)]">
        {!quiz && (
          <ExampleTile
            word={word}
            example={state.example}
            rate={state.rate}
            loop={state.loop}
            playing={player.playing}
            videoRef={player.videoRef}
            onExample={(example) => dispatch({ type: "SET_EXAMPLE", example })}
            onTogglePlay={player.togglePlay}
            onCycleRate={() => dispatch({ type: "CYCLE_RATE" })}
            onToggleLoop={() => dispatch({ type: "TOGGLE_LOOP" })}
            onVideoReady={player.onVideoReady}
            onPlayState={player.setPlaying}
          />
        )}

        {/* Camera: always on screen (the only tile in Quiz). */}
        <div className="w-full">
          <CameraTile
            mode={state.mode}
            videoRef={session.videoRef}
            canvasRef={session.canvasRef}
            mirrored={mirrored}
            onMirroredChange={setMirrored}
            showTracking={session.showTracking}
            onShowTracking={session.setShowTracking}
            hint={session.tracker.hint}
            framingOk={session.tracker.framingOk}
            modelStatus={session.modelStatus}
            cameraError={session.cameraError}
            elapsedMs={session.elapsedMs}
            recordedUrl={session.recordedUrl}
            countdown={session.countdown}
            actions={cameraActions}
          />
        </div>
      </div>

      {/* Below the cards: the Practice | Quiz switch while ready, the result panel after a take. Upload / Record / Stop live in the camera card; keep the height otherwise so nothing jumps. */}
      {state.mode === "result" && result ? (
        <ResultPanel
          result={result}
          message={session.feedbackText}
          onNext={goNext}
          onTryAgain={() => dispatch({ type: "TRY_AGAIN" })}
          secondary={quiz ? { label: "Show me the sign", onClick: () => onViewChange("practice") } : undefined}
        >
          {session.debug && (
            <div className="mt-3 rounded-media border border-dashed border-sage-200 p-2 text-xs text-ink-muted">
              (debug) pose-coach slot: handshape · location · movement (not built yet)
            </div>
          )}
        </ResultPanel>
      ) : state.mode === "ready" ? (
        <div className="flex justify-center" data-testid="action-bar">
          <ViewSwitch view={view} onChange={onViewChange} />
        </div>
      ) : (
        <div className="h-12" aria-hidden="true" />
      )}

      <input
        ref={fileRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) void session.uploadFile(f);
        }}
      />

      {session.debug && (
        <div className="space-y-2">
          {session.debugInfo && (
            <pre className="text-xs bg-sage-50 text-ink rounded-media p-3 whitespace-pre-wrap font-mono">{session.debugInfo}</pre>
          )}
          <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">
            <PillButton size="sm" variant="secondary" disabled={session.loggedAttempts === 0} onClick={session.downloadAttempts}>
              Download attempts (JSON)
            </PillButton>
            <PillButton size="sm" variant="ghost" disabled={session.loggedAttempts === 0} onClick={session.clearAttempts}>
              Clear
            </PillButton>
            <span>{session.loggedAttempts} attempt(s) logged in memory. Nothing is uploaded.</span>
          </div>
        </div>
      )}
    </div>
  );
};
