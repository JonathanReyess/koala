import { useCallback, useEffect, useReducer, useRef, useState, type CSSProperties } from "react";
import { Camera, Square, Upload, Video } from "lucide-react";
import { ExampleTile } from "./ExampleTile";
import { CameraTile } from "./CameraTile";
import { AnalyzingPanel, ResultPanel } from "./ResultPanel";
import { PrimaryButton, QuietButton } from "./buttons";
import { usePracticeSession } from "@/hooks/usePracticeSession";
import { useExamplePlayer } from "@/hooks/useExamplePlayer";
import { initialStageState, stageReducer, type ResultKind } from "@/lib/stage/stageMachine";
import { stageCssVars } from "@/lib/stage/theme";
import { cn } from "@/lib/utils";

interface StageProps {
  word: string;
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
 * The Learn -> Practice stage for one word. Learn: big example video (+ optional camera picture-in-picture).
 * Practice: example and camera side by side. State machine: lib/stage/stageMachine.ts.
 */
export const Stage = ({ word, onNext, onFeedback }: StageProps) => {
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
  const session = usePracticeSession({ word, mode: state.mode, showMe: state.showMe, dispatch, onFeedback });
  const fileRef = useRef<HTMLInputElement>(null);

  const goNext = useCallback(() => {
    dispatch({ type: "NEXT", word });
    onNext();
  }, [word, onNext]);

  const recordOrStop = useCallback(() => {
    switch (state.mode) {
      case "practice":
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

  const inLearn = state.mode === "learn";
  const ready = session.modelStatus === "ready";
  const result: ResultKind | null = state.result;

  return (
    <div
      data-testid="stage"
      data-mode={state.mode}
      style={
        {
          ...stageCssVars(),
          "--split": inLearn ? "var(--split-learn)" : "var(--split-practice)",
        } as CSSProperties
      }
      className="w-full space-y-5"
    >
      <div className={cn("grid grid-cols-1 gap-5 md:grid-cols-[var(--split)]", inLearn ? "md:items-end" : "md:items-stretch")}>
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
          onReplay={player.replay}
          onToggleLoop={() => dispatch({ type: "TOGGLE_LOOP" })}
          onVideoReady={player.onVideoReady}
          onPlayState={player.setPlaying}
        />

        {/* Camera: a practice tile, or (Learn + "Show me") a picture-in-picture at the bottom-right of the stage,
            >= 22% wide (min 200px), inset from the corner, beside the example so it never covers its hands. */}
        <div
          className={cn(
            inLearn
              ? "w-[max(var(--pip-width),var(--pip-min-width))] justify-self-end md:w-full p-[var(--pip-inset)]"
              : "w-full h-full",
          )}
        >
          {inLearn && !state.showMe && (
            <div className="rounded-[var(--tile-radius)] border-[length:var(--tile-border)] border-dashed border-[color:var(--tile-border-color)] bg-white/70 p-4 flex flex-col items-center gap-3 text-center">
              <p className="text-sm font-medium text-gray-700">Want to check yourself? (optional)</p>
              <QuietButton large pressed={false} onClick={() => dispatch({ type: "TOGGLE_SHOW_ME" })} data-testid="show-me">
                <Camera className="h-6 w-6" aria-hidden="true" />
                Show me
              </QuietButton>
            </div>
          )}
          <CameraTile
            className={cn(inLearn && !state.showMe && "hidden")}
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
            compact={inLearn}
            onHide={() => dispatch({ type: "TOGGLE_SHOW_ME" })}
          />
        </div>
      </div>

      {/* Bottom bar: ONE primary action at a time. */}
      {state.mode === "result" && result ? (
        <ResultPanel
          result={result}
          message={session.feedbackText}
          onNext={goNext}
          onTryAgain={() => dispatch({ type: "TRY_AGAIN" })}
          onWatchAgain={() => dispatch({ type: "WATCH_AGAIN" })}
        >
          {session.debug && (
            <div className="mt-3 rounded-lg border border-dashed border-gray-400 bg-white/60 p-2 text-xs text-gray-600">
              (debug) pose-coach slot: handshape · location · movement — not built yet
            </div>
          )}
        </ResultPanel>
      ) : state.mode === "grading" ? (
        <AnalyzingPanel />
      ) : (
        <div className="flex flex-col-reverse sm:flex-row items-stretch sm:items-center justify-center gap-3" data-testid="action-bar">
          {inLearn && (
            <PrimaryButton onClick={() => dispatch({ type: "READY" })} data-testid="ready">
              <Video className="h-6 w-6" aria-hidden="true" />
              I'm ready to practice
            </PrimaryButton>
          )}

          {state.mode === "practice" && (
            <>
              <QuietButton onClick={() => dispatch({ type: "WATCH_AGAIN" })} data-testid="watch-again">
                Watch again
              </QuietButton>
              <QuietButton onClick={() => fileRef.current?.click()} disabled={!ready} data-testid="upload">
                <Upload className="h-5 w-5" aria-hidden="true" />
                Upload a video
              </QuietButton>
              <PrimaryButton onClick={recordOrStop} disabled={!ready || session.cameraError} data-testid="record" aria-keyshortcuts="R">
                <Camera className="h-6 w-6" aria-hidden="true" />
                Record
              </PrimaryButton>
            </>
          )}

          {state.mode === "countdown" && (
            <>
              <QuietButton onClick={recordOrStop} data-testid="cancel-countdown">
                Cancel
              </QuietButton>
              <div role="status" aria-live="assertive" className="text-center min-w-[9rem]" data-testid="countdown">
                <span className="block text-sm font-bold uppercase tracking-wide text-gray-700">Get ready</span>
                <span className="block text-6xl font-black tabular-nums leading-none">{session.countdown ?? ""}</span>
              </div>
            </>
          )}

          {state.mode === "recording" && (
            <PrimaryButton onClick={recordOrStop} data-testid="stop" aria-keyshortcuts="R">
              <Square className="h-6 w-6 fill-current" aria-hidden="true" />
              Stop
            </PrimaryButton>
          )}
        </div>
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
            <pre className="text-xs bg-gray-100 rounded-lg p-3 whitespace-pre-wrap font-mono">{session.debugInfo}</pre>
          )}
          <div className="flex items-center gap-2 text-xs text-gray-600">
            <QuietButton disabled={session.loggedAttempts === 0} onClick={session.downloadAttempts} className="min-h-[var(--tap-min)]">
              Download attempts (JSON)
            </QuietButton>
            <QuietButton disabled={session.loggedAttempts === 0} onClick={session.clearAttempts}>
              Clear
            </QuietButton>
            <span>{session.loggedAttempts} attempt(s) logged in memory — nothing is uploaded.</span>
          </div>
        </div>
      )}
    </div>
  );
};
