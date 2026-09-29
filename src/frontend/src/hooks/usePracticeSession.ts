import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { loadModels, createHolisticLandmarker, LoadedModels } from "@/lib/inference/loader";
import { gradeClip } from "@/lib/inference/pipeline";
import type { Grade } from "@/lib/inference/grading";
import { feedbackMessage } from "@/lib/inference/messages";
import {
  attemptCount,
  attemptsToJson,
  buildAttemptRecord,
  clearAttempts,
  logAttempt,
  subscribeAttempts,
} from "@/lib/inference/attemptLog";
import { buildClip } from "@/lib/inference/preprocess";
import { isClassInModel } from "@/lib/inference/labels";
import { extractVideoMode, loadVideoElement } from "@/lib/inference/extract";
import { classIdForWord, wordForClassId, ALL_WORDS } from "@/data/words";
import { useLandmarkTracker } from "@/hooks/useLandmarkTracker";
import { cameraWanted, type ResultKind, type StageEvent, type StageMode } from "@/lib/stage/stageMachine";

const COUNTDOWN_SECONDS = 3;

interface Options {
  word: string;
  mode: StageMode;
  showMe: boolean;
  dispatch: (e: StageEvent) => void;
  /** Spaced-repetition hook: correct -> credit, confused/incorrect -> miss (close/not_detected: no call). */
  onFeedback?: (word: string, correct: boolean) => void;
}

const resultKindOf = (g: Grade): ResultKind => g.status;

/** Dev-only: `?forceResult=correct|close|confused|incorrect|not_detected|error` skips the model (for screenshots/UI work). */
function forcedResult(): string | null {
  if (!import.meta.env.DEV) return null;
  return new URLSearchParams(location.search).get("forceResult");
}

function forcedGrade(status: Grade["status"], targetId: number): Grade {
  const other = ALL_WORDS.find((w) => w.classId !== targetId)!.classId;
  const base = { countsAsAttempt: status !== "not_detected", countsAsMiss: status === "confused" || status === "incorrect" };
  switch (status) {
    case "correct":
      return { status, top1: targetId, top1Prob: 0.9, targetProb: 0.9, namesTop1: false, ...base };
    case "close":
      return { status, top1: other, top1Prob: 0.5, targetProb: 0.3, namesTop1: false, ...base };
    case "confused":
      return { status, top1: other, top1Prob: 0.9, targetProb: 0.01, namesTop1: true, ...base };
    case "incorrect":
      return { status, top1: other, top1Prob: 0.4, targetProb: 0.02, namesTop1: false, ...base };
    default:
      return { status: "not_detected", reason: "hands", ...base };
  }
}

/**
 * Everything behind the Practice stage that isn't layout: camera lifecycle, model loading, the 3-2-1 countdown,
 * recording (MediaRecorder + live landmarks), upload extraction, grading, the debug log. It reports progress
 * to the stage state machine through `dispatch`.
 */
export function usePracticeSession({ word, mode, showMe, dispatch, onFeedback }: Options) {
  const videoRef = useRef<HTMLVideoElement>(null); // live camera
  const canvasRef = useRef<HTMLCanvasElement>(null); // skeleton overlay (mirrors with the camera)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recorderActiveRef = useRef(false);
  const countdownTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const modeRef = useRef(mode);
  modeRef.current = mode;

  const [models, setModels] = useState<LoadedModels | null>(null);
  const [modelStatus, setModelStatus] = useState<"loading" | "ready" | "error">("loading");
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [grade, setGrade] = useState<Grade | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [debugInfo, setDebugInfo] = useState<string | null>(null);
  const debug = new URLSearchParams(location.search).get("debug") === "1";
  const loggedAttempts = useSyncExternalStore(subscribeAttempts, attemptCount);

  // Skeleton overlay is opt-in; the choice is remembered per browser.
  const [showTracking, setShowTrackingState] = useState<boolean>(() => {
    try {
      return localStorage.getItem("koala.showTracking") === "1";
    } catch {
      return false;
    }
  });
  const setShowTracking = (on: boolean) => {
    setShowTrackingState(on);
    try {
      localStorage.setItem("koala.showTracking", on ? "1" : "0");
    } catch {
      /* preference just isn't remembered */
    }
  };

  const tracker = useLandmarkTracker({
    videoRef,
    canvasRef,
    landmarker: models?.landmarker ?? null,
    active: cameraOn && (mode === "practice" || mode === "countdown" || mode === "recording" || (mode === "learn" && showMe)),
    recording: mode === "recording",
    drawOverlay: showTracking,
  });

  // --- Models: lazy-load + cache on entering the stage --------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    loadModels()
      .then((m) => {
        if (cancelled) return;
        setModels(m);
        setModelStatus("ready");
      })
      .catch((e) => {
        console.error("[koala] model load failed", e);
        if (!cancelled) setModelStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // --- Camera ---------------------------------------------------------------------------------------------
  const startCamera = useCallback(async () => {
    if (videoRef.current?.srcObject) return true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
      setCameraError(false);
      return true;
    } catch {
      console.error("Could not access camera.");
      setCameraError(true);
      return false;
    }
  }, []);

  const stopCamera = useCallback(() => {
    const v = videoRef.current;
    if (v?.srcObject) {
      (v.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      v.srcObject = null;
    }
    setCameraOn(false);
  }, []);

  const wantCamera = cameraWanted({ mode, showMe });
  useEffect(() => {
    if (wantCamera) void startCamera();
    else if (!recorderActiveRef.current) stopCamera(); // the recorder's onstop releases the camera after flushing
  }, [wantCamera, startCamera, stopCamera]);
  useEffect(() => () => stopCamera(), [stopCamera]);

  // --- Reset per word -------------------------------------------------------------------------------------
  const clearTimers = () => {
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    countdownTimerRef.current = elapsedTimerRef.current = null;
  };
  const discardTake = useCallback(() => {
    setRecordedUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return null;
    });
    setGrade(null);
    setDebugInfo(null);
    setCountdown(null);
    setElapsedMs(0);
  }, []);
  useEffect(() => {
    clearTimers();
    if (mediaRecorderRef.current && recorderActiveRef.current) {
      mediaRecorderRef.current.onstop = null;
      mediaRecorderRef.current.stop();
      recorderActiveRef.current = false;
    }
    discardTake();
  }, [word, discardTake]);
  useEffect(() => () => clearTimers(), []);

  // Leaving the result screen (Try again / Watch again) drops the previous take.
  useEffect(() => {
    if (mode === "practice" || mode === "learn") discardTake();
  }, [mode, discardTake]);

  // --- Grading ---------------------------------------------------------------------------------------------
  const gradeAndShow = async (clip: ReturnType<typeof buildClip>, startedAt: number, source: "recorded" | "upload") => {
    const targetId = classIdForWord(word);
    const forced = forcedResult();
    if (forced && targetId !== undefined) {
      const g = forcedGrade(forced as Grade["status"], targetId);
      setGrade(g);
      dispatch({ type: "GRADED", result: forced === "error" ? "error" : resultKindOf(g) });
      return;
    }
    if (!models || targetId === undefined || !isClassInModel(models.labels, targetId)) {
      dispatch({ type: "GRADED", result: "error" });
      return;
    }
    try {
      const result = await gradeClip(models, clip, targetId, { alwaysPredict: debug });
      const g = result.grade;
      setGrade(g);
      if (debug) {
        const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
        const f = result.fractions;
        setDebugInfo(
          `top 5: ` +
            result.top5.map((r) => `${wordForClassId(r.classId)} ${(r.prob * 100).toFixed(1)}%`).join(", ") +
            `\nframes with pose ${pct(f.pose)}, any hand ${pct(f.anyHand)} (left ${pct(f.leftHand)}, right ${pct(f.rightHand)})` +
            `\ngrade: ${g.status}${g.reason ? ` (${g.reason})` : ""}, target p=${g.targetProb?.toFixed(3) ?? "—"}`,
        );
        // In-memory only (never uploaded): exported on demand via "Download attempts (JSON)".
        logAttempt(
          buildAttemptRecord({
            source,
            targetWord: word,
            targetClassId: targetId,
            top5: result.top5,
            grade: g,
            fractions: f,
            wordFor: (id) => wordForClassId(id),
            delegate: models.timings.delegate,
            backend: models.timings.backend,
            liveFps: source === "recorded" ? tracker.fps : null,
            clip,
          }),
        );
        console.log("[koala:debug]", { grade: g, top5: result.top5, fractions: f });
      }
      // not_detected: not the user's fault, no attempt. close: neutral (neither miss nor credit).
      if (g.status !== "not_detected" && g.status !== "close") {
        onFeedback?.(word, g.status === "correct"); // correct -> credit; confused/incorrect -> miss
      }
      console.info(
        `[koala:perf] Stop→result ${(performance.now() - startedAt).toFixed(0)} ms (inference ${result.inferenceMs.toFixed(0)} ms), ` +
          `pose ${(result.fractions.pose * 100).toFixed(0)}% anyHand ${(result.fractions.anyHand * 100).toFixed(0)}%`,
        g,
      );
      dispatch({ type: "GRADED", result: resultKindOf(g) });
    } catch (error) {
      console.error("[koala] inference failed", error);
      dispatch({ type: "GRADED", result: "error" });
    }
  };

  /** Uploaded clip: same VIDEO-mode (tracking) extraction as the live camera, then grade. */
  const uploadFile = async (file: File) => {
    if (modeRef.current !== "practice" && modeRef.current !== "result") return;
    discardTake();
    dispatch({ type: "UPLOAD" });
    setRecordedUrl(URL.createObjectURL(file));
    const startedAt = performance.now();
    try {
      const forced = forcedResult();
      if (forced) {
        await gradeAndShow(buildClip([]), startedAt, "upload");
        return;
      }
      // Fresh instance per upload so tracking state from earlier clips can't leak in.
      const landmarker = await createHolisticLandmarker("VIDEO");
      const { video, revoke } = await loadVideoElement(file);
      try {
        await gradeAndShow(await extractVideoMode(video, landmarker), startedAt, "upload");
      } finally {
        landmarker.close();
        revoke();
      }
    } catch (error) {
      console.error("[koala] upload extraction failed", error);
      dispatch({ type: "GRADED", result: "error" });
    }
  };

  // --- Recording -------------------------------------------------------------------------------------------
  const beginRecording = async () => {
    if (modeRef.current !== "practice" && modeRef.current !== "result") return;
    discardTake();
    dispatch({ type: "RECORD" });
    if (!(await startCamera())) {
      dispatch({ type: "CANCEL_COUNTDOWN" });
      return;
    }
    try {
      const stream = videoRef.current?.srcObject as MediaStream;
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];
      recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      recorder.onstop = () => {
        recorderActiveRef.current = false;
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        setRecordedUrl(URL.createObjectURL(blob));
        stopCamera();
      };
      tracker.startRecording();
      let count = COUNTDOWN_SECONDS;
      setCountdown(count);
      countdownTimerRef.current = setInterval(() => {
        count -= 1;
        if (count > 0) {
          setCountdown(count);
          return;
        }
        clearInterval(countdownTimerRef.current!);
        countdownTimerRef.current = null;
        setCountdown(null);
        recorder.start();
        recorderActiveRef.current = true;
        dispatch({ type: "COUNTDOWN_DONE" });
        const t0 = performance.now();
        setElapsedMs(0);
        elapsedTimerRef.current = setInterval(() => setElapsedMs(performance.now() - t0), 200);
      }, 1000);
    } catch {
      console.error("Camera error.");
      dispatch({ type: "CANCEL_COUNTDOWN" });
    }
  };

  const cancelCountdown = () => {
    if (modeRef.current !== "countdown") return;
    clearTimers();
    setCountdown(null);
    mediaRecorderRef.current = null;
    dispatch({ type: "CANCEL_COUNTDOWN" });
  };

  const stopRecording = () => {
    if (modeRef.current !== "recording") return;
    const startedAt = performance.now();
    clearTimers();
    dispatch({ type: "STOP" });
    // Freeze the landmark buffer first, then stop the recorder (its onstop provides the playback clip).
    const frames = tracker.takeRecording();
    mediaRecorderRef.current?.stop();
    const clip = buildClip(frames);
    // Yield once so the "Analyzing" state paints before the (synchronous) preprocessing/inference work.
    setTimeout(() => void gradeAndShow(clip, startedAt, "recorded"), 0);
  };

  const downloadAttempts = () => {
    const blob = new Blob([attemptsToJson()], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `koala-attempts-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return {
    videoRef,
    canvasRef,
    modelStatus,
    cameraOn,
    cameraError,
    tracker,
    showTracking,
    setShowTracking,
    countdown,
    elapsedMs,
    grade,
    feedbackText: grade ? feedbackMessage(grade, (id) => wordForClassId(id)) : "",
    recordedUrl,
    beginRecording,
    cancelCountdown,
    stopRecording,
    uploadFile,
    debug,
    debugInfo,
    loggedAttempts,
    downloadAttempts,
    clearAttempts,
  };
}

export type PracticeSession = ReturnType<typeof usePracticeSession>;
