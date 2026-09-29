import { useState, useRef, useEffect, useSyncExternalStore } from "react";
import { SurfaceCard, MediaWell, MediaScrim, PillButton, PillButtonRow, OverlayChip, OverlayToggle, OverlayIconButton } from "@/components/ds";
import {
  Camera,
  StopCircle,
  CheckCircle,
  XCircle,
  Upload,
  Play,
  Pause,
  Loader,
  AlertTriangle,
  Lightbulb,
  Sparkles,
  CircleAlert,
} from "lucide-react";
import { loadModels, createHolisticLandmarker, LoadedModels } from "@/lib/inference/loader";
import { gradeClip } from "@/lib/inference/pipeline";
import { Grade } from "@/lib/inference/grading";
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
import { classIdForWord, wordForClassId } from "@/data/words";
import { extractVideoMode, loadVideoElement } from "@/lib/inference/extract";
import { useLandmarkTracker } from "@/hooks/useLandmarkTracker";

interface LearningCardProps {
  word: string;
  onNext: () => void;
  onPrevious: () => void;
  onFeedback?: (word: string, correct: boolean) => void;
}

type FeedbackState =
  | "idle"
  | "correct"
  | "close"
  | "confused"
  | "incorrect"
  | "processing"
  | "not_detected"
  | "error";

interface VideoFile {
  blob: Blob;
  url: string;
}

export const LearningCard = ({
  word,
  onNext,
  onPrevious,
  onFeedback,
}: LearningCardProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [feedback, setFeedback] = useState<FeedbackState>("idle");
  const [videoFile, setVideoFile] = useState<VideoFile | null>(null);
  const [isReadyToSubmit, setIsReadyToSubmit] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Display-only: toggles a CSS flip on the preview (and skeleton canvas). MediaPipe and MediaRecorder read the
  // unflipped camera pixels, so landmarks are identical either way (verified: nose x unchanged when toggling).
  const [isMirrored, setIsMirrored] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [models, setModels] = useState<LoadedModels | null>(null);
  const [modelStatus, setModelStatus] = useState<"loading" | "ready" | "error">("loading");
  const [grade, setGrade] = useState<Grade | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [clipSource, setClipSource] = useState<"recorded" | "upload" | null>(null);
  const debug = new URLSearchParams(location.search).get("debug") === "1";
  const [debugInfo, setDebugInfo] = useState<string | null>(null);
  const loggedAttempts = useSyncExternalStore(subscribeAttempts, attemptCount);
  // Skeleton overlay is opt-in; the choice is remembered per browser.
  const [showTracking, setShowTracking] = useState<boolean>(() => {
    try {
      return localStorage.getItem("koala.showTracking") === "1";
    } catch {
      return false;
    }
  });
  const toggleTracking = (on: boolean) => {
    setShowTracking(on);
    try {
      localStorage.setItem("koala.showTracking", on ? "1" : "0");
    } catch {
      /* storage unavailable: preference just isn't remembered */
    }
  };
  const showPerf = import.meta.env.DEV || new URLSearchParams(location.search).has("perf");

  const tracker = useLandmarkTracker({
    videoRef,
    canvasRef,
    landmarker: models?.landmarker ?? null,
    active: cameraOn && !videoFile,
    recording: isRecording,
    drawOverlay: showTracking,
  });

  const downloadAttempts = () => {
    const blob = new Blob([attemptsToJson()], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `koala-attempts-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const feedbackText = grade ? feedbackMessage(grade, (id) => wordForClassId(id)) : "";

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  // Lazy-load the ONNX model + landmarker on entering Practice; cached for later visits.
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

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
    } catch {
      console.error("Could not access camera.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
    }
    setCameraOn(false);
  };

  const resetState = () => {
    setFeedback("idle");
    setVideoFile(null);
    setIsRecording(false);
    setIsReadyToSubmit(false);
    setCountdown(null);
    setGrade(null);
    setDebugInfo(null);
    setClipSource(null);
  };

  useEffect(() => {
    resetState();
    startCamera();
  }, [word]);

  /** Grades a sampled raw clip against the current word; updates feedback + spaced repetition. */
  const gradeAndShow = async (
    clip: ReturnType<typeof buildClip>,
    startedAt: number,
    source: "recorded" | "upload",
  ) => {
    const targetId = classIdForWord(word);
    if (!models || targetId === undefined || !isClassInModel(models.labels, targetId)) {
      setFeedback("error");
      return;
    }
    try {
      const result = await gradeClip(models, clip, targetId, { alwaysPredict: debug });
      const g = result.grade;
      setGrade(g);
      if (debug) {
        const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
        const f = result.fractions;
        const text =
          `top 5: ` +
          result.top5.map((r) => `${wordForClassId(r.classId)} ${(r.prob * 100).toFixed(1)}%`).join(", ") +
          `\nframes with pose ${pct(f.pose)}, any hand ${pct(f.anyHand)} (left ${pct(f.leftHand)}, right ${pct(f.rightHand)})` +
          `\ngrade: ${g.status}${g.reason ? ` (${g.reason})` : ""}, target p=${g.targetProb?.toFixed(3) ?? "—"}`;
        setDebugInfo(text);
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
      if (g.status === "not_detected") setFeedback("not_detected");
      else if (g.status === "close") setFeedback("close");
      else {
        setFeedback(g.status);
        // correct -> credit; confused/incorrect -> miss.
        if (onFeedback) onFeedback(word, g.status === "correct");
      }
      const total = performance.now() - startedAt;
      console.info(
        `[koala:perf] Stop→result ${total.toFixed(0)} ms (inference ${result.inferenceMs.toFixed(0)} ms), ` +
          `pose ${(result.fractions.pose * 100).toFixed(0)}% anyHand ${(result.fractions.anyHand * 100).toFixed(0)}%`,
        g,
      );
    } catch (error) {
      console.error("[koala] inference failed", error);
      setFeedback("error");
    }
  };

  /**
   * Uploaded clip: same VIDEO-mode (tracking) extraction as the live camera, then grade. (On the 98 bundled
   * example clips VIDEO mode was 98/98 top-1 vs 81/98 for IMAGE mode — see PERF.md / the parity page.)
   */
  const runUploadInference = async (videoBlob: Blob) => {
    setFeedback("processing");
    const startedAt = performance.now();
    try {
      // Fresh instance per upload so tracking state from earlier clips can't leak in.
      const landmarker = await createHolisticLandmarker("VIDEO");
      const { video, revoke } = await loadVideoElement(videoBlob);
      try {
        const clip = await extractVideoMode(video, landmarker);
        await gradeAndShow(clip, startedAt, "upload");
      } finally {
        landmarker.close();
        revoke();
      }
    } catch (error) {
      console.error("[koala] upload extraction failed", error);
      setFeedback("error");
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    resetState();
    stopCamera();
    setVideoFile({ blob: file, url: URL.createObjectURL(file) });
    setClipSource("upload");
    setIsReadyToSubmit(true);
  };

  const startRecording = async () => {
    resetState();
    setIsReadyToSubmit(false);
    await startCamera();

    try {
      const stream = videoRef.current?.srcObject as MediaStream;
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        setVideoFile({ blob, url: URL.createObjectURL(blob) });
        setIsRecording(false);
        setClipSource("recorded");
        setIsReadyToSubmit(true);
        stopCamera();
      };

      tracker.startRecording();
      let count = 3;
      setCountdown(count);
      const interval = setInterval(() => {
        count -= 1;
        if (count > 0) setCountdown(count);
        else {
          clearInterval(interval);
          setCountdown(null);
          mediaRecorder.start();
          setIsRecording(true);
        }
      }, 1000);
    } catch {
      console.error("Camera error.");
    }
  };

  const stopRecording = () => {
    const startedAt = performance.now();
    // Freeze the landmark buffer before anything else changes, then stop the recorder (for replay).
    const frames = tracker.takeRecording();
    mediaRecorderRef.current?.stop();
    setFeedback("processing");
    const clip = buildClip(frames);
    // Yield once so the "Analyzing" state paints before the (synchronous) preprocessing/inference work.
    setTimeout(() => void gradeAndShow(clip, startedAt, "recorded"), 0);
  };

  const handlePlaybackToggle = () => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying) {
      video.pause();
      setIsPlaying(false);
    } else {
      video.play();
      setIsPlaying(true);
      video.onended = () => setIsPlaying(false);
    }
  };

  const showLoadingScrim = modelStatus !== "ready";
  const feedbackScrim = (() => {
    const msg = feedbackText;
    switch (feedback) {
      case "processing":
        return <MediaScrim passive icon={<Loader className="h-12 w-12 animate-spin" />} label="Analyzing your sign..." />;
      case "correct":
        return <MediaScrim passive icon={<CheckCircle className="h-12 w-12" />} label={msg} />;
      case "close":
        return <MediaScrim passive tone="ink" icon={<CircleAlert className="h-12 w-12" />} label={msg} />;
      case "confused":
      case "incorrect":
        return <MediaScrim passive tone="ink" icon={<XCircle className="h-12 w-12" />} label={msg} />;
      case "not_detected":
        return <MediaScrim passive tone="ink" icon={<AlertTriangle className="h-12 w-12" />} label={msg} />;
      case "error":
        return (
          <MediaScrim
            passive
            tone="ink"
            icon={<AlertTriangle className="h-12 w-12" />}
            label="Something went wrong checking that clip, this isn't your signing. Please try again."
          />
        );
      default:
        return null;
    }
  })();

  const scrim = showLoadingScrim ? (
    modelStatus === "loading" ? (
      <MediaScrim
        passive
        icon={<Loader className="h-12 w-12 animate-spin" />}
        label="Getting the sign checker ready…"
      >
        <span className="text-sm opacity-90">First time only, this loads right in your browser.</span>
      </MediaScrim>
    ) : (
      <MediaScrim
        passive
        tone="ink"
        icon={<AlertTriangle className="h-12 w-12" />}
        label="Couldn't load the sign checker."
      >
        <span className="text-sm opacity-90">Check your connection and refresh the page.</span>
      </MediaScrim>
    )
  ) : countdown !== null ? (
    <MediaScrim passive tone="ink" label={<span className="text-8xl font-bold leading-none animate-pulse">{countdown}</span>} />
  ) : (
    feedbackScrim
  );

  const showHint = cameraOn && !videoFile && modelStatus === "ready" && feedback === "idle" && countdown === null;

  const footer = feedback === "processing" ? (
    <PillButton block disabled icon={<Loader className="h-5 w-5 animate-spin" />}>
      Analyzing...
    </PillButton>
  ) : isRecording ? (
    <PillButton block onClick={stopRecording} icon={<StopCircle className="h-5 w-5" />}>
      Stop Recording
    </PillButton>
  ) : isReadyToSubmit ? (
    <PillButtonRow>
      <PillButton variant="secondary" onClick={handlePlaybackToggle} icon={isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}>
        {isPlaying ? "Pause" : "Replay"}
      </PillButton>
      <PillButton
        variant="secondary"
        icon={<Camera className="h-5 w-5" />}
        onClick={() => {
          setVideoFile(null);
          setFeedback("idle");
          setGrade(null);
          setDebugInfo(null);
          setClipSource(null);
          setIsReadyToSubmit(false);
          startCamera();
        }}
      >
        Re-record
      </PillButton>
      {clipSource === "upload" && (feedback === "idle" || feedback === "error") && (
        <PillButton icon={<CheckCircle className="h-5 w-5" />} onClick={() => videoFile && runUploadInference(videoFile.blob)}>
          Submit
        </PillButton>
      )}
    </PillButtonRow>
  ) : (
    <PillButtonRow>
      <PillButton variant="secondary" disabled={modelStatus !== "ready"} onClick={() => fileInputRef.current?.click()} icon={<Upload className="h-5 w-5" />}>
        Upload Video
      </PillButton>
      <PillButton disabled={modelStatus !== "ready"} onClick={startRecording} icon={<Camera className="h-5 w-5" />}>
        Start Recording
      </PillButton>
    </PillButtonRow>
  );

  const mirrorClass = isMirrored && !videoFile ? "scale-x-[-1]" : "";

  return (
    <div className="space-y-4">
      <SurfaceCard footer={footer}>
        <MediaWell
          scrim={scrim}
          topLeft={
            showHint && (
              <OverlayChip
                role="status"
                aria-live="polite"
                tone={tracker.hint ? "default" : tracker.framingOk ? "sage" : "default"}
                icon={tracker.hint ? <Lightbulb className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
              >
                {tracker.hint ??
                  (tracker.framingOk
                    ? isRecording
                      ? "Looking good, keep signing."
                      : "You're all set. Press Start Recording."
                    : "Getting a good look at you…")}
              </OverlayChip>
            )
          }
          topRight={modelStatus === "ready" && !videoFile && <OverlayToggle label="Show tracking" checked={showTracking} onCheckedChange={toggleTracking} />}
          bottomLeft={showPerf && tracker.fps > 0 && <OverlayChip className="font-mono text-xs">{tracker.fps.toFixed(1)} fps</OverlayChip>}
          bottomRight={
            <OverlayIconButton label="Toggle mirror" onClick={() => setIsMirrored(!isMirrored)}>
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 transition-transform ${isMirrored ? "scale-x-[-1]" : ""}`} aria-hidden="true">
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M2.14935 19.5257C2.33156 19.8205 2.65342 20 3 20H10C10.5523 20 11 19.5523 11 19V4.99998C11 4.5362 10.6811 4.13328 10.2298 4.02673C9.77838 3.92017 9.31298 4.13795 9.10557 4.55276L2.10557 18.5528C1.95058 18.8628 1.96714 19.2309 2.14935 19.5257ZM4.61804 18L9 9.23604V18H4.61804ZM13 19C13 19.5523 13.4477 20 14 20H21C21.3466 20 21.6684 19.8205 21.8507 19.5257C22.0329 19.2309 22.0494 18.8628 21.8944 18.5528L14.8944 4.55276C14.687 4.13795 14.2216 3.92017 13.7702 4.02673C13.3189 4.13328 13 4.5362 13 4.99998V19Z"
                  fill="currentColor"
                />
              </svg>
            </OverlayIconButton>
          }
        >
          <video
            ref={videoRef}
            key={videoFile?.url}
            src={videoFile?.url}
            autoPlay={!videoFile || isPlaying}
            muted={!videoFile}
            playsInline
            className={mirrorClass}
          />
          {/* Skeleton overlay: mirrored together with the camera preview */}
          <canvas ref={canvasRef} className={mirrorClass} />
        </MediaWell>
        <input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileUpload} className="hidden" />
      </SurfaceCard>

      {debug && (
        <div className="space-y-2">
          {debugInfo && (
            <pre className="rounded-media bg-sage-50 p-3 text-xs text-ink whitespace-pre-wrap font-mono">{debugInfo}</pre>
          )}
          <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">
            <PillButton variant="secondary" disabled={loggedAttempts === 0} onClick={downloadAttempts}>
              Download attempts (JSON)
            </PillButton>
            <PillButton variant="ghost" disabled={loggedAttempts === 0} onClick={clearAttempts}>
              Clear
            </PillButton>
            <span>{loggedAttempts} attempt(s) logged in memory. Nothing is uploaded.</span>
          </div>
        </div>
      )}
    </div>
  );
};
