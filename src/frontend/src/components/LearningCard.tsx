import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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
import { buildClip } from "@/lib/inference/preprocess";
import { classIdToWord, isClassInModel, wordToClassId } from "@/lib/inference/labels";
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
  | "incorrect"
  | "processing"
  | "not_detected"
  | "error";

interface VideoFile {
  blob: Blob;
  url: string;
}

export const getWordToIdMap = () => ({
  hi: "1",
  what: "2",
  meet: "3",
  "bi bim rice": "4",
  glad: "5",
  hobby: "6",
  me: "7",
  movie: "8",
  face: "9",
  see: "10",
  name: "11",
  read: "12",
  thank: "13",
  equal: "14",
  sorry: "15",
  eat: "16",
  fine: "17",
  "do effort": "18",
  next: "19",
  age: "20",
  again: "21",
  "how many": "22",
  day: "23",
  "good, nice": "24",
  when: "25",
  we: "26",
  subway: "27",
  "be friendly": "28",
  bus: "29",
  ride: "30",
  "cell phone": "31",
  where: "32",
  number: "33",
  location: "34",
  guide: "35",
  responsibility: "36",
  who: "37",
  arrive: "38",
  family: "39",
  time: "40",
  introduction: "41",
  receive: "42",
  "please?": "43",
  walk: "44",
  parents: "45",
  "10 minutes": "46",
  sister: "47",
  study: "48",
  human: "49",
  now: "50",
  special: "51",
  yesterday: "52",
  education: "53",
  test: "54",
  end: "55",
  you: "56",
  worried_about: "57",
  marry: "58",
  effort: "59",
  no: "60",
  sweat: "61",
  yet: "62",
  finally: "63",
  born: "64",
  success: "65",
  favor: "66",
  Seoul: "67",
  dinner: "68",
  experience: "69",
  invite: "70",
  food: "71",
  want: "72",
  visit: "73",
  "one hour": "74",
  far: "75",
  good: "76",
  care: "77",
});

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
  const WORD_TO_ID_MAP = getWordToIdMap();
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isMirrored, setIsMirrored] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [models, setModels] = useState<LoadedModels | null>(null);
  const [modelStatus, setModelStatus] = useState<"loading" | "ready" | "error">("loading");
  const [grade, setGrade] = useState<Grade | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [clipSource, setClipSource] = useState<"recorded" | "upload" | null>(null);
  const showPerf = import.meta.env.DEV || new URLSearchParams(location.search).has("perf");

  const tracker = useLandmarkTracker({
    videoRef,
    canvasRef,
    landmarker: models?.landmarker ?? null,
    active: cameraOn && !videoFile,
    recording: isRecording,
  });

  const ID_TO_WORD_MAP: { [id: string]: string } = Object.fromEntries(
    Object.entries(WORD_TO_ID_MAP).map(([word, id]) => [id, word])
  );

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
    setClipSource(null);
  };

  useEffect(() => {
    resetState();
    startCamera();
  }, [word]);

  /** Grades a sampled raw clip against the current word; updates feedback + spaced repetition. */
  const gradeAndShow = async (clip: ReturnType<typeof buildClip>, startedAt: number) => {
    const targetId = wordToClassId(WORD_TO_ID_MAP, word.toLowerCase());
    if (!models || targetId === undefined || !isClassInModel(models.labels, targetId)) {
      setFeedback("error");
      return;
    }
    try {
      const result = await gradeClip(models, clip, targetId);
      const g = result.grade;
      setGrade(g);
      // not_detected: not the user's fault, no attempt. close: neutral (neither miss nor credit).
      if (g.status === "not_detected") setFeedback("not_detected");
      else if (g.status === "close") setFeedback("close");
      else {
        setFeedback(g.status);
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
        await gradeAndShow(clip, startedAt);
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
    setTimeout(() => void gradeAndShow(clip, startedAt), 0);
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

  return (
    <Card className="w-full border-0 shadow-lg bg-white dark:bg-gray-900 rounded-2xl overflow-hidden">
      <CardContent className="p-6 space-y-6">
        <div className="relative aspect-video bg-black rounded-xl overflow-hidden">
          <video
            ref={videoRef}
            key={videoFile?.url}
            src={videoFile?.url}
            autoPlay={!videoFile || isPlaying}
            muted={!videoFile}
            playsInline
            className={`w-full h-full object-cover ${
              isMirrored && !videoFile ? "scale-x-[-1]" : ""
            }`}
          />

          {/* Skeleton overlay: mirrored together with the camera preview */}
          <canvas
            ref={canvasRef}
            className={`absolute inset-0 w-full h-full pointer-events-none ${
              isMirrored && !videoFile ? "scale-x-[-1]" : ""
            }`}
          />

          {/* Live framing hint, before and during recording */}
          {cameraOn && !videoFile && modelStatus === "ready" && feedback === "idle" && countdown === null && (
            <div className="absolute top-3 inset-x-3 flex justify-center pointer-events-none">
              <div
                role="status"
                aria-live="polite"
                className={`px-4 py-2 rounded-full text-sm font-medium shadow-md backdrop-blur-sm flex items-center gap-2 ${
                  tracker.hint
                    ? "bg-amber-100/95 text-amber-900"
                    : tracker.framingOk
                      ? "bg-green-100/95 text-green-900"
                      : "bg-white/90 text-gray-700"
                }`}
              >
                {tracker.hint ? (
                  <Lightbulb className="w-4 h-4 shrink-0" />
                ) : (
                  <Sparkles className="w-4 h-4 shrink-0" />
                )}
                {tracker.hint ??
                  (tracker.framingOk
                    ? isRecording
                      ? "Looking good — keep signing."
                      : "You're all set — press Start Recording."
                    : "Get set: both shoulders and a hand in view.")}
              </div>
            </div>
          )}

          {showPerf && tracker.fps > 0 && (
            <div className="absolute bottom-4 left-4 px-2 py-1 rounded bg-black/60 text-white text-xs font-mono">
              {tracker.fps.toFixed(1)} fps
            </div>
          )}

          {modelStatus !== "ready" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm text-center px-6">
              {modelStatus === "loading" ? (
                <>
                  <Loader className="w-12 h-12 text-white animate-spin mb-3" />
                  <span className="text-white text-lg font-medium">Getting the sign checker ready…</span>
                  <span className="text-white/80 text-sm mt-1">First time only — this loads right in your browser.</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-12 h-12 text-yellow-400 mb-3" />
                  <span className="text-white text-lg font-medium">Couldn't load the sign checker.</span>
                  <span className="text-white/80 text-sm mt-1">Check your connection and refresh the page.</span>
                </>
              )}
            </div>
          )}

          {countdown !== null && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm">
              <div className="text-white text-8xl md:text-9xl font-bold animate-pulse">
                {countdown}
              </div>
            </div>
          )}

          {feedback === "processing" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm">
              <Loader className="w-16 h-16 text-white animate-spin mb-4" />
              <span className="text-white text-lg font-medium">
                Analyzing your sign...
              </span>
            </div>
          )}

          {feedback === "incorrect" && (
            <div className="absolute inset-0 flex items-center justify-center bg-red-500/20 backdrop-blur-sm">
              <div className="bg-white dark:bg-gray-900 px-8 py-6 rounded-2xl shadow-2xl flex items-center gap-4">
                <XCircle className="w-12 h-12 text-red-600" />
                <span className="text-2xl font-semibold text-gray-900 dark:text-white">
                  {grade?.top1 !== undefined
                    ? `That looked like “${classIdToWord(WORD_TO_ID_MAP, grade.top1)}”. Try again!`
                    : "Try again!"}
                </span>
              </div>
            </div>
          )}

          {feedback === "correct" && (
            <div className="absolute inset-0 flex items-center justify-center bg-green-500/20 backdrop-blur-sm">
              <div className="bg-white dark:bg-gray-900 px-8 py-6 rounded-2xl shadow-2xl flex items-center gap-4">
                <CheckCircle className="w-12 h-12 text-green-600" />
                <span className="text-2xl font-semibold text-gray-900 dark:text-white">
                  Perfect!
                </span>
              </div>
            </div>
          )}

          {feedback === "close" && (
            <div className="absolute inset-0 flex items-center justify-center bg-amber-500/20 backdrop-blur-sm">
              <div className="bg-white dark:bg-gray-900 px-8 py-6 rounded-2xl shadow-2xl flex items-center gap-4 max-w-sm text-center">
                <CircleAlert className="w-12 h-12 text-amber-500 shrink-0" />
                <span className="text-xl font-semibold text-gray-900 dark:text-white">
                  Almost — it looked a bit like “
                  {grade?.top1 !== undefined ? classIdToWord(WORD_TO_ID_MAP, grade.top1) : "another sign"}”.
                </span>
              </div>
            </div>
          )}

          {feedback === "not_detected" && (
            <div className="absolute inset-0 flex items-center justify-center bg-yellow-500/20 backdrop-blur-sm">
              <div className="bg-white dark:bg-gray-900 px-8 py-6 rounded-2xl shadow-2xl flex items-center gap-4 max-w-sm text-center">
                <AlertTriangle className="w-12 h-12 text-yellow-600 shrink-0" />
                <span className="text-lg font-semibold text-gray-900 dark:text-white">
                  Make sure both hands and shoulders are in frame, then try
                  again.
                </span>
              </div>
            </div>
          )}

          {feedback === "error" && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-500/20 backdrop-blur-sm">
              <div className="bg-white dark:bg-gray-900 px-8 py-6 rounded-2xl shadow-2xl flex items-center gap-4 max-w-sm text-center">
                <AlertTriangle className="w-12 h-12 text-gray-600 shrink-0" />
                <span className="text-lg font-semibold text-gray-900 dark:text-white">
                  Something went wrong checking that clip — this isn't your signing. Please try again.
                </span>
              </div>
            </div>
          )}

          <button
            onClick={() => setIsMirrored(!isMirrored)}
            className="absolute bottom-4 right-4 p-3 rounded-full bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm shadow-lg hover:scale-110 transition-transform"
            aria-label="Toggle mirror"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className={`w-5 h-5 transition-transform ${
                isMirrored ? "scale-x-[-1]" : ""
              }`}
            >
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M2.14935 19.5257C2.33156 19.8205 2.65342 20 3 20H10C10.5523 20 11 19.5523 11 19V4.99998C11 4.5362 10.6811 4.13328 10.2298 4.02673C9.77838 3.92017 9.31298 4.13795 9.10557 4.55276L2.10557 18.5528C1.95058 18.8628 1.96714 19.2309 2.14935 19.5257ZM4.61804 18L9 9.23604V18H4.61804ZM13 19C13 19.5523 13.4477 20 14 20H21C21.3466 20 21.6684 19.8205 21.8507 19.5257C22.0329 19.2309 22.0494 18.8628 21.8944 18.5528L14.8944 4.55276C14.687 4.13795 14.2216 3.92017 13.7702 4.02673C13.3189 4.13328 13 4.5362 13 4.99998V19Z"
                fill="currentColor"
                className="text-gray-700 dark:text-gray-300"
              />
            </svg>
          </button>
        </div>

        <div className="space-y-3">
          {!isRecording && !isReadyToSubmit && (
            <div className="flex flex-col sm:flex-row gap-3">
              <Button
                disabled={modelStatus !== "ready"}
                onClick={() => fileInputRef.current?.click()}
                className="flex-1 h-12 text-base font-medium rounded-full bg-[#5e877a] text-white hover:bg-[#5e877a] hover:brightness-110 transition-all"
                variant="outline"
              >
                <Upload className="mr-2 h-5 w-5" />
                Upload Video
              </Button>
              <Button
                disabled={modelStatus !== "ready"}
                onClick={startRecording}
                className="flex-1 h-12 text-base font-medium rounded-full hover:opacity-90 transition-opacity"
              >
                <Camera className="mr-2 h-5 w-5" />
                Start Recording
              </Button>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileUpload}
            className="hidden"
          />

          {isRecording && (
            <Button
              onClick={stopRecording}
              className="w-full h-12 text-base font-medium rounded-full bg-red-600 hover:bg-red-700 text-white transition-colors"
            >
              <StopCircle className="mr-2 h-5 w-5" />
              Stop Recording
            </Button>
          )}

          {isReadyToSubmit && feedback !== "processing" && (
            <div className="flex flex-col sm:flex-row gap-3">
              <Button
                onClick={handlePlaybackToggle}
                variant="outline"
                className="flex-1 h-12 text-base font-medium rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                {isPlaying ? (
                  <Pause className="mr-2 h-5 w-5" />
                ) : (
                  <Play className="mr-2 h-5 w-5" />
                )}
                {isPlaying ? "Pause" : "Replay"}
              </Button>

              <Button
                onClick={() => {
                  setVideoFile(null);
                  setFeedback("idle");
                  setGrade(null);
                  setClipSource(null);
                  setIsReadyToSubmit(false);
                  startCamera();
                }}
                variant="outline"
                className="flex-1 h-12 text-base font-medium rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <Camera className="mr-2 h-5 w-5" />
                Re-record
              </Button>

              {clipSource === "upload" && (feedback === "idle" || feedback === "error") && (
                <Button
                  onClick={() => videoFile && runUploadInference(videoFile.blob)}
                  className="flex-1 h-12 text-base font-medium rounded-full hover:opacity-90 transition-opacity"
                >
                  <CheckCircle className="mr-2 h-5 w-5" />
                  Submit
                </Button>
              )}
            </div>
          )}

          {feedback === "processing" && (
            <Button
              disabled
              className="w-full h-12 text-base font-medium rounded-full bg-gray-400 cursor-not-allowed"
            >
              <Loader className="mr-2 h-5 w-5 animate-spin" />
              Analyzing...
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
