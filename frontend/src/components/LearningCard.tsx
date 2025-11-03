import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Camera, StopCircle, CheckCircle, XCircle, Upload, Play, Pause, Loader } from "lucide-react";
import { toast } from "sonner";
import { useIsMobile } from "@/hooks/use-mobile";

interface LearningCardProps {
  word: string;
  onNext: () => void;
  onPrevious: () => void;
}

type FeedbackState = "idle" | "correct" | "incorrect" | "processing";

interface VideoFile {
  blob: Blob;
  url: string;
}

// Map word string to class ID
export const getWordToIdMap = () => ({
  "hi": "1",
  "meet": "3",
  "me": "7",
  "see": "10",
  "name": "11",
  "thank": "13",
  "equal": "14",
  "sorry": "15",
  "age": "20",
  //"how many": "22",
  "day": "23",
  "please?": "43",
  "study": "48",
  "human": "49",
  "now": "50",
  "test": "54",
  "yet": "62",
  "finally": "63",
  "dinner": "68",
  "experience": "69",
  "invite": "70",
  "food": "71",
  "want": "72",
  "good": "76",
  "care": "77",
});

export const LearningCard = ({ word, onNext, onPrevious }: LearningCardProps) => {
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
  const isMobile = useIsMobile();
  const [isMirrored, setIsMirrored] = useState(true); // default mirrored


  const ID_TO_WORD_MAP: { [id: string]: string } = Object.fromEntries(
    Object.entries(WORD_TO_ID_MAP).map(([word, id]) => [id, word])
  );

  // Start camera on mount
  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  // Wake up backend on mount
  useEffect(() => {
    const wakeUpBackend = async () => {
      try {
        const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
        await fetch(`${API_URL}/`);
        console.log('Backend is ready');
      } catch {
        console.log('Waking up backend...');
      }
    };
    wakeUpBackend();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch {
      toast.error("Could not access camera.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
    }
  };

  // Reset all state
  const resetState = () => {
    setFeedback("idle");
    setVideoFile(null);
    setIsRecording(false);
    setIsReadyToSubmit(false);
    setCountdown(null);
  };

  // Reset when word changes
  useEffect(() => {
    resetState();
    startCamera();
  }, [word]);

  const runInference = async (videoBlob: Blob) => {
    setFeedback("processing");
    toast.info("Sending video for analysis...");

    const expectedClassLabel = WORD_TO_ID_MAP[word.toLowerCase()];
    if (!expectedClassLabel) {
      setFeedback("incorrect");
      toast.error(`Word "${word}" is not mapped to a class ID.`);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("video", videoBlob, "sign_video.webm");

      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
        credentials: "include",
      });

      clearTimeout(timeoutId);

      if (!response.ok) throw new Error(`Request failed: ${response.status}`);

      const result = await response.json();
      if (!result.success) throw new Error(result.error || "Prediction failed.");

      const predictedClassLabel = String(result.predicted_class);
      const isCorrect = predictedClassLabel === expectedClassLabel;
      const predictedWord = ID_TO_WORD_MAP[predictedClassLabel] || "Unknown";
      const expectedWord = ID_TO_WORD_MAP[expectedClassLabel] || "Unknown";

      setFeedback(isCorrect ? "correct" : "incorrect");
      toast[isCorrect ? "success" : "error"](
        isCorrect
          ? `Great job! You signed "${predictedWord}" correctly!`
          : `Incorrect. Model predicted "${predictedWord}", expected "${expectedWord}".`
      );
    } catch (error: any) {
      console.error("Inference error:", error);
      setFeedback("incorrect");
      toast.error(error.name === 'AbortError' ? "Request timed out." : "Error during prediction.");
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    resetState();
    setVideoFile({ blob: file, url: URL.createObjectURL(file) });
    setIsReadyToSubmit(true);
    toast.info("Video uploaded. Review and submit.");
  };

  const startRecording = async () => {
    resetState();
    setIsReadyToSubmit(false);
    await startCamera();

    try {
      const stream = videoRef.current?.srcObject as MediaStream;
      if (!stream) throw new Error("Camera stream unavailable");

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        setVideoFile({ blob, url: URL.createObjectURL(blob) });
        setIsRecording(false);
        setIsReadyToSubmit(true);
        stopCamera();
        toast.info("Recording complete. Review or submit.");
      };

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
          toast.info("Recording started!");
        }
      }, 1000);
    } catch {
      toast.error("Could not access camera.");
      resetState();
    }
  };

  const stopRecording = () => mediaRecorderRef.current?.stop();

  const handleRetry = async () => {
    resetState();
    await startCamera();
    toast.info("Camera ready for re-recording!");
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

  const isPredicting = feedback === "processing";
  const isIdle = feedback === "idle" && !isRecording;

  return (
    <Card className="w-full max-w-2xl shadow-xl bg-[hsl(var(--card))] text-[hsl(var(--card-foreground))]">
      <CardContent className="pt-6 space-y-6">
        {/* VIDEO AREA */}
        <div className="relative aspect-video bg-[hsl(var(--muted))] border-2 border-dashed border-[hsl(var(--border))] rounded-xl overflow-hidden shadow-inner">
  <video
    ref={videoRef}
    autoPlay={!videoFile}
    muted={!videoFile}
    playsInline
    src={videoFile ? videoFile.url : undefined}
    className={`w-full h-full object-cover ${isMirrored ? "scale-x-[-1]" : ""}`}
  />

  {/* Mirror Button Overlay */}
  <button
    onClick={() => setIsMirrored(!isMirrored)}
    className="absolute bottom-4 right-4 p-2 w-10 h-10 flex items-center justify-center rounded-full bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] shadow-lg transition-transform hover:scale-110"
  >
    <svg
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`w-6 h-6 transition-transform ${isMirrored ? "scale-x-[-1]" : ""}`}
    >
      <g id="SVGRepo_bgCarrier" strokeWidth="0"></g>
      <g id="SVGRepo_tracerCarrier" strokeLinecap="round" strokeLinejoin="round"></g>
      <g id="SVGRepo_iconCarrier">
        <path
          fillRule="evenodd"
          clipRule="evenodd"
          d="M2.14935 19.5257C2.33156 19.8205 2.65342 20 3 20H10C10.5523 20 11 19.5523 11 19V4.99998C11 4.5362 10.6811 4.13328 10.2298 4.02673C9.77838 3.92017 9.31298 4.13795 9.10557 4.55276L2.10557 18.5528C1.95058 18.8628 1.96714 19.2309 2.14935 19.5257ZM4.61804 18L9 9.23604V18H4.61804ZM13 19C13 19.5523 13.4477 20 14 20H21C21.3466 20 21.6684 19.8205 21.8507 19.5257C22.0329 19.2309 22.0494 18.8628 21.8944 18.5528L14.8944 4.55276C14.687 4.13795 14.2216 3.92017 13.7702 4.02673C13.3189 4.13328 13 4.5362 13 4.99998V19Z"
          fill="#ffffff"
        ></path>
      </g>
    </svg>
  </button>
</div>


        {/* CONTROLS */}
        <div className="space-y-4">
          {(isIdle || feedback === "incorrect") && !isReadyToSubmit && (
            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                onClick={() => fileInputRef.current?.click()}
                disabled={isPredicting}
                className="flex-1 text-lg py-6 bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:brightness-90 transition flex items-center justify-center gap-2"
                >
                <Upload className="h-5 w-5" /> Upload Video
              </Button>
              <Button
                onClick={startRecording}
                disabled={isPredicting}
                className="flex-1 text-lg py-6 bg-[hsl(var(--accent))] text-[hsl(var(--accent-foreground))] hover:bg-[hsl(190,29%,28%)] transition flex items-center justify-center gap-2"
                >
                <Camera className="h-5 w-5" /> Start Recording
              </Button>
            </div>
          )}

          <input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileUpload} className="hidden" />

          <div className="flex flex-wrap justify-center gap-4 mt-4">
            {isRecording && (
              <Button size="lg" onClick={stopRecording} className="text-lg px-8 py-6 bg-[hsl(var(--destructive))] text-[hsl(var(--destructive-foreground))]">
                <StopCircle className="h-5 w-5" /> Stop Recording
              </Button>
            )}

            {isReadyToSubmit && (
              <>
                <Button
                  size="lg"
                  onClick={handlePlaybackToggle}
                  variant="outline"
                  className="text-lg px-8 py-6 flex items-center justify-center gap-2"
                >
                  {isPlaying ? <><Pause className="h-5 w-5" /> Pause</> : <><Play className="h-5 w-5" /> Replay</>}
                </Button>

                <Button
                  size="lg"
                  onClick={handleRetry}
                  variant="outline"
                  className="text-lg px-8 py-6 flex items-center justify-center gap-2"
                >
                  <Camera className="h-5 w-5" /> Re-record
                </Button>

                <Button
                  size="lg"
                  onClick={() => videoFile && runInference(videoFile.blob)}
                  className="text-lg px-8 py-6 bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))]"
                >
                  <CheckCircle className="h-5 w-5" /> Submit
                </Button>
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
