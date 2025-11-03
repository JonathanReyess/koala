import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Camera, StopCircle, CheckCircle, XCircle, Upload, Play, Loader } from "lucide-react";
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
  "how many": "22",
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
  const WORD_TO_ID_MAP = getWordToIdMap();
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isMobile = useIsMobile();

  const ID_TO_WORD_MAP: { [id: string]: string } = Object.fromEntries(
    Object.entries(WORD_TO_ID_MAP).map(([word, id]) => [id, word])
  );

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

  // Reset all state
  const resetState = () => {
    setFeedback("idle");
    setVideoFile(null);
    setIsRecording(false);
    setIsReadyToSubmit(false);
    setCountdown(null);

    if (videoRef.current) {
      videoRef.current.src = "";
      if (videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
      videoRef.current.srcObject = null;
    }
  };

  // Reset when word changes
  useEffect(() => {
    resetState();
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
  
      // --- LOGGING FOR DEBUGGING ---
      console.log("Raw model output:", result);
  
      if (!result.success) throw new Error(result.error || "Prediction failed.");
  
      const predictedClassLabel = String(result.predicted_class);
      const isCorrect = predictedClassLabel === expectedClassLabel;
      const predictedWord = ID_TO_WORD_MAP[predictedClassLabel] || "Unknown";
      const expectedWord = ID_TO_WORD_MAP[expectedClassLabel] || "Unknown";
  
      console.log(`Expected: ${expectedWord} (${expectedClassLabel}), Predicted: ${predictedWord} (${predictedClassLabel})`);
  
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
  

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
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

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (videoRef.current) videoRef.current.srcObject = stream;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        setVideoFile({ blob, url: URL.createObjectURL(blob) });
        setIsRecording(false);
        setIsReadyToSubmit(true);
        stream.getTracks().forEach(track => track.stop());
        if (videoRef.current) videoRef.current.srcObject = null;
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
  const handleRetry = () => resetState();

  const isPredicting = feedback === "processing";
  const isVideoReady = !!videoFile && !isRecording && !isPredicting;
  const isIdle = feedback === "idle" && !isRecording;

  return (
    <Card className="w-full max-w-2xl shadow-xl bg-[hsl(var(--card))] text-[hsl(var(--card-foreground))]">
      <CardContent className="pt-6 space-y-6">
        {/* Video display */}
        <div className="relative aspect-video bg-[hsl(var(--muted))] border-2 border-dashed border-[hsl(var(--border))] rounded-xl overflow-hidden shadow-inner">
          <video
            ref={videoRef}
            autoPlay={!videoFile?.url}
            muted
            playsInline
            controls={isVideoReady}
            src={videoFile?.url || undefined}
            className="w-full h-full object-cover"
          />

          {countdown !== null && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-50">
              <span className="text-white text-6xl font-bold animate-pulse">{countdown}</span>
            </div>
          )}

          {isIdle && (
            <div className="absolute inset-0 flex items-center justify-center bg-[hsl(var(--muted))/70]">
              <Camera className="h-16 w-16 text-[hsl(var(--muted-foreground))]" />
            </div>
          )}

          {isRecording && (
            <div className="absolute top-4 left-4 flex items-center px-3 py-1 bg-[hsl(var(--destructive))] rounded-full shadow-lg">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
              </span>
              <span className="ml-2 text-sm font-semibold text-[hsl(var(--destructive-foreground))]">Recording...</span>
            </div>
          )}

          {feedback === "processing" && (
            <div className="absolute inset-0 bg-gray-800/90 flex flex-col items-center justify-center animate-pulse">
              <Loader className="h-10 w-10 text-white animate-spin mb-3" />
              <p className="text-white text-xl font-semibold">Analyzing Sign...</p>
            </div>
          )}

          {feedback === "correct" && (
            <div className="absolute inset-0 bg-[hsl(var(--success))/90] flex items-center justify-center animate-in fade-in duration-500">
              <CheckCircle className="h-24 w-24 text-[hsl(var(--success-foreground))]" />
            </div>
          )}

          {feedback === "incorrect" && (
            <div className="absolute inset-0 bg-[hsl(var(--destructive))/90] flex items-center justify-center animate-in fade-in duration-500">
              <XCircle className="h-24 w-24 text-[hsl(var(--destructive-foreground))]" />
            </div>
          )}
        </div>

        {/* Controls */}
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
              <Button size="lg" onClick={stopRecording} className="text-lg px-8 py-6 bg-[hsl(var(--destructive))] text-[hsl(var(--destructive-foreground))] shadow-xl flex items-center justify-center gap-2">
                <StopCircle className="h-5 w-5" /> Stop Recording
              </Button>
            )}

            {isReadyToSubmit && (
              <>
                <Button size="lg" onClick={handleRetry} variant="outline" className="text-lg px-8 py-6 border-[hsl(var(--border))] text-[hsl(var(--foreground))] hover:bg-[hsl(var(--muted))] flex items-center justify-center gap-2">
                  <Camera className="h-5 w-5" /> Re-record
                </Button>
                <Button size="lg" onClick={() => videoFile && runInference(videoFile.blob)} className="text-lg px-8 py-6 bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] shadow-xl hover:brightness-90 flex items-center justify-center gap-2">
                  <Play className="h-5 w-5" /> Submit
                </Button>
              </>
            )}

            {feedback === "incorrect" && (
              <Button size="lg" onClick={handleRetry} variant="outline" className="text-lg px-8 py-6 border-[hsl(var(--destructive))] text-[hsl(var(--destructive))] hover:bg-[hsl(var(--muted))] flex items-center justify-center gap-2">
                <XCircle className="h-5 w-5" /> Try Again
              </Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
