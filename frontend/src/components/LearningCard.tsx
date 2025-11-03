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
export const WORD_TO_ID_MAP: { [key: string]: string } = {
  // " ": "0",
  "hi": "1",
  // "what": "2",
  "meat": "3",
  // "bi bim rice": "4",
  // "glad": "5",
  // "hobby": "6",
  "me": "7",
  // "movie": "8",
  // "face": "9",
  "see": "10",
  "name": "11",
  // "read": "12",
  "thank": "13",
  "equal": "14",
  "sorry": "15",
  // "eat": "16",
  // "fine": "17",
  // "do effort": "18",
  // "next": "19",
  "age": "20",
  // "again": "21",
  "how many": "22",
  "day": "23",
  "good, nice": "24",
  // "when": "25",
  // "we": "26",
  // "subway": "27",
  // "be friendly": "28",
  // "bus": "29",
  // "ride": "30",
  // "cell phone": "31",
  // "where": "32",
  "number": "33",
  // "location": "34",
  // "guide": "35",
  // "responsibility": "36",
  // "who": "37",
  "please?": "43",
  // "walk": "44",
  // "parents": "45",
  // "10 minutes": "46",
  // "sister": "47",
  "study": "48",
  "human": "49",
  "now": "50",
  // "special": "51",
  // "yesterday": "52",
  "education": "53",
  "test": "54",
  // "end": "55",
  // "you": "56",
  // "worried_about": "57",
  // "marry": "58",
  // "effort": "59",
  // "no": "60",
  "yet": "62",
  "finally": "63",
  // "born": "64",
  // "success": "65",
  // "favor": "66",
  // "Seoul": "67",
  "dinner": "68",
  "experience": "69",
  "invite": "70",
  "food": "71",
  "want": "72",
  // "visit": "73",
  // "one hour": "74",
  // "far": "75",
  "good": "76",
  "care": "77",
  // skipping empty labels 78-100
};



export const LearningCard = ({ word, onNext, onPrevious }: LearningCardProps) => {
  const [isRecording, setIsRecording] = useState(false);
  const [feedback, setFeedback] = useState<FeedbackState>("idle");
  const [videoFile, setVideoFile] = useState<VideoFile | null>(null);
  const [isReadyToSubmit, setIsReadyToSubmit] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isMobile = useIsMobile();

  // Reset all recording/video/feedback state
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

  // Reset state when word changes (optional, if parent re-renders with new word)
  useEffect(() => {
    resetState();
  }, [word]);

  const runInference = async (videoBlob: Blob) => {
    setFeedback("processing");
    toast.info("Sending video for analysis...");

    const expectedClassLabel = WORD_TO_ID_MAP[word.toLowerCase()];
    if (!expectedClassLabel) {
      setFeedback("incorrect");
      toast.error(`Error: Word "${word}" is not mapped to a class ID.`);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("video", videoBlob, "sign_video.webm");

      
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);

      const result = await response.json();
      if (!result.success) throw new Error(result.error || "Prediction failed.");

      const predictedClassLabel = String(result.predicted_class);
      const isCorrect = predictedClassLabel === expectedClassLabel;

      setFeedback(isCorrect ? "correct" : "incorrect");
      toast[isCorrect ? "success" : "error"](
        isCorrect
          ? "Great job! That's correct!"
          : `Incorrect. Model predicted "${predictedClassLabel}". Expected "${expectedClassLabel}".`
      );
    } catch (error) {
      console.error(error);
      setFeedback("incorrect");
      toast.error("Error during model inference.");
    }
  };

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      resetState();
      setVideoFile({ blob: file, url: URL.createObjectURL(file) });
      runInference(file);
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
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
        stream.getTracks().forEach((track) => track.stop());
        if (videoRef.current) videoRef.current.srcObject = null;
        toast.info("Recording complete. Review or submit your video.");
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
    } catch (error) {
      console.error(error);
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
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-8 bg-[hsl(var(--background))] relative">

      {/* Navigation & Word */}
      <div className="flex items-center justify-between w-full max-w-2xl mb-8">
        {/* Previous Button */}
        <button
          onClick={() => {
            onPrevious();
            resetState();
          }}
          className="p-2 hover:opacity-80"
        >
          <svg
            fill="#b6d3b7"
            viewBox="0 0 32 32"
            className="w-10 h-10"
            style={{ transform: "scaleX(-1)" }}
          >
            <path d="M25.468,14.508l-20.967,-0.008c-0.828,-0  -1.501,0.672 -1.501,1.499c-0,0.828 0.672,1.501 1.499,1.501l21.125,0.009c-0.107,0.159 -0.234,0.306 -0.377,0.439c-3.787,3.502 -9.68,8.951 -9.68,8.951c-0.608,0.562 -0.645,1.511 -0.083,2.119c0.562,0.608 1.512,0.645 2.12,0.083c-0,0 5.892,-5.448 9.68,-8.95c1.112,-1.029 1.751,-2.47 1.766,-3.985c0.014,-1.515 -0.596,-2.968 -1.688,-4.018l-9.591,-9.221c-0.596,-0.574 -1.547,-0.556 -2.121,0.041c-0.573,0.597 -0.555,1.547 0.042,2.121l9.591,9.221c0.065,0.063 0.127,0.129 0.185,0.198Z"/>
          </svg>
        </button>

        <h2 className="text-5xl font-extrabold text-[hsl(var(--primary))]">{word}</h2>

        {/* Next Button */}
        <button
          onClick={() => {
            onNext();
            resetState();
          }}
          className="p-2 hover:opacity-80"
        >
          <svg fill="#b6d3b7" viewBox="0 0 32 32" className="w-10 h-10">
            <path d="M25.468,14.508l-20.967,-0.008c-0.828,-0  -1.501,0.672 -1.501,1.499c-0,0.828 0.672,1.501 1.499,1.501l21.125,0.009c-0.107,0.159 -0.234,0.306 -0.377,0.439c-3.787,3.502 -9.68,8.951 -9.68,8.951c-0.608,0.562 -0.645,1.511 -0.083,2.119c0.562,0.608 1.512,0.645 2.12,0.083c-0,0 5.892,-5.448 9.68,-8.95c1.112,-1.029 1.751,-2.47 1.766,-3.985c0.014,-1.515 -0.596,-2.968 -1.688,-4.018l-9.591,-9.221c-0.596,-0.574 -1.547,-0.556 -2.121,0.041c-0.573,0.597 -0.555,1.547 0.042,2.121l9.591,9.221c0.065,0.063 0.127,0.129 0.185,0.198Z"/>
          </svg>
        </button>
      </div>

      <Card className="w-full max-w-2xl shadow-xl bg-[hsl(var(--card))] text-[hsl(var(--card-foreground))]">
        <CardContent className="pt-6 space-y-6">

          {/* Video */}
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

            {/* Countdown */}
            {countdown !== null && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-50">
                <span className="text-white text-6xl font-bold animate-pulse">{countdown}</span>
              </div>
            )}

            {/* Placeholder */}
            {isIdle && (
              <div className="absolute inset-0 flex items-center justify-center bg-[hsl(var(--muted))/70]">
                <Camera className="h-16 w-16 text-[hsl(var(--muted-foreground))]" />
              </div>
            )}

            {/* Recording */}
            {isRecording && (
              <div className="absolute top-4 left-4 flex items-center px-3 py-1 bg-[hsl(var(--destructive))] rounded-full shadow-lg">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                </span>
                <span className="ml-2 text-sm font-semibold text-[hsl(var(--destructive-foreground))]">Recording...</span>
              </div>
            )}

            {/* Processing */}
            {feedback === "processing" && (
              <div className="absolute inset-0 bg-gray-800/90 flex flex-col items-center justify-center animate-pulse">
                <Loader className="h-10 w-10 text-white animate-spin mb-3" />
                <p className="text-white text-xl font-semibold">Analyzing Sign...</p>
              </div>
            )}

            {/* Feedback */}
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
                  className="flex-1 text-lg py-6 bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:brightness-90 transition"
                >
                  <Upload className="mr-2 h-5 w-5" /> Upload Video
                </Button>
                <Button
                  onClick={startRecording}
                  disabled={isPredicting}
                  className="flex-1 text-lg py-6 bg-[hsl(var(--accent))] text-[hsl(var(--accent-foreground))] hover:bg-[hsl(190,29%,28%)] transition"
                >
                  <Camera className="mr-2 h-5 w-5" /> Start Recording
                </Button>
              </div>
            )}

            <input ref={fileInputRef} type="file" accept="video/*" onChange={handleVideoUpload} className="hidden" />

            <div className="flex flex-wrap justify-center gap-4 mt-4">
              {isRecording && (
                <Button size="lg" onClick={stopRecording} className="text-lg px-8 py-6 bg-[hsl(var(--destructive))] text-[hsl(var(--destructive-foreground))] shadow-xl">
                  <StopCircle className="mr-2 h-5 w-5" /> Stop Recording
                </Button>
              )}

              {isReadyToSubmit && (
                <>
                  <Button size="lg" onClick={handleRetry} variant="outline" className="text-lg px-8 py-6 border-[hsl(var(--border))] text-[hsl(var(--foreground))] hover:bg-[hsl(var(--muted))]">
                    <Camera className="mr-2 h-5 w-5" /> Re-record
                  </Button>
                  <Button size="lg" onClick={() => videoFile && runInference(videoFile.blob)} className="text-lg px-8 py-6 bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] shadow-xl hover:brightness-90">
                    <Play className="mr-2 h-5 w-5" /> Submit
                  </Button>
                </>
              )}

              {feedback === "incorrect" && (
                <Button size="lg" onClick={handleRetry} variant="outline" className="text-lg px-8 py-6 border-[hsl(var(--destructive))] text-[hsl(var(--destructive))] hover:bg-[hsl(var(--muted))]">
                  <XCircle className="mr-2 h-5 w-5" /> Try Again
                </Button>
              )}
            </div>
          </div>

        </CardContent>
      </Card>
    </div>
  );
};
