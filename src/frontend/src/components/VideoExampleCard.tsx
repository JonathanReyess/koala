import { useState, useEffect, useRef } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Eye, EyeOff, RotateCcw } from "lucide-react";

interface VideoExampleCardProps {
  word: string;
}

export const VideoExampleCard = ({ word }: VideoExampleCardProps) => {
  const [isVisible, setIsVisible] = useState(true);
  const [example, setExample] = useState<1 | 2>(1);
  const [isEnded, setIsEnded] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Reload video when word or example changes
  useEffect(() => {
    if (videoRef.current) {
      setIsEnded(false);
      videoRef.current.load();
    }
  }, [word, example]);

  const toggleVisibility = () => setIsVisible(!isVisible);
  const toggleExample = () => setExample(example === 1 ? 2 : 1);

  const handleReplay = () => {
    if (videoRef.current) {
      setIsEnded(false);
      videoRef.current.currentTime = 0;
      videoRef.current.play();
    }
  };

  return (
    <div className="flex-1 flex flex-col justify-center items-center w-full">
      <Card className="w-full h-full shadow-xl bg-[hsl(var(--card))] text-[hsl(var(--card-foreground))] flex flex-col">
        <CardContent className="pt-6 space-y-6 flex flex-col flex-1">
          {/* Video Section */}
          <div className="relative aspect-video w-full bg-[hsl(var(--muted))] border-2 border-dashed border-[hsl(var(--border))] rounded-xl overflow-hidden shadow-inner flex items-center justify-center">
            {isVisible ? (
              <>
                <video
                  ref={videoRef}
                  controls
                  autoPlay
                  muted
                  className="w-full h-full object-cover"
                  onEnded={() => setIsEnded(true)}
                >
                  <source
                    src={`/videos/${word}_example${example}.mp4`}
                    type="video/mp4"
                  />
                  Your browser does not support the video tag.
                </video>

                {/* Replay Button Overlay */}
                {isEnded && (
                  <button
                    onClick={handleReplay}
                    className="absolute inset-0 flex items-center justify-center bg-black/40 text-white hover:bg-black/60 transition rounded-xl"
                  >
                    <RotateCcw className="w-16 h-16" />
                  </button>
                )}
              </>
            ) : (
              <div className="flex items-center justify-center w-full h-full text-[hsl(var(--foreground))] text-xl font-semibold bg-[hsl(var(--muted))]">
                Video Hidden
              </div>
            )}
          </div>

          {/* Buttons */}
          <div className="flex flex-wrap justify-center gap-4 mt-4 w-full">
            <Button
              size="lg"
              onClick={toggleVisibility}
              className="flex-1 text-lg py-6 bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:brightness-90 transition flex items-center justify-center gap-2"
            >
              {isVisible ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
              {isVisible ? "Hide Video" : "Unhide Video"}
            </Button>

            <Button
              size="lg"
              onClick={toggleExample}
              className="flex-1 text-lg py-6 bg-[hsl(var(--accent))] text-[hsl(var(--accent-foreground))] hover:bg-[hsl(190,29%,28%)] transition flex items-center justify-center gap-2"
            >
              Example {example === 1 ? 2 : 1}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
