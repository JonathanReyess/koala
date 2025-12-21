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
      <Card className="w-full border-0 shadow-lg bg-white dark:bg-gray-900 rounded-2xl overflow-hidden">
        <CardContent className="p-6 space-y-6">
          {/* Video Section */}
          <div className="relative aspect-video bg-black rounded-xl overflow-hidden">
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
                    className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm hover:bg-black/70 transition-all group"
                  >
                    <div className="flex flex-col items-center gap-2">
                      <RotateCcw className="w-16 h-16 text-white group-hover:scale-110 transition-transform" />
                      <span className="text-white text-lg font-medium">
                        Replay
                      </span>
                    </div>
                  </button>
                )}
              </>
            ) : (
              <div className="flex items-center justify-center w-full h-full bg-gray-100 dark:bg-gray-800">
                <div className="flex flex-col items-center gap-3">
                  <EyeOff className="w-12 h-12 text-gray-400 dark:text-gray-600" />
                  <span className="text-gray-600 dark:text-gray-400 text-lg font-medium">
                    Video Hidden
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3">
            <Button
              onClick={toggleVisibility}
              className="flex-1 h-12 text-base font-medium rounded-full hover:opacity-90 transition-opacity"
              variant="outline"
            >
              {isVisible ? (
                <EyeOff className="mr-2 h-5 w-5" />
              ) : (
                <Eye className="mr-2 h-5 w-5" />
              )}
              {isVisible ? "Hide Example" : "Show Example"}
            </Button>

            <Button
              onClick={toggleExample}
              className="flex-1 h-12 text-base font-medium rounded-full hover:opacity-90 transition-opacity"
            >
              Example {example === 1 ? 2 : 1}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
