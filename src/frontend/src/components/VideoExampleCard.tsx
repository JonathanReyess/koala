import { useState, useEffect, useRef } from "react";
import { SurfaceCard, MediaWell, MediaScrim, PillButton, PillButtonRow } from "@/components/ds";
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
      <SurfaceCard
        className="w-full"
        footer={
          <PillButtonRow>
            <PillButton
              variant="secondary"
              onClick={toggleVisibility}
              icon={isVisible ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
            >
              {isVisible ? "Hide Example" : "Show Example"}
            </PillButton>
            <PillButton onClick={toggleExample}>Example {example === 1 ? 2 : 1}</PillButton>
          </PillButtonRow>
        }
      >
        <MediaWell
          scrim={
            !isVisible ? (
              <MediaScrim passive icon={<EyeOff className="h-12 w-12" />} label="Video Hidden" />
            ) : isEnded ? (
              <MediaScrim
                icon={<RotateCcw className="h-12 w-12" />}
                label="Replay"
                onClick={handleReplay}
                aria-label="Replay the example video"
              />
            ) : null
          }
        >
          {isVisible && (
            <video ref={videoRef} controls autoPlay muted onEnded={() => setIsEnded(true)}>
              <source src={`/videos/${word}_example${example}.mp4`} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          )}
        </MediaWell>
      </SurfaceCard>
    </div>
  );
};
