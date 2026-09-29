import { useEffect, useRef, useState } from "react";
import { RotateCcw } from "lucide-react";
import { AppDialog, MediaWell, PillButton, PillButtonRow } from "@/components/ds";

interface CompareDialogProps {
  word: string;
  /** Confusable words (with demo videos) to compare against. */
  others: string[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** Plays two signs' demo videos side by side, in sync, so learners can see how they differ. */
export const CompareDialog = ({ word, others, open, onOpenChange }: CompareDialogProps) => {
  const [other, setOther] = useState(others[0]);
  const leftRef = useRef<HTMLVideoElement>(null);
  const rightRef = useRef<HTMLVideoElement>(null);

  // Keep the selection valid when the word (and so the candidate list) changes.
  useEffect(() => {
    if (!others.includes(other)) setOther(others[0]);
  }, [others, other]);

  const replay = () => {
    for (const v of [leftRef.current, rightRef.current]) {
      if (!v) continue;
      v.currentTime = 0;
      void v.play();
    }
  };

  useEffect(() => {
    if (open) replay();
  }, [open, other, word]);

  if (!other) return null;
  const clip = (w: string) => `/videos/${w}_example1.mp4`;

  return (
    <AppDialog
      open={open}
      onOpenChange={onOpenChange}
      size="lg"
      title={`Compare “${word}” with “${other}”`}
      description="These signs are easy to mix up. Watch how the hands, movement and position differ."
      footer={
        <PillButtonRow>
          <PillButton variant="secondary" icon={<RotateCcw className="h-5 w-5" />} onClick={replay}>
            Replay both
          </PillButton>
          <PillButton onClick={() => onOpenChange(false)}>Done</PillButton>
        </PillButtonRow>
      }
    >
      {others.length > 1 && (
        <div className="mb-4 flex flex-wrap gap-2" role="group" aria-label="Compare with">
          {others.map((w) => (
            <PillButton key={w} size="sm" variant={w === other ? "primary" : "secondary"} aria-pressed={w === other} onClick={() => setOther(w)}>
              {w}
            </PillButton>
          ))}
        </div>
      )}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {[
          { w: word, ref: leftRef },
          { w: other, ref: rightRef },
        ].map(({ w, ref }) => (
          <figure key={w}>
            <MediaWell className="m-0">
              <video ref={ref} key={w} src={clip(w)} muted loop playsInline autoPlay aria-label={`Demo of ${w}`} />
            </MediaWell>
            <figcaption className="mt-2 text-center font-semibold text-ink">{w}</figcaption>
          </figure>
        ))}
      </div>
    </AppDialog>
  );
};
