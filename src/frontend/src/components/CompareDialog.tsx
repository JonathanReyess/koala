import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { RotateCcw } from "lucide-react";

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
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>
            Compare “{word}” with “{other}”
          </DialogTitle>
          <DialogDescription>
            These signs are easy to mix up. Watch how the hands, movement and position differ.
          </DialogDescription>
        </DialogHeader>

        {others.length > 1 && (
          <div className="flex flex-wrap gap-2" role="group" aria-label="Compare with">
            {others.map((w) => (
              <Button
                key={w}
                size="sm"
                variant={w === other ? "default" : "outline"}
                className="rounded-full"
                onClick={() => setOther(w)}
              >
                {w}
              </Button>
            ))}
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {[
            { w: word, ref: leftRef },
            { w: other, ref: rightRef },
          ].map(({ w, ref }) => (
            <figure key={w} className="space-y-2">
              <div className="aspect-video bg-black rounded-xl overflow-hidden">
                <video
                  ref={ref}
                  key={w}
                  src={clip(w)}
                  className="w-full h-full object-cover"
                  muted
                  loop
                  playsInline
                  autoPlay
                  aria-label={`Demo of ${w}`}
                />
              </div>
              <figcaption className="text-center font-semibold text-gray-900 dark:text-gray-100">{w}</figcaption>
            </figure>
          ))}
        </div>

        <Button variant="outline" className="rounded-full self-center" onClick={replay}>
          <RotateCcw className="mr-2 h-4 w-4" />
          Replay both
        </Button>
      </DialogContent>
    </Dialog>
  );
};
