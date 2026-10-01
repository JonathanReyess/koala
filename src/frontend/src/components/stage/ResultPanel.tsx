import { AlertTriangle, CheckCircle, CircleAlert, XCircle } from "lucide-react";
import type { ReactNode } from "react";
import { PillButton, SurfaceCard } from "@/components/ds";
import { primaryActionFor, type ResultKind } from "@/lib/stage/stageMachine";
import { cn } from "@/lib/utils";

/** Sage is the only brand colour: a positive result is sage, everything else is neutral ink. The icon + label carry the meaning. */
const LOOK: Record<ResultKind, { icon: typeof CheckCircle; positive: boolean; label: string }> = {
  correct: { icon: CheckCircle, positive: true, label: "Correct" },
  close: { icon: CircleAlert, positive: false, label: "Almost" },
  confused: { icon: XCircle, positive: false, label: "Not quite" },
  incorrect: { icon: XCircle, positive: false, label: "Not quite" },
  not_detected: { icon: AlertTriangle, positive: false, label: "Couldn't see you" },
  error: { icon: AlertTriangle, positive: false, label: "Something went wrong" },
};

interface Props {
  result: ResultKind;
  /** The existing feedback message (see lib/inference/messages.ts). */
  message: string;
  onNext: () => void;
  onTryAgain: () => void;
  onWatchAgain: () => void;
  /** Extra content (debug view). */
  children?: ReactNode;
}

/**
 * Shown BELOW the cards (never over the video). One primary action, by outcome:
 * correct -> "Next sign"; everything else -> "Try again". "Watch again" is always the quieter secondary.
 */
export const ResultPanel = ({ result, message, onNext, onTryAgain, onWatchAgain, children }: Props) => {
  const look = LOOK[result];
  const Icon = look.icon;
  const primary = primaryActionFor(result);
  const text = result === "error" ? "Something went wrong checking that clip. This isn't your signing. Please try again." : message;
  return (
    <SurfaceCard data-testid="result-panel" data-result={result} className="px-4 py-3 md:px-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div role="status" aria-live="polite" className="flex min-w-0 items-center gap-4">
          <span className={cn("inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-full", look.positive ? "bg-sage-50 text-sage-700" : "bg-sage-50 text-ink")}>
            <Icon className="h-7 w-7" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">{look.label}</p>
            <p className="text-lg font-bold leading-snug text-ink md:text-base lg:text-lg xl:text-xl" data-testid="result-message">
              {text}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 flex-col-reverse gap-3 sm:flex-row sm:items-center">
          <PillButton size="lg" variant="secondary" onClick={onWatchAgain} data-testid="watch-again">
            Watch again
          </PillButton>
          <PillButton size="lg" onClick={primary.event === "NEXT" ? onNext : onTryAgain} data-testid="primary-action">
            {primary.label}
          </PillButton>
        </div>
      </div>

      {/* Reserved for future per-part feedback (handshape / location / movement) from the pose coach.
          Intentionally empty for now: `empty:hidden` collapses it until something is rendered here. */}
      <section data-slot="pose-coach-feedback" aria-label="Per-part feedback: handshape, location, movement" className="mt-3 empty:hidden" />
      {children}
    </SurfaceCard>
  );
};
