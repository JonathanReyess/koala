import { AlertTriangle, CheckCircle, CircleAlert, Loader, XCircle } from "lucide-react";
import type { ReactNode } from "react";
import { PrimaryButton, QuietButton } from "./buttons";
import { primaryActionFor, type ResultKind } from "@/lib/stage/stageMachine";
import { cn } from "@/lib/utils";

const LOOK: Record<ResultKind, { icon: typeof CheckCircle; tone: string; label: string }> = {
  correct: { icon: CheckCircle, tone: "border-green-700 bg-green-50 text-green-900", label: "Correct" },
  close: { icon: CircleAlert, tone: "border-amber-600 bg-amber-50 text-amber-900", label: "Almost" },
  confused: { icon: XCircle, tone: "border-red-700 bg-red-50 text-red-900", label: "Not quite" },
  incorrect: { icon: XCircle, tone: "border-red-700 bg-red-50 text-red-900", label: "Not quite" },
  not_detected: { icon: AlertTriangle, tone: "border-yellow-600 bg-yellow-50 text-yellow-900", label: "Couldn't see you" },
  error: { icon: AlertTriangle, tone: "border-gray-600 bg-gray-100 text-gray-900", label: "Something went wrong" },
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
 * Shown BELOW the tiles (never over the video). One primary action, by outcome:
 * correct -> "Next sign"; everything else -> "Try again". "Watch again" is always the quieter secondary.
 */
export const ResultPanel = ({ result, message, onNext, onTryAgain, onWatchAgain, children }: Props) => {
  const look = LOOK[result];
  const Icon = look.icon;
  const primary = primaryActionFor(result);
  const errorText = result === "error" ? "Something went wrong checking that clip — this isn't your signing. Please try again." : message;
  return (
    <div data-testid="result-panel" data-result={result} className={cn("rounded-[var(--tile-radius)] border-[length:var(--tile-border)] p-4 md:p-5", look.tone)}>
      <div className="flex flex-col md:flex-row md:items-center gap-4 justify-between">
        <div role="status" aria-live="polite" className="flex items-center gap-4 min-w-0">
          <Icon className="h-12 w-12 shrink-0" aria-hidden="true" />
          <div className="min-w-0">
            <p className="text-sm font-bold uppercase tracking-wide">{look.label}</p>
            <p className="text-2xl md:text-3xl font-extrabold leading-tight" data-testid="result-message">{errorText}</p>
          </div>
        </div>
        <div className="flex flex-col-reverse sm:flex-row items-stretch sm:items-center gap-3 shrink-0">
          <QuietButton large onClick={onWatchAgain} data-testid="watch-again">
            Watch again
          </QuietButton>
          <PrimaryButton onClick={primary.event === "NEXT" ? onNext : onTryAgain} data-testid="primary-action">
            {primary.label}
          </PrimaryButton>
        </div>
      </div>

      {/* Reserved for future per-part feedback (handshape / location / movement) from the pose coach.
          Intentionally empty for now: `empty:hidden` collapses it until something is rendered here. */}
      <section
        data-slot="pose-coach-feedback"
        aria-label="Per-part feedback: handshape, location, movement"
        className="empty:hidden mt-4"
      />
      {children}
    </div>
  );
};

/** Shown in the same spot while the clip is being analysed. */
export const AnalyzingPanel = () => (
  <div data-testid="analyzing-panel" role="status" className="flex items-center justify-center gap-4 rounded-[var(--tile-radius)] border-[length:var(--tile-border)] border-[color:var(--tile-border-color)] bg-white p-5 min-h-[var(--tap-large)]">
    <Loader className="h-8 w-8 animate-spin" aria-hidden="true" />
    <span className="text-2xl font-bold">Analyzing your sign…</span>
  </div>
);
