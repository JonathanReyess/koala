import type { StageView } from "@/lib/stage/stageMachine";
import { cn } from "@/lib/utils";

interface Props {
  view: StageView;
  onChange: (view: StageView) => void;
  className?: string;
}

/** Practice (watch the sign, then try it) vs Quiz (only the word and your camera). Pill-shaped. */
export const ViewSwitch = ({ view, onChange, className }: Props) => (
  <div
    role="group"
    aria-label="Mode"
    data-testid="view-switch"
    className={cn("inline-flex h-12 items-stretch gap-0.5 rounded-full bg-surface p-0.5 [border:var(--card-border)]", className)}
  >
    {(["practice", "quiz"] as const).map((v) => (
      <button
        key={v}
        type="button"
        aria-pressed={view === v}
        onClick={() => onChange(v)}
        className={cn(
          "rounded-full px-6 text-base font-semibold transition-colors",
          "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage-600",
          view === v ? "bg-sage-600 text-white" : "text-sage-700 hover:bg-sage-50",
        )}
      >
        {v === "practice" ? "Practice" : "Quiz"}
      </button>
    ))}
  </div>
);
