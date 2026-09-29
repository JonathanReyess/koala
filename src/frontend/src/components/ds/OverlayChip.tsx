import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

export type OverlayChipTone = "default" | "sage" | "danger";

const TONE: Record<OverlayChipTone, string> = {
  default: "bg-white/80 text-ink",
  sage: "bg-sage-50 text-sage-700",
  danger: "bg-white/90 text-danger",
};

const chipBase =
  "inline-flex items-center gap-2 min-h-8 max-w-full rounded-chip px-3 py-1 text-sm font-medium [border:var(--card-border)] shadow-sm";

interface OverlayChipProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: OverlayChipTone;
  icon?: ReactNode;
}

/** Small pill that sits ON the media (status, hints, errors). Wraps its text instead of clipping. */
export const OverlayChip = ({ tone = "default", icon, className, children, ...p }: OverlayChipProps) => (
  <span {...p} className={cn(chipBase, "whitespace-normal", TONE[tone], className)}>
    {icon && <span className="shrink-0 inline-flex" aria-hidden="true">{icon}</span>}
    <span className="min-w-0">{children}</span>
  </span>
);

interface OverlayToggleProps {
  label: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  tone?: OverlayChipTone;
  className?: string;
}

/** A labelled on/off chip (role="switch") for options over the media, e.g. "Show tracking". */
export const OverlayToggle = ({ label, checked, onCheckedChange, tone = "default", className }: OverlayToggleProps) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    onClick={() => onCheckedChange(!checked)}
    className={cn(
      chipBase,
      "cursor-pointer select-none hover:bg-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage-600",
      TONE[tone],
      className,
    )}
  >
    <span
      aria-hidden="true"
      className={cn("relative h-4 w-7 shrink-0 rounded-full transition-colors", checked ? "bg-sage-600" : "bg-ink-muted")}
    >
      <span
        className={cn("absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all", checked ? "left-[14px]" : "left-0.5")}
      />
    </span>
    {label}
  </button>
);

/** Icon-only round chip button for media overlays (flip, replay). Needs an aria-label. */
export const OverlayIconButton = ({
  label,
  onClick,
  children,
  className,
}: {
  label: string;
  onClick?: () => void;
  children: ReactNode;
  className?: string;
}) => (
  <button
    type="button"
    aria-label={label}
    onClick={onClick}
    className={cn(
      "inline-flex h-10 w-10 items-center justify-center rounded-full bg-white/80 text-ink [border:var(--card-border)] shadow-sm",
      "hover:bg-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage-600",
      className,
    )}
  >
    {children}
  </button>
);
