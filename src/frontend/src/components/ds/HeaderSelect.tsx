import type { ReactNode } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";

export interface HeaderSelectOption {
  value: string;
  label: ReactNode;
}

interface HeaderSelectProps {
  value: string;
  onValueChange: (value: string) => void;
  options: HeaderSelectOption[];
  "aria-label": string;
  className?: string;
}

/** Pill-shaped select for the page header (sage-50, hairline border): e.g. "All words (67)". */
export const HeaderSelect = ({ value, onValueChange, options, className, ...p }: HeaderSelectProps) => (
  <Select value={value} onValueChange={onValueChange}>
    <SelectTrigger
      aria-label={p["aria-label"]}
      className={cn(
        "h-btn rounded-chip bg-sage-50 text-sage-700 font-semibold [border:var(--card-border)] px-4 shadow-none",
        "focus:ring-0 focus:ring-offset-0 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage-600",
        className,
      )}
    >
      <SelectValue />
    </SelectTrigger>
    <SelectContent className="rounded-card bg-surface text-ink shadow-card [border:var(--card-border)]">
      {options.map((o) => (
        <SelectItem key={o.value} value={o.value} className="rounded-media focus:bg-sage-50 focus:text-sage-700">
          {o.label}
        </SelectItem>
      ))}
    </SelectContent>
  </Select>
);

/** Static pill for header metadata (not interactive). */
export const HeaderChip = ({ children, className, title }: { children: ReactNode; className?: string; title?: string }) => (
  <span title={title} className={cn("inline-flex items-center gap-2 h-8 rounded-chip bg-sage-50 px-3 text-sm font-semibold text-sage-700 [border:var(--card-border)]", className)}>
    {children}
  </span>
);
