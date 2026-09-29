import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

interface SurfaceCardProps extends Omit<HTMLAttributes<HTMLElement>, "title"> {
  /** Optional header row (padded). */
  header?: ReactNode;
  /** Optional footer, typically PillButtons. Padded 16px from the card edges; the media above is not. */
  footer?: ReactNode;
}

/**
 * White card: token radius, hairline border, very soft shadow. Children (usually a MediaWell) are NOT padded
 * by the card; padding exists only around the header and footer slots.
 */
export const SurfaceCard = ({ header, footer, className, children, ...p }: SurfaceCardProps) => (
  <section
    {...p}
    className={cn(
      "flex flex-col bg-surface text-ink rounded-card shadow-card [border:var(--card-border)] overflow-hidden",
      className,
    )}
  >
    {header && <div className="px-4 pt-4">{header}</div>}
    {children}
    {footer && <div className="p-4 pt-1 mt-auto">{footer}</div>}
  </section>
);
