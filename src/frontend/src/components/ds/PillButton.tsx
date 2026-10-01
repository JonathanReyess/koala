import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export type PillButtonVariant = "primary" | "secondary" | "ghost" | "danger";

interface PillButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: PillButtonVariant;
  icon?: ReactNode;
  /** Fill the available width (use for equal-width buttons in a footer). */
  block?: boolean;
  /** sm 36px (inline), md 44px (default), touch 48px (minimum tap target for stage controls), lg 60px (primary actions). */
  size?: "sm" | "md" | "touch" | "lg";
}

const SIZE = {
  sm: "h-9 min-h-9 px-4 text-sm",
  md: "h-btn min-h-btn px-5 text-[15px]",
  touch: "h-12 min-h-12 px-5 text-[15px]",
  lg: "h-[60px] min-h-[60px] px-7 text-lg",
} as const;

const VARIANT: Record<PillButtonVariant, string> = {
  primary: "bg-sage-600 text-white hover:bg-sage-700 active:bg-sage-700",
  secondary: "bg-sage-200 text-ink hover:brightness-95 active:brightness-90",
  ghost: "bg-transparent text-sage-700 hover:bg-sage-50 active:bg-sage-50",
  danger: "bg-danger text-white hover:brightness-95 active:brightness-90",
};

/**
 * Pill button: full radius, 44px tall, icon + label. Variants: primary (dark sage), secondary (light sage),
 * ghost, danger (Reset / destructive only). Generic on purpose: no page-specific naming.
 */
export const PillButton = forwardRef<HTMLButtonElement, PillButtonProps>(
  ({ variant = "primary", size = "md", icon, block, className, children, type = "button", ...p }, ref) => (
    <button
      ref={ref}
      type={type}
      {...p}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-btn font-semibold whitespace-nowrap select-none",
        SIZE[size],
        "transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage-600",
        "disabled:opacity-50 disabled:pointer-events-none",
        block && "w-full",
        VARIANT[variant],
        className,
      )}
    >
      {icon && <span className="shrink-0 inline-flex" aria-hidden="true">{icon}</span>}
      {children}
    </button>
  ),
);
PillButton.displayName = "PillButton";

/** Footer row of equal-width PillButtons (12px gap, wraps to a column on very narrow screens). */
export const PillButtonRow = ({ children, className, wrap }: { children: ReactNode; className?: string; wrap?: boolean }) => (
  <div className={cn("flex gap-3 [&>*]:flex-1", wrap ? "flex-wrap [&>*]:min-w-[7.5rem]" : "[&>*]:min-w-0", className)}>{children}</div>
);
