import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/**
 * The stage's only two button looks: ONE primary (filled, brand CTA colour, 60px tall) per screen, and quieter
 * secondary buttons (outlined, 48px tall). Both meet the minimum tap size from the theme tokens.
 */
type Props = ButtonHTMLAttributes<HTMLButtonElement>;

export const PrimaryButton = forwardRef<HTMLButtonElement, Props>(({ className, ...p }, ref) => (
  <button
    ref={ref}
    type="button"
    {...p}
    className={cn(
      "inline-flex items-center justify-center gap-3 rounded-full px-8 text-xl font-bold text-white shadow-md",
      "min-h-[var(--tap-large)] min-w-[var(--tap-large)] bg-[var(--cta)] hover:bg-[var(--cta-hover)]",
      "focus-visible:outline focus-visible:outline-4 focus-visible:outline-offset-2 focus-visible:outline-[var(--cta)]",
      "disabled:opacity-50 disabled:cursor-not-allowed transition-colors",
      className,
    )}
  />
));
PrimaryButton.displayName = "PrimaryButton";

export const QuietButton = forwardRef<HTMLButtonElement, Props & { large?: boolean; pressed?: boolean }>(
  ({ className, large, pressed, ...p }, ref) => (
    <button
      ref={ref}
      type="button"
      aria-pressed={pressed}
      {...p}
      className={cn(
        "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-full border border-gray-300 bg-white shadow-sm px-5",
        "text-base font-semibold text-gray-900 hover:bg-gray-50",
        large ? "min-h-[var(--tap-large)] min-w-[var(--tap-large)]" : "min-h-[var(--tap-min)] min-w-[var(--tap-min)]",
        pressed && "bg-[var(--cta)] text-white border-[var(--cta)] hover:bg-[var(--cta-hover)]",
        "focus-visible:outline focus-visible:outline-4 focus-visible:outline-offset-2 focus-visible:outline-[var(--cta)]",
        "disabled:opacity-50 disabled:cursor-not-allowed transition-colors",
        className,
      )}
    />
  ),
);
QuietButton.displayName = "QuietButton";
