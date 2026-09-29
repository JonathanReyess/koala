import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface AppDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  /** Footer actions: usually a PillButtonRow with two PillButtons. */
  footer?: ReactNode;
  /** Confirm-style dialogs use role="alertdialog". */
  alert?: boolean;
  /** max width: md (confirmations) or lg (media, e.g. side-by-side compare). */
  size?: "md" | "lg";
  className?: string;
}

/**
 * The one modal style: same card language as SurfaceCard (radius, hairline border, soft shadow),
 * 40% ink dim overlay with at most 4px blur, title + body + footer of PillButtons.
 */
export const AppDialog = ({ open, onOpenChange, title, description, children, footer, alert, size = "md", className }: AppDialogProps) => (
  <Dialog.Root open={open} onOpenChange={onOpenChange}>
    <Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 z-50 bg-[var(--scrim)] backdrop-blur-[4px] data-[state=open]:animate-in data-[state=open]:fade-in-0" />
      <Dialog.Content
        role={alert ? "alertdialog" : "dialog"}
        className={cn(
          "fixed left-1/2 top-1/2 z-50 w-[calc(100%-2rem)] -translate-x-1/2 -translate-y-1/2 max-h-[calc(100dvh-2rem)] overflow-y-auto",
          "bg-surface text-ink rounded-card shadow-card [border:var(--card-border)] p-6 focus:outline-none",
          "data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95",
          size === "md" ? "max-w-md" : "max-w-3xl",
          className,
        )}
      >
        <Dialog.Title className="pr-10 text-xl font-bold text-ink">{title}</Dialog.Title>
        {description ? (
          <Dialog.Description className="mt-2 text-[15px] leading-relaxed text-ink-muted">{description}</Dialog.Description>
        ) : (
          <Dialog.Description className="sr-only">{typeof title === "string" ? title : "Dialog"}</Dialog.Description>
        )}
        {children && <div className="mt-4">{children}</div>}
        {footer && <div className="mt-6">{footer}</div>}
        <Dialog.Close
          aria-label="Close"
          className="absolute right-4 top-4 inline-flex h-9 w-9 items-center justify-center rounded-full text-ink-muted hover:bg-sage-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage-600"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </Dialog.Close>
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>
);
