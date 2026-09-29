import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

interface MediaWellProps extends HTMLAttributes<HTMLDivElement> {
  /** <video> / <canvas> / <img>: sized to fill the well and cropped with object-fit: cover. */
  children?: ReactNode;
  /** CSS aspect-ratio; ONE value is used for every well on a page so columns line up. */
  aspect?: string;
  /** Overlay slots (padded 12px, on the media, never in the card chrome). Rows wrap instead of colliding. */
  topLeft?: ReactNode;
  topRight?: ReactNode;
  bottomLeft?: ReactNode;
  bottomRight?: ReactNode;
  /** Full-cover state (replay, busy, result): usually a <MediaScrim/>. */
  scrim?: ReactNode;
  /** Vertical anchor of the crop (keeps hands in view). */
  objectPosition?: string;
}

/** The single aspect ratio used by every MediaWell (16:10 keeps signers' hands visible on 16:9 clips). */
export const MEDIA_ASPECT = "16 / 10";

/**
 * Inset media well: sits `--media-inset` from the card edge, has its own radius, clips its content.
 * Overlays are placed by slot; nothing else should be absolutely positioned over the media.
 */
export const MediaWell = ({
  children, aspect = MEDIA_ASPECT, topLeft, topRight, bottomLeft, bottomRight, scrim, objectPosition = "center bottom", className, style, ...p
}: MediaWellProps) => {
  const row = (left: ReactNode, right: ReactNode, align: "start" | "end") =>
    left || right ? (
      <div className={cn("flex flex-wrap justify-between gap-2 pointer-events-none [&>*]:pointer-events-auto", align === "start" ? "items-start" : "items-end")}>
        <div className="flex flex-wrap items-center gap-2 min-w-0 max-w-full">{left}</div>
        <div className="flex flex-wrap items-center justify-end gap-2 min-w-0 max-w-full ml-auto">{right}</div>
      </div>
    ) : null;

  return (
    <div
      {...p}
      style={{ aspectRatio: aspect, ["--media-object-position" as string]: objectPosition, ...style }}
      className={cn(
        "relative m-[var(--media-inset)] overflow-hidden rounded-media bg-ink",
        "[&>video]:absolute [&>video]:inset-0 [&>video]:h-full [&>video]:w-full [&>video]:object-cover [&>video]:[object-position:var(--media-object-position)]",
        "[&>canvas]:absolute [&>canvas]:inset-0 [&>canvas]:h-full [&>canvas]:w-full [&>canvas]:pointer-events-none",
        "[&>img]:absolute [&>img]:inset-0 [&>img]:h-full [&>img]:w-full [&>img]:object-cover",
        className,
      )}
    >
      {children}
      {scrim}
      {(topLeft || topRight || bottomLeft || bottomRight) && (
        <div className="absolute inset-0 z-20 flex flex-col justify-between gap-2 p-3 pointer-events-none">
          {row(topLeft, topRight, "start") ?? <div />}
          {row(bottomLeft, bottomRight, "end")}
        </div>
      )}
    </div>
  );
};

interface MediaScrimProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  icon?: ReactNode;
  label?: ReactNode;
  /** sage: replay / busy / positive states; ink: neutral or needs-attention states (no red on media). */
  tone?: "sage" | "ink";
  children?: ReactNode;
  /** Render as a plain (non-interactive) block, e.g. while analysing. */
  passive?: boolean;
}

/** Muted scrim with a centred icon + label (the Replay treatment). A button when clickable, otherwise a block. */
export const MediaScrim = ({ icon, label, tone = "sage", passive, className, children, onClick, ...p }: MediaScrimProps) => {
  const cls = cn(
    "absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 p-6 text-center text-white",
    tone === "sage" ? "bg-[var(--media-scrim)]" : "bg-[var(--media-scrim-ink)]",
    className,
  );
  const content = (
    <>
      {icon && <span className="inline-flex" aria-hidden="true">{icon}</span>}
      {label && <span className="text-lg font-semibold leading-snug max-w-[28ch]">{label}</span>}
      {children}
    </>
  );
  if (passive || !onClick) return <div className={cls} role="status">{content}</div>;
  return (
    <button
      type="button"
      onClick={onClick}
      {...p}
      className={cn(cls, "cursor-pointer focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-4px] focus-visible:outline-white")}
    >
      {content}
    </button>
  );
};
