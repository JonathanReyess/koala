import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

/** Thick-bordered video tile: top bar, video area, bottom bar. Controls live in the bars, never over the video. */
export const Tile = ({
  className,
  recording,
  children,
  ...p
}: HTMLAttributes<HTMLElement> & { recording?: boolean; children: ReactNode }) => (
  <section
    {...p}
    className={cn(
      "h-full overflow-hidden bg-[#f4efe6] flex flex-col border-solid rounded-[var(--tile-radius)]",
      "border-[length:var(--tile-border)]",
      recording ? "border-[color:var(--tile-border-recording)]" : "border-[color:var(--tile-border-color)]",
      className,
    )}
  >
    {children}
  </section>
);

export const TileBar = ({ className, ...p }: HTMLAttributes<HTMLDivElement>) => (
  <div {...p} className={cn("flex flex-wrap items-center justify-between gap-2 px-3 py-2 bg-[#f4efe6] min-h-[var(--tap-min)]", className)} />
);

/** 16:9 video area. Videos inside crop with object-position centre-bottom so hands stay visible. */
export const TileVideoArea = ({ className, style, ...p }: HTMLAttributes<HTMLDivElement>) => (
  <div {...p} style={{ maxHeight: "var(--video-max-h)", ...style }} className={cn("relative aspect-video bg-black", className)} />
);
