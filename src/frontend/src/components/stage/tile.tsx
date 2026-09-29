import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

/** Clean white video card: soft shadow, hairline border. Top bar, video area, bottom bar; nothing floats over the video. */
export const Tile = ({
  className,
  recording,
  children,
  ...p
}: HTMLAttributes<HTMLElement> & { recording?: boolean; children: ReactNode }) => (
  <section
    {...p}
    className={cn(
      "h-full overflow-hidden bg-white flex flex-col border-solid rounded-[var(--tile-radius)] shadow-[var(--tile-shadow)]",
      "border-[length:var(--tile-border)] border-[color:var(--tile-border-color)]",
      // Recording: a red ring (the "Recording" label + timer say it too, so it's not colour alone).
      recording && "ring-[3px] ring-[color:var(--tile-border-recording)]",
      className,
    )}
  >
    {children}
  </section>
);

export const TileBar = ({
  className,
  position = "top",
  ...p
}: HTMLAttributes<HTMLDivElement> & { position?: "top" | "bottom" }) => (
  <div
    {...p}
    className={cn(
      "flex flex-wrap items-center justify-between gap-2 px-4 py-2 bg-white min-h-[var(--tap-min)] border-gray-100",
      position === "top" ? "border-b" : "border-t",
      className,
    )}
  />
);

/** 16:9 video area. Videos inside crop with object-position centre-bottom so hands stay visible. */
export const TileVideoArea = ({ className, style, ...p }: HTMLAttributes<HTMLDivElement>) => (
  <div {...p} style={{ maxHeight: "var(--video-max-h)", ...style }} className={cn("relative aspect-video bg-black", className)} />
);
