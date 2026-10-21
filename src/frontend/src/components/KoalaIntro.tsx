import { useLayoutEffect, useRef } from "react";
import { animate } from "framer-motion";

// Big koala sits 2rem left of the corner koala's centre and hangs below the
// screen edge by this much so its legs stay hidden.
const BIG_X_OFFSET = -32;
const bigWidth = () => Math.min(innerWidth * 0.32, innerHeight * 0.45);
const bigSink = () => Math.min(innerWidth * 0.072, innerHeight * 0.098);

interface KoalaIntroProps {
  // false renders the koala straight into its corner spot.
  playIntro: boolean;
  onIntroDone?: () => void;
}

// One element the whole time: it's laid out in the corner and the intro is
// just transforms (scale/x/y) that animate back to identity, so nothing is
// swapped or remounted mid-animation.
export const KoalaIntro = ({ playIntro, onIntroDone }: KoalaIntroProps) => {
  const boxRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useLayoutEffect(() => {
    const box = boxRef.current;
    const img = imgRef.current;
    if (!playIntro || !box || !img) return;

    const rest = box.getBoundingClientRect();
    const scale = bigWidth() / rest.width;
    // The img inside is pushed down 25% for the corner peek; undo that while
    // big so the sink alone decides how much of the koala is hidden.
    const y = innerHeight + bigSink() - rest.bottom - rest.height * scale * 0.25;
    let cancelled = false;
    const controls: { stop: () => void }[] = [];
    const run = (...args: Parameters<typeof animate>) => {
      const c = animate(...args);
      controls.push(c);
      return c;
    };

    (async () => {
      run(box, { x: BIG_X_OFFSET, y, scale: 0 }, { duration: 0 });
      run(
        box,
        { scale, opacity: 1 },
        {
          scale: { type: "spring", stiffness: 260, damping: 18 },
          opacity: { duration: 0.15 },
        },
      );
      await run(
        img,
        { rotate: [0, -8, 8, -8, 8, -4, 0] },
        { delay: 0.4, duration: 1.1, ease: "easeInOut" },
      );
      if (cancelled) return;

      // Critically damped spring: eases off from rest, lands without overshoot.
      await run(
        box,
        { x: 0, y: 0, scale: 1 },
        { type: "spring", duration: 0.9, bounce: 0 },
      );
      if (!cancelled) onIntroDone?.();
    })();

    return () => {
      cancelled = true;
      controls.forEach((c) => c.stop());
    };
    // runs once per mount; remount to replay
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      ref={boxRef}
      className="absolute bottom-2 right-12 z-[60] w-48 md:w-72 pointer-events-none"
      // Hidden until the intro has moved it into the big position.
      style={{ transformOrigin: "50% 100%", opacity: playIntro ? 0 : undefined }}
    >
      {/* Separate wrapper: framer's rotate on the img would overwrite this translate. */}
      <div className="translate-y-1/4">
        <img
          ref={imgRef}
          src="/koala_wave.png"
          // Intrinsic size so the box has its real height before the image
          // loads; the intro measures it on mount.
          width={2226}
          height={1920}
          alt=""
          aria-hidden="true"
          className="w-full select-none"
          style={{ transformOrigin: "50% 100%" }}
        />
      </div>
    </div>
  );
};
