import { useLayoutEffect, type DependencyList, type RefObject } from "react";

/**
 * Sizes the media wells inside `rootRef` so the whole page fits the viewport without scrolling.
 *
 * It measures, in the browser, how much height everything EXCEPT the media wells takes (header, word row,
 * card chrome, buttons, attribution and paddings), then sets `--video-max-h` on the root to the height that
 * is left. Cards derive their width from it (see theme.ts --card-max-w), so videos shrink without cropping.
 * Only on md+ screens; phones stack the cards and scroll as usual.
 */
export function useFitToViewport(rootRef: RefObject<HTMLElement>, deps: DependencyList, { minPx = 200, slackPx = 4 } = {}) {
  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root || typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const wide = window.matchMedia("(min-width: 768px)");
    let raf = 0;

    const fit = () => {
      if (!wide.matches) {
        root.style.removeProperty("--video-max-h");
        return;
      }
      const main = root.closest("main") ?? root.parentElement;
      const last = main?.lastElementChild as HTMLElement | null;
      if (!main || !last) return;
      const wells = Array.from(root.querySelectorAll<HTMLElement>("[data-media-well]")).filter((w) => w.offsetParent !== null);
      if (wells.length === 0) return;
      const wellH = Math.max(...wells.map((w) => w.getBoundingClientRect().height));
      // Bottom of the actual content (main stretches to fill the screen, so use its last child + padding).
      const contentBottom = last.getBoundingClientRect().bottom + window.scrollY + parseFloat(getComputedStyle(main).paddingBottom || "0");
      const chrome = contentBottom - wellH;
      const cap = Math.max(minPx, Math.floor(window.innerHeight - chrome - slackPx));
      const next = `${cap}px`;
      if (root.style.getPropertyValue("--video-max-h") !== next) root.style.setProperty("--video-max-h", next);
    };

    // Two passes: setting the cap can change wrapping (e.g. the controls grid), so re-measure once more.
    const schedule = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        fit();
        raf = requestAnimationFrame(fit);
      });
    };

    schedule();
    window.addEventListener("resize", schedule);
    wide.addEventListener("change", schedule);
    void document.fonts?.ready.then(schedule);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", schedule);
      wide.removeEventListener("change", schedule);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
