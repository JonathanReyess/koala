// @vitest-environment jsdom
import { describe, it, expect, vi, beforeAll, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { AppDialog, HeaderSelect, MediaScrim, MediaWell, MEDIA_ASPECT, OverlayChip, OverlayIconButton, OverlayToggle, PillButton, PillButtonRow, SurfaceCard } from "./index";

beforeAll(() => {
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as never;
  // Radix Select/Dialog probe these in jsdom
  (Element.prototype as never as { hasPointerCapture: () => boolean }).hasPointerCapture = () => false;
  Element.prototype.scrollIntoView = () => {};
});
afterEach(cleanup);

describe("PillButton", () => {
  it("renders a semantic 44px pill with icon + label and defaults to type=button", () => {
    render(<PillButton icon={<svg data-testid="i" />}>Record</PillButton>);
    const b = screen.getByRole("button", { name: "Record" });
    expect(b.getAttribute("type")).toBe("button");
    expect(b.className).toContain("rounded-btn");
    expect(b.className).toContain("h-btn");
    expect(b.querySelector('[data-testid="i"]')).toBeTruthy();
    expect(b.querySelector("span[aria-hidden=true]")).toBeTruthy(); // decorative icon
  });
  it("has primary / secondary / ghost / danger variants using sage + danger tokens only", () => {
    const cls = (v: "primary" | "secondary" | "ghost" | "danger") => {
      const { unmount } = render(<PillButton variant={v}>x</PillButton>);
      const c = screen.getByRole("button").className;
      unmount();
      return c;
    };
    expect(cls("primary")).toMatch(/bg-sage-600.*text-white/);
    expect(cls("secondary")).toMatch(/bg-sage-200.*text-ink/);
    expect(cls("ghost")).toContain("bg-transparent");
    expect(cls("danger")).toContain("bg-danger");
  });
  it("disabled is 50% opacity, not clickable, and ignores clicks", () => {
    const onClick = vi.fn();
    render(<PillButton disabled onClick={onClick}>Nope</PillButton>);
    const b = screen.getByRole("button") as HTMLButtonElement;
    expect(b.disabled).toBe(true);
    expect(b.className).toContain("disabled:opacity-50");
    expect(b.className).toContain("disabled:pointer-events-none");
    fireEvent.click(b);
    expect(onClick).not.toHaveBeenCalled();
  });
  it("has a sage focus ring: 2px, offset 2px", () => {
    render(<PillButton>x</PillButton>);
    const c = screen.getByRole("button").className;
    expect(c).toContain("focus-visible:outline-2");
    expect(c).toContain("focus-visible:outline-offset-2");
    expect(c).toContain("focus-visible:outline-sage-600");
  });
  it("PillButtonRow gives equal-width children", () => {
    render(<PillButtonRow><PillButton>A</PillButton><PillButton>B</PillButton></PillButtonRow>);
    const row = screen.getByText("A").closest("div")!;
    expect(row.className).toContain("[&>*]:flex-1");
    expect(row.className).toContain("gap-3"); // 12px
  });
});

describe("SurfaceCard + MediaWell", () => {
  it("SurfaceCard uses the card tokens and pads only the footer/header slots, not the media", () => {
    render(
      <SurfaceCard data-testid="card" header={<span>Head</span>} footer={<span data-testid="foot">Foot</span>}>
        <div data-testid="media" />
      </SurfaceCard>,
    );
    const card = screen.getByTestId("card");
    expect(card.className).toContain("rounded-card");
    expect(card.className).toContain("shadow-card");
    expect(card.className).toContain("[border:var(--card-border)]");
    expect(screen.getByTestId("foot").parentElement!.className).toContain("p-4");
    expect(screen.getByTestId("media").parentElement).toBe(card); // media sits directly in the card, unpadded
  });

  it("MediaWell is inset, has its own radius, one shared aspect ratio, and fills its video", () => {
    render(
      <MediaWell data-testid="well">
        <video data-testid="v" />
      </MediaWell>,
    );
    const well = screen.getByTestId("well");
    expect(well.className).toContain("m-[var(--media-inset)]");
    expect(well.className).toContain("rounded-media");
    expect(well.className).toContain("overflow-hidden");
    expect(well.style.aspectRatio.replace(/\s/g, "")).toBe(MEDIA_ASPECT.replace(/\s/g, ""));
    expect(well.className).toContain("[&>video]:object-cover");
  });

  it("overlay slots sit inside the well (not the card chrome), padded 12px, and wrap", () => {
    render(
      <SurfaceCard data-testid="card">
        <MediaWell data-testid="well" topLeft={<OverlayChip>Hint</OverlayChip>} topRight={<OverlayToggle label="Show tracking" checked={false} onCheckedChange={() => {}} />} bottomRight={<OverlayIconButton label="Flip">f</OverlayIconButton>}>
          <video />
        </MediaWell>
      </SurfaceCard>,
    );
    const well = screen.getByTestId("well");
    for (const el of [screen.getByText("Hint"), screen.getByRole("switch", { name: "Show tracking" }), screen.getByRole("button", { name: "Flip" })]) {
      expect(well.contains(el)).toBe(true);
    }
    const overlay = screen.getByText("Hint").closest("div.absolute")!;
    expect(overlay.className).toContain("p-3");
    expect(overlay.innerHTML).toContain("flex-wrap");
  });

  it("MediaScrim: a labelled button when clickable (replay), a status block when passive", () => {
    const onClick = vi.fn();
    const { rerender } = render(<MediaWell scrim={<MediaScrim icon={<svg />} label="Replay" onClick={onClick} />} />);
    fireEvent.click(screen.getByRole("button", { name: "Replay" }));
    expect(onClick).toHaveBeenCalled();
    rerender(<MediaWell scrim={<MediaScrim passive label="Video Hidden" />} />);
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.getByRole("status").textContent).toBe("Video Hidden");
  });
});

describe("Overlay chips", () => {
  it("OverlayToggle is an accessible switch that toggles", () => {
    const Harness = () => {
      const [on, setOn] = useState(false);
      return <OverlayToggle label="Show tracking" checked={on} onCheckedChange={setOn} />;
    };
    render(<Harness />);
    const sw = screen.getByRole("switch", { name: "Show tracking" });
    expect(sw.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(sw);
    expect(sw.getAttribute("aria-checked")).toBe("true");
  });
  it("OverlayChip wraps long text instead of clipping, and tones use tokens", () => {
    render(<OverlayChip tone="sage">A fairly long status message that must wrap</OverlayChip>);
    const chip = screen.getByText(/fairly long/).closest("span")!.parentElement!;
    expect(chip.className).toContain("whitespace-normal");
    expect(chip.className).toContain("bg-sage-50");
  });
});

describe("AppDialog", () => {
  it("uses the card language, a dim overlay, title + body + footer of PillButtons; alert => alertdialog", () => {
    const onOpenChange = vi.fn();
    render(
      <AppDialog
        alert
        open
        onOpenChange={onOpenChange}
        title="Reset all progress?"
        description="This cannot be undone."
        footer={<PillButtonRow><PillButton variant="secondary">Cancel</PillButton><PillButton variant="danger">Reset Progress</PillButton></PillButtonRow>}
      />,
    );
    const dlg = screen.getByRole("alertdialog");
    expect(within(dlg).getByText("Reset all progress?")).toBeTruthy();
    expect(within(dlg).getByText("This cannot be undone.")).toBeTruthy();
    expect(within(dlg).getByRole("button", { name: "Cancel" })).toBeTruthy();
    expect(dlg.className).toContain("rounded-card");
    expect(dlg.className).toContain("shadow-card");
    const overlay = document.querySelector("[class*='var(--scrim)']")!;
    expect(overlay.className).toContain("backdrop-blur-[4px]");
    fireEvent.click(within(dlg).getByRole("button", { name: "Close" }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});

describe("HeaderSelect", () => {
  it("renders the current option as a sage pill trigger", () => {
    render(<HeaderSelect aria-label="Choose a deck" value="all" onValueChange={() => {}} options={[{ value: "all", label: "All words (67)" }, { value: "time", label: "Time (10)" }]} />);
    const t = screen.getByRole("combobox", { name: "Choose a deck" });
    expect(t.textContent).toContain("All words (67)");
    expect(t.className).toContain("rounded-chip");
    expect(t.className).toContain("bg-sage-50");
  });
});
