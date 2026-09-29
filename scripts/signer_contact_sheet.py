#!/usr/bin/env python3
"""Make a labelled contact sheet of the 20 KSL-77 signers (00-19) and a template signer table.

For each signer this picks one raw video and one clear mid-sign frame (both shoulders detected and, preferably,
both hands up, in the middle of the signing), then writes into --out:

    contact_sheet.png   one grid image, each tile labelled with the signer id (and the source video)
    signers.csv         template to fill in by eye:
                        signer_id, presentation (M/F/unknown), dominant_hand (R; L for 08),
                        exclude (0/1; 1 for 08), notes

Copy the completed table to data/signers.csv in the repo; scripts/make_demo_clips.py reads it
(skips exclude=1 signers, and --balance-presentation uses the presentation column).

Frame choice: among the signer's videos, the one where hands are visible in the most sampled frames (pose in
all of them); within it, the sampled frame at the middle of the hand-visible run, preferring frames with both
hands. Uses features_v2's manifest.csv + mask.npy (no MediaPipe needed).

Usage (Colab, from the repo root):
    python scripts/signer_contact_sheet.py \\
        --raw-videos /content/drive/MyDrive/KSL_Project/KSL_Raw_Videos \\
        --features-dir /content/drive/MyDrive/KSL_Project/features_v2 \\
        --out /content/drive/MyDrive/KSL_Project/signer_sheet
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_demo_clips import remap_raw_path, sampled_frame_indices  # noqa: E402

N_SIGNERS = 20
LEFT_HAND = slice(0, 21)
RIGHT_HAND = slice(21, 42)
SHOULDERS = (43, 44)
# Prefill for the template (RESULTS.md: signer 08 is the only left-hand-dominant / likely mirrored signer).
PREFILL = {"08": {"dominant_hand": "L", "exclude": 1, "notes": "likely left-handed or mirrored clip (RESULTS.md); excluded from demo clips"}}
CSV_COLUMNS = ["signer_id", "presentation", "dominant_hand", "exclude", "notes"]


def choose_frame(mask: np.ndarray, n_frames: int) -> Optional[tuple[int, int]]:
    """(sample_slot, frame_index) of a clear mid-sign frame from a (32, 47) mask, or None if unusable."""
    shoulders = mask[:, SHOULDERS[0]].astype(bool) & mask[:, SHOULDERS[1]].astype(bool)
    left = mask[:, LEFT_HAND].any(axis=1)
    right = mask[:, RIGHT_HAND].any(axis=1)
    usable = shoulders & (left | right)
    if not usable.any():
        return None
    slots = np.flatnonzero(usable)
    mid = (slots[0] + slots[-1]) / 2  # middle of the signing
    both = usable & left & right
    pool = np.flatnonzero(both) if both.any() else slots
    slot = int(pool[np.argmin(np.abs(pool - mid))])
    return slot, int(sampled_frame_indices(n_frames, mask.shape[0])[slot])


def choose_video(manifest: pd.DataFrame, mask: np.ndarray, signer_id: str) -> Optional[tuple[pd.Series, int, int]]:
    """Best (manifest row, slot, frame) for a signer: pose in every sampled frame, most hand-visible frames."""
    rows = manifest[(manifest["signer_id"] == signer_id) & (manifest["n_sampled_ok"] > 0)]
    rows = rows[rows["frac_frames_with_pose"] >= 0.95]
    best = None
    for pos, row in rows.sort_values("frac_frames_with_any_hand", ascending=False, kind="stable").iterrows():
        pick = choose_frame(mask[int(row["idx"]) if "idx" in row else pos], int(row["n_frames"]))
        if pick is not None:
            best = (row, pick[0], pick[1])
            break
    return best


def read_frame(video: Path, frame_index: int):
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


def label_tile(tile, text: str, sub: str = ""):
    import cv2

    h, w = tile.shape[:2]
    cv2.rectangle(tile, (0, 0), (max(150, 24 * len(text)) , 52), (0, 0, 0), -1)
    cv2.putText(tile, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    if sub:
        cv2.rectangle(tile, (0, h - 26), (w, h), (0, 0, 0), -1)
        cv2.putText(tile, sub, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return tile


def make_sheet(tiles: list, cols: int, tile_w: int, tile_h: int, pad: int = 6):
    rows = -(-len(tiles) // cols)
    sheet = np.full((rows * (tile_h + pad) + pad, cols * (tile_w + pad) + pad, 3), 32, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (tile_h + pad), pad + c * (tile_w + pad)
        sheet[y : y + tile_h, x : x + tile_w] = t
    return sheet


def write_template(path: Path, signer_ids: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for sid in signer_ids:
            pre = PREFILL.get(sid, {})
            w.writerow({
                "signer_id": sid,
                "presentation": "unknown",
                "dominant_hand": pre.get("dominant_hand", "R"),
                "exclude": pre.get("exclude", 0),
                "notes": pre.get("notes", ""),
            })


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-videos", required=True)
    p.add_argument("--features-dir", required=True, help="Dir with manifest.csv and mask.npy (extract_landmarks.py output)")
    p.add_argument("--out", required=True)
    p.add_argument("--tile-width", type=int, default=384)
    p.add_argument("--cols", type=int, default=5)
    return p


def main(argv=None) -> int:
    import cv2

    args = build_arg_parser().parse_args(argv)
    fdir = Path(args.features_dir)
    manifest = pd.read_csv(fdir / "manifest.csv", dtype={"signer_id": str})
    mask = np.load(fdir / "mask.npy")
    signer_ids = [f"{i:02d}" for i in range(N_SIGNERS)]
    tile_w = args.tile_width
    tile_h = tile_w * 9 // 16
    tiles, missing = [], []
    for sid in signer_ids:
        pick = choose_video(manifest, mask, sid)
        frame = None
        if pick is not None:
            row, slot, frame_index = pick
            src = remap_raw_path(Path(args.raw_videos), row["video_path"])
            frame = read_frame(src, frame_index) if src.exists() else None
            sub = f"{src.parent.name}/{src.name}  frame {frame_index}"
        if frame is None:
            missing.append(sid)
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            sub = "no usable frame"
        else:
            tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        tiles.append(label_tile(tile, sid, sub))
        print(f"signer {sid}: {sub}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "contact_sheet.png"), make_sheet(tiles, args.cols, tile_w, tile_h))
    write_template(out / "signers.csv", signer_ids)
    print(f"\nWrote {out/'contact_sheet.png'} and {out/'signers.csv'} (template).")
    if missing:
        print("WARNING: no usable frame for signer(s): " + ", ".join(missing))
    print("Next: fill in presentation (M/F) and check dominant_hand/exclude in signers.csv, then copy it to data/signers.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
