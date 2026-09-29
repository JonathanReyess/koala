#!/usr/bin/env python3
"""Make demo clips for trained classes that have none in src/frontend/public/videos.

For each such class this picks 2 source videos from KSL_Raw_Videos/<class_id>/,
trims the idle frames at the start and end using the per-frame hand mask from
features_v2, re-encodes them as small H.264 MP4s named <word>_example1.mp4 /
<word>_example2.mp4 (lowercase .mp4; <word> = the practice word in
src/frontend/src/data/vocab.json, which is also what the app looks up), and
writes them plus a provenance table into --out (a Drive folder) so you can
download them into src/frontend/public/videos.

Source and licence are the same as the existing demo clips: the KSL-77 dataset
(Yang et al., MMM 2020), CC BY-NC 4.0. An ATTRIBUTION.txt is written next to the
clips and the app's on-page attribution stays as is.

Clip choice: among a class's held-out test clips, prefer ones the model got
right in *every* run of runs/calibration/predictions.csv with the highest mean
target probability (i.e. clear, prototypical signing), from two different
signers when possible.

Signer table (data/signers.csv, made from scripts/signer_contact_sheet.py):
signers with exclude=1 are never used. With --balance-presentation each word
gets its best male- and best female-presenting signer (each still chosen by the
usual ranking); if a word has no valid clip for one presentation it falls back
to the best two clips and is listed at the end.

Trimming: features_v2/mask.npy marks, for the 32 frames sampled per clip, which
joints MediaPipe found. The active window runs from the first to the last
sampled frame with a hand detected, widened by one sample step plus --pad
seconds, and never shorter than --min-seconds.

Usage (Colab; mount Drive first, run from the repo root):
    python scripts/make_demo_clips.py \\
        --raw-videos /content/drive/MyDrive/KSL_Project/KSL_Raw_Videos \\
        --features-dir /content/drive/MyDrive/KSL_Project/features_v2 \\
        --predictions /content/drive/MyDrive/KSL_Project/runs/calibration/predictions.csv \\
        --out /content/drive/MyDrive/KSL_Project/demo_clips

To regenerate clips for all 67 words with balanced signers:
    ... --all --pad 0.5 --balance-presentation

Add --dry-run to print the picks and trim windows without needing the raw videos or ffmpeg.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEQ_LEN = 32
LEFT_RIGHT_HANDS = slice(0, 42)

ATTRIBUTION = """These demo clips are trimmed and re-encoded from videos of the Korean Sign Language (KSL-77)
dataset:

  Yang, S., Jung, S., Kang, H., & Kim, C. (2020). The Korean Sign Language Dataset for Action Recognition.
  In MultiMedia Modeling (MMM 2020), Springer, Cham. https://doi.org/10.1007/978-3-030-37731-1_43

Licence: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) — attribution required,
non-commercial use only. Changes made: trimmed idle frames at the start/end, downscaled and re-encoded
(H.264, no audio). See ATTRIBUTION.md in the repository.

Signer selection: the demo clips deliberately exclude signers 05 and 06 (minors) and 08 (left-handed).
Each word shows one male-presenting and one female-presenting signer where available.
This applies only to the demo clips shown in the app: the landmarks of ALL signers are still
used for training and evaluation. (Signer table: data/signers.csv.)
"""


# --------------------------------------------------------------------------
# Pure helpers (unit-tested in test_make_demo_clips.py)
# --------------------------------------------------------------------------


def load_vocab(path: Path) -> dict[int, str]:
    """class id -> practice word (the demo-video filename stem)."""
    words = json.loads(Path(path).read_text())["words"]
    return {int(w["classId"]): w["english"] for w in words}


def words_with_clips(videos_dir: Path) -> set[str]:
    files = {p.name for p in Path(videos_dir).glob("*.mp4")}
    stems = {n[: -len("_example1.mp4")] for n in files if n.endswith("_example1.mp4")}
    return {s for s in stems if f"{s}_example2.mp4" in files}


def sampled_frame_indices(n_frames: int, seq_len: int = SEQ_LEN) -> np.ndarray:
    """The frames extract_landmarks.py sampled: np.linspace(0, n-1, 32, dtype=int)."""
    return np.linspace(0, n_frames - 1, seq_len, dtype=int)


def active_window(
    mask: np.ndarray,
    n_frames: int,
    fps: float,
    pad_seconds: float = 0.25,
    min_seconds: float = 1.0,
) -> tuple[float, float]:
    """(start_s, end_s) covering the signing, from the 32-frame hand mask.

    mask: (32, 47) uint8. Falls back to the whole clip when no hand was ever detected.
    """
    duration = n_frames / fps
    idx = sampled_frame_indices(n_frames, mask.shape[0])
    active = np.flatnonzero(mask[:, LEFT_RIGHT_HANDS].any(axis=1))
    if active.size == 0:
        return 0.0, duration
    first = idx[max(active[0] - 1, 0)]  # one sample step of context on each side
    last = idx[min(active[-1] + 1, len(idx) - 1)]
    start = max(0.0, first / fps - pad_seconds)
    end = min(duration, (last + 1) / fps + pad_seconds)
    if end - start < min_seconds:  # grow symmetrically up to the clip bounds
        mid = (start + end) / 2
        start, end = max(0.0, mid - min_seconds / 2), min(duration, mid + min_seconds / 2)
    return float(start), float(end)


def load_signers(path: Path) -> dict[str, dict]:
    """data/signers.csv -> {signer_id: {presentation: M|F|UNKNOWN, dominant_hand, exclude: bool, notes}}."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"signer_id", "presentation", "exclude"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        sid = str(r["signer_id"]).strip().zfill(2)
        pres = str(r["presentation"]).strip().upper() or "UNKNOWN"
        if pres not in {"M", "F", "UNKNOWN"}:
            raise ValueError(f"{path}: signer {sid} has presentation {r['presentation']!r}; use M, F or unknown")
        exclude = str(r["exclude"]).strip() in {"1", "true", "True", "TRUE", "yes"}
        out[sid] = {
            "presentation": pres,
            "dominant_hand": str(r.get("dominant_hand", "")).strip().upper(),
            "exclude": exclude,
            "notes": str(r.get("notes", "")),
        }
    return out


def rank_candidates(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    class_id: int,
    signers: Optional[dict[str, dict]] = None,
) -> pd.DataFrame:
    """Held-out clips of `class_id`, best first.

    predictions.csv has one row per (run, test clip); `sample_idx` indexes the manifest *after*
    dataset.drop_undetected_samples. Clips are ranked by: correct in every run, hands present in >= half the
    sampled frames, then mean target probability.
    """
    p = predictions[predictions["target_class_id"] == class_id]
    g = p.groupby("sample_idx").agg(
        n_runs=("run", "count"),
        all_correct=("correct", "all"),
        mean_target_prob=("target_prob", "mean"),
    ).reset_index()
    g = g.merge(manifest[["signer_id", "video_path", "n_frames", "frac_frames_with_any_hand"]], left_on="sample_idx", right_index=True)
    if signers:
        excluded = g["signer_id"].map(lambda s: bool(signers.get(str(s), {}).get("exclude", False))).astype(bool)
        g = g[~excluded]
    g["presentation"] = g["signer_id"].map(lambda s: (signers or {}).get(str(s), {}).get("presentation", "UNKNOWN"))
    g["hands_ok"] = g["frac_frames_with_any_hand"] >= 0.5
    return g.sort_values(["all_correct", "hands_ok", "mean_target_prob"], ascending=False, kind="stable").reset_index(drop=True)


def pick_two(candidates: pd.DataFrame, n: int = 2) -> list[pd.Series]:
    """Best clip, then the best from a *different* signer; same-signer fallback only if needed."""
    chosen: list[pd.Series] = []
    used: set[str] = set()
    for _, row in candidates.iterrows():
        if len(chosen) == n:
            break
        if str(row["signer_id"]) not in used:
            chosen.append(row)
            used.add(str(row["signer_id"]))
    for _, row in candidates.iterrows():
        if len(chosen) == n:
            break
        if not any(row["sample_idx"] == c["sample_idx"] for c in chosen):
            chosen.append(row)
    return chosen


def pick_balanced(candidates: pd.DataFrame) -> tuple[list[pd.Series], Optional[str]]:
    """Best male-presenting and best female-presenting clip (candidates are already ranked best-first).

    Returns (picks ordered best-first, fallback note). If either presentation has no candidate, falls back to
    pick_two() and the note says which presentation(s) were missing.
    """
    best: dict[str, pd.Series] = {}
    for _, row in candidates.iterrows():
        best.setdefault(row["presentation"], row)
    if "M" in best and "F" in best:
        two = [best["M"], best["F"]]
        order = sorted(range(2), key=lambda i: int(candidates.index[candidates["sample_idx"] == two[i]["sample_idx"]][0]))
        return [two[i] for i in order], None
    missing = [p for p in ("M", "F") if p not in best]
    return pick_two(candidates), "no valid " + "/".join(missing) + " clip; used best two"


def remap_raw_path(raw_root: Path, manifest_video_path: str) -> Path:
    """manifest.csv stores the Colab path used at extraction; rebase it onto --raw-videos
    (<root>/<class folder>/<file>)."""
    p = Path(manifest_video_path)
    return Path(raw_root) / p.parent.name / p.name


# --------------------------------------------------------------------------
# ffmpeg
# --------------------------------------------------------------------------


def probe_fps(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    num, _, den = out.partition("/")
    return float(num) / float(den or 1)


def encode_clip(src: Path, start_s: float, end_s: float, dst: Path, height: int = 480, crf: int = 27) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{end_s - start_s:.3f}",
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


@dataclass
class Plan:
    word: str
    class_id: int
    example: int
    row: pd.Series
    src: Path
    start_s: float
    end_s: float
    presentation: str = "UNKNOWN"
    fallback: Optional[str] = None


def load_dataset_tables(features_dir: Path, min_hand_frac: float):
    """(filtered manifest, filtered mask) exactly as the training/eval pipeline sees them."""
    sys.path.insert(0, str(REPO))
    from dataset import drop_undetected_samples  # imported lazily: pulls in torch

    manifest = pd.read_csv(features_dir / "manifest.csv", dtype={"signer_id": str})
    mask = np.load(features_dir / "mask.npy").astype(np.uint8)
    feats_path = features_dir / "features.npy"
    features = np.load(feats_path, mmap_mode="r") if feats_path.exists() else np.zeros((len(mask), 1), dtype=np.float32)
    _, mask_f, manifest_f = drop_undetected_samples(features, mask, manifest, min_hand_frac=min_hand_frac)
    return manifest_f, mask_f


def build_plans(
    args,
    vocab: dict[int, str],
    manifest: pd.DataFrame,
    mask: np.ndarray,
    predictions: pd.DataFrame,
    fps_of,
    signers: Optional[dict[str, dict]] = None,
) -> tuple[list[Plan], list[str]]:
    if args.classes:
        wanted = {int(c) for c in args.classes}
    elif args.all:
        wanted = set(vocab)
    else:
        have = words_with_clips(Path(args.existing_videos))
        wanted = {cid for cid, w in vocab.items() if w not in have}
    plans: list[Plan] = []
    warnings: list[str] = []
    for cid in sorted(wanted):
        word = vocab[cid]
        cands = rank_candidates(predictions, manifest, cid, signers)
        if len(cands) < 2:
            warnings.append(f"{word} (class {cid}): only {len(cands)} usable held-out clip(s) (after excluding signers); skipped")
            continue
        fallback = None
        if getattr(args, "balance_presentation", False):
            picks, fallback = pick_balanced(cands)
        else:
            picks = pick_two(cands)
        if len({str(r['signer_id']) for r in picks}) < 2:
            warnings.append(f"{word}: both clips are from signer {picks[0]['signer_id']} (no second signer available)")
        for r in picks:
            if not r["all_correct"]:
                warnings.append(f"{word}: clip {r['video_path'].split('/')[-1]} was not correct in every run (best available)")
        for i, r in enumerate(picks, start=1):
            src = remap_raw_path(Path(args.raw_videos), r["video_path"])
            fps = fps_of(src, r)
            start, end = active_window(mask[int(r["sample_idx"])], int(r["n_frames"]), fps, args.pad, args.min_seconds)
            plans.append(Plan(word, cid, i, r, src, start, end, r["presentation"], fallback))
    return plans, warnings


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-videos", required=True, help="KSL_Raw_Videos root (<class_id>/<NN>_<class_id>.MP4)")
    p.add_argument("--features-dir", required=True, help="extract_landmarks.py output dir (manifest.csv, mask.npy[, features.npy])")
    p.add_argument("--predictions", required=True, help="predictions.csv from scripts/calibrate_thresholds.py")
    p.add_argument("--out", required=True, help="Output folder (e.g. a Drive folder)")
    p.add_argument("--vocab", default=str(REPO / "src/frontend/src/data/vocab.json"))
    p.add_argument("--existing-videos", default=str(REPO / "src/frontend/public/videos"), help="Classes whose words already have both clips here are skipped")
    p.add_argument("--classes", nargs="*", help="Only these class ids (overrides the missing-clips default)")
    p.add_argument("--all", action="store_true", help="Make clips for every class, even ones that already have demo clips")
    p.add_argument("--signers", default=None, help="Signer table CSV (default: data/signers.csv if present)")
    p.add_argument("--balance-presentation", action="store_true", help="Prefer one M and one F signer per word (needs presentation in the signer table)")
    p.add_argument("--min-hand-frac", type=float, default=0.0, help="Must match the runs' config.json (config f used 0.0)")
    p.add_argument("--pad", type=float, default=0.25, help="Seconds of context kept before/after the signing")
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--crf", type=int, default=27, help="x264 quality (higher = smaller)")
    p.add_argument("--dry-run", action="store_true", help="Print picks and trim windows only (no raw videos/ffmpeg needed)")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.dry_run and not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        print("ffmpeg/ffprobe not found (Colab has them; otherwise `apt-get install ffmpeg`).", file=sys.stderr)
        return 2
    vocab = load_vocab(Path(args.vocab))
    signers = None
    signers_path = Path(args.signers) if args.signers else REPO / "data" / "signers.csv"
    if signers_path.exists():
        signers = load_signers(signers_path)
        excluded = sorted(k for k, v in signers.items() if v["exclude"])
        print(f"Signer table {signers_path}: excluding {', '.join(excluded) or 'none'}")
    elif args.signers:
        print(f"--signers file not found: {signers_path}", file=sys.stderr)
        return 2
    else:
        print(f"WARNING: {signers_path} not found — no signers excluded (e.g. 08). Create it with scripts/signer_contact_sheet.py.")
    if args.balance_presentation:
        have = {v["presentation"] for v in (signers or {}).values() if not v["exclude"]}
        if not {"M", "F"} <= have:
            print("--balance-presentation needs both M and F (non-excluded) signers in the signer table.", file=sys.stderr)
            return 2
    manifest, mask = load_dataset_tables(Path(args.features_dir), args.min_hand_frac)
    predictions = pd.read_csv(args.predictions, dtype={"signer_id": str})
    if predictions["sample_idx"].max() >= len(manifest):
        print(f"predictions.csv refers to sample {predictions['sample_idx'].max()} but the filtered manifest has "
              f"{len(manifest)} rows — check --min-hand-frac / --features-dir match the calibration run.", file=sys.stderr)
        return 2

    def fps_of(src: Path, row: pd.Series) -> float:
        if args.dry_run and not src.exists():
            return 30.0  # nominal; dry-run only
        return probe_fps(src)

    plans, warnings = build_plans(args, vocab, manifest, mask, predictions, fps_of, signers)
    if not plans:
        print("Nothing to do (every class already has clips, or no candidates).")
        for w in warnings:
            print("WARNING:", w)
        return 0

    out = Path(args.out)
    rows = []
    total = 0
    print(f"{'word':<16}{'ex':<4}{'signer':<8}{'pres.':<7}{'p(target)':<11}{'window (s)':<14}source")
    for pl in plans:
        dst = out / f"{pl.word}_example{pl.example}.mp4"
        size = 0
        if not args.dry_run:
            if not pl.src.exists():
                print(f"ERROR: raw video not found: {pl.src}", file=sys.stderr)
                return 1
            encode_clip(pl.src, pl.start_s, pl.end_s, dst, height=args.height, crf=args.crf)
            size = dst.stat().st_size
            total += size
        r = pl.row
        print(f"{pl.word:<16}{pl.example:<4}{r['signer_id']:<8}{pl.presentation[:1]:<7}{r['mean_target_prob']:<11.3f}"
              f"{pl.start_s:5.2f}-{pl.end_s:5.2f}   {'/'.join(pl.src.parts[-2:])}" + (f"  {size/1024:.0f} KB" if size else ""))
        rows.append({
            "word": pl.word, "example": pl.example, "class_id": pl.class_id, "file": dst.name,
            "source_video": "/".join(pl.src.parts[-2:]), "signer_id": r["signer_id"], "presentation": pl.presentation,
            "balance_fallback": pl.fallback or "", "sample_idx": int(r["sample_idx"]),
            "mean_heldout_target_prob": round(float(r["mean_target_prob"]), 4), "n_runs": int(r["n_runs"]),
            "correct_in_all_runs": bool(r["all_correct"]), "start_s": round(pl.start_s, 3), "end_s": round(pl.end_s, 3),
            "bytes": size,
        })
    for w in warnings:
        print("WARNING:", w)
    if args.balance_presentation:
        fb = sorted({(pl.word, pl.fallback) for pl in plans if pl.fallback})
        n_words = len({pl.word for pl in plans})
        print(f"\nPresentation balance: {n_words - len(fb)}/{n_words} words got one M + one F clip.")
        for word, note in fb:
            print(f"  fallback: {word}: {note}")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "demo_clips_manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    (out / "ATTRIBUTION.txt").write_text(ATTRIBUTION)
    print(f"\nWrote {len(rows)} clips ({total/1e6:.1f} MB) + demo_clips_manifest.csv + ATTRIBUTION.txt to {out}")
    print("Next: download the .mp4 files into src/frontend/public/videos/ (names are already <word>_example1/2.mp4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
