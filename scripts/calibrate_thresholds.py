#!/usr/bin/env python3
"""Calibrate the browser's confidence thresholds (CORRECT_MIN / CLOSE_MIN /
CONFUSION_MIN in src/frontend/src/lib/inference/config.ts) from held-out
signer predictions.

For each `signer_kfold` checkpoint dir (the fold_*/ dirs train.py wrote) this
re-derives the exact signer split evaluate.py uses, runs every held-out test
clip through the model (optionally with mirror TTA, i.e. the softmax average
the browser uses), and saves one row per test clip:

    predictions.csv   run, fold, sample_idx, signer_id, target_class_id,
                      target_word, target_prob, target_rank, top1_class_id,
                      top1_word, top1_prob, top2_*, top3_*, correct
    probs.npz         full (N, K) averaged probabilities + targets, so the
                      threshold sweep can be re-run without the model/data
                      (use --from-npz).

It then prints, for a range of thresholds, what each gate would do, and
suggests values. The browser grading order is (see grading.ts):

    correct   target is top-1 AND p(target) >= CORRECT_MIN
    close     target in top-3 AND p(target) >= CLOSE_MIN
    confused  top-1 != target AND p(top-1) >= CONFUSION_MIN   -> names top-1
    otherwise incorrect (no word named)

How to read the tables (all held-out clips are correctly signed, so every
model error on them is a *false accusation* if we name a word):

  CORRECT_MIN   "correct accepted": share of correct top-1 predictions that
                still pass the gate (higher = fewer good signs rejected).
                "false accept": for a signer practising word c who actually
                signs some other word, how often we'd say "Perfect!"
                (attempts = every held-out clip x every wrong target word).
  CONFUSION_MIN "wrong shown": share of the model's wrong predictions that
                would be shown as "That looked like X" (each is a correct
                sign we'd mis-name). "naming precision": among clips where we
                would name a word, the share where the named word really is
                the word signed (this is what a user who signed a *different*
                word would experience). "coverage": share of all clips that
                get a name at all.
  CLOSE_MIN     "close kept": share of genuine attempts that miss top-1 but
                have the target in top-3 and would still be called "close".
                "false close": share of (clip x wrong target) attempts that
                would be called "close".

Usage (Colab, from the repo root, after evaluating with the same features):
    python scripts/calibrate_thresholds.py \\
        --features features_v2/features.npy --mask features_v2/mask.npy \\
        --manifest features_v2/manifest.csv --tta-mirror \\
        --checkpoint-dir runs/f_mirror_norm runs/f_seed1 runs/f_seed2 \\
        --out-dir runs/calibration

    # re-run just the analysis (no torch/data needed):
    python scripts/calibrate_thresholds.py --from-npz runs/calibration/probs.npz
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Grading logic: a NumPy mirror of src/frontend/src/lib/inference/grading.ts
# (same check order). Kept dependency-free so it can be unit-tested anywhere.
# --------------------------------------------------------------------------

CORRECT, CLOSE, CONFUSED, INCORRECT = "correct", "close", "confused", "incorrect"


def grade_attempt(
    probs: np.ndarray,
    target: int,
    correct_min: float,
    close_min: float,
    confusion_min: float,
    close_top_k: int = 3,
) -> str:
    """Grade one attempt (probs: (K,), target: dense class index)."""
    order = np.argsort(-probs, kind="stable")
    top1 = int(order[0])
    p_target = float(probs[target])
    if top1 == target and p_target >= correct_min:
        return CORRECT
    if target in order[:close_top_k] and p_target >= close_min:
        return CLOSE
    if top1 != target and float(probs[top1]) >= confusion_min:
        return CONFUSED
    return INCORRECT


def threshold_grid(lo: float = 0.05, hi: float = 0.95, step: float = 0.05) -> np.ndarray:
    return np.round(np.arange(lo, hi + 1e-9, step), 4)


def sweep_correct_min(probs: np.ndarray, y: np.ndarray, grid: np.ndarray) -> list[dict]:
    """probs (N,K), y (N,) dense targets."""
    n, k = probs.shape
    top1 = probs.argmax(1)
    p1 = probs.max(1)
    is_correct = top1 == y
    n_correct = max(int(is_correct.sum()), 1)
    rows = []
    for t in grid:
        accepted = int((is_correct & (p1 >= t)).sum())
        # Impostor attempts: clip of word y practised as target c != y. It is graded "correct" only when the
        # top-1 is c (a model error, top1 != y) and p(top1) >= t: exactly one such target per wrong clip.
        false_accepts = int(((~is_correct) & (p1 >= t)).sum())
        rows.append(
            {
                "t": float(t),
                "correct_accepted": accepted / n_correct,
                "false_accept": false_accepts / (n * (k - 1)),
            }
        )
    return rows


def sweep_confusion_min(probs: np.ndarray, y: np.ndarray, grid: np.ndarray) -> list[dict]:
    n = len(y)
    top1 = probs.argmax(1)
    p1 = probs.max(1)
    wrong = top1 != y
    n_wrong = max(int(wrong.sum()), 1)
    rows = []
    for t in grid:
        named = p1 >= t  # would be named whenever the target isn't top-1 (impostor attempts)
        n_named = int(named.sum())
        rows.append(
            {
                "t": float(t),
                "wrong_shown": int((wrong & named).sum()) / n_wrong,
                "wrong_shown_share_of_all": int((wrong & named).sum()) / n,
                "naming_precision": (int((~wrong & named).sum()) / n_named) if n_named else float("nan"),
                "coverage": n_named / n,
            }
        )
    return rows


def sweep_close_min(probs: np.ndarray, y: np.ndarray, grid: np.ndarray, top_k: int = 3) -> list[dict]:
    n, k = probs.shape
    order = np.argsort(-probs, axis=1, kind="stable")[:, :top_k]
    top1 = order[:, 0]
    in_topk_not_top1 = np.array([(y[i] in order[i]) and (top1[i] != y[i]) for i in range(n)])
    p_target = probs[np.arange(n), y]
    n_near = max(int(in_topk_not_top1.sum()), 1)
    rows = []
    for t in grid:
        kept = int((in_topk_not_top1 & (p_target >= t)).sum())
        # Impostor: wrong targets c inside the top-k with p_c >= t (counting every such c per clip).
        mask_topk = np.zeros_like(probs, dtype=bool)
        np.put_along_axis(mask_topk, order, True, axis=1)
        mask_topk[np.arange(n), y] = False  # exclude the genuine target
        false_close = int((mask_topk & (probs >= t)).sum())
        rows.append(
            {
                "t": float(t),
                "close_kept": kept / n_near,
                "false_close": false_close / (n * (k - 1)),
            }
        )
    return rows


def suggest(
    correct_rows: list[dict],
    close_rows: list[dict],
    confusion_rows: list[dict],
    max_false_accept: float = 0.005,
    max_false_close: float = 0.02,
    min_naming_precision: float = 0.95,
) -> dict:
    """Pick thresholds by explicit, overridable rules (see CLI flags)."""
    # CORRECT_MIN: the lowest t whose false-accept rate is <= max_false_accept. (Rationale: rejecting a correct sign
    # is costly to learners, so start from the most permissive gate that still keeps false "Perfect!"s rare. On
    # the KSL calibration the false-accept curve is nearly flat, so this rule alone lands at the bottom of the
    # grid; the shipped value is a judgement call informed by learner testing -- see RESULTS.md.)
    notes = []
    ok = [r for r in correct_rows if r["false_accept"] <= max_false_accept]
    if not ok:
        notes.append(f"No CORRECT_MIN gets false-accept <= {max_false_accept:.1%}; using the highest grid value.")
    correct_min = min((r["t"] for r in ok), default=correct_rows[-1]["t"])
    # CLOSE_MIN: the lowest t whose false-close rate is <= max_false_close.
    ok = [r for r in close_rows if r["false_close"] <= max_false_close]
    if not ok:
        notes.append(f"No CLOSE_MIN gets false-close <= {max_false_close:.1%}; using the highest grid value.")
    close_min = min((r["t"] for r in ok), default=close_rows[-1]["t"])
    # CONFUSION_MIN: the lowest t whose naming precision is >= min_naming_precision.
    ok = [r for r in confusion_rows if r["naming_precision"] == r["naming_precision"] and r["naming_precision"] >= min_naming_precision]
    if not ok:
        notes.append(
            f"No CONFUSION_MIN reaches naming precision >= {min_naming_precision:.0%}; using the highest grid value. "
            "Consider never naming words (set CONFUSION_MIN above 1) or relaxing --min-naming-precision."
        )
    confusion_min = min((r["t"] for r in ok), default=confusion_rows[-1]["t"])
    if close_min >= correct_min:
        notes.append(f"CLOSE_MIN ({close_min}) >= CORRECT_MIN ({correct_min}): 'close' would be unreachable for top-1 hits; lower CLOSE_MIN.")
    if confusion_min < correct_min:
        notes.append(f"CONFUSION_MIN ({confusion_min}) < CORRECT_MIN ({correct_min}): unusual; check the tables.")
    return {"CORRECT_MIN": correct_min, "CLOSE_MIN": close_min, "CONFUSION_MIN": confusion_min, "notes": notes}


def outcome_distribution(probs: np.ndarray, y: np.ndarray, cfg: dict, genuine: bool, rng_seed: int = 0) -> dict:
    """Share of each grade for genuine attempts (target = y) or, for impostors, every clip x every wrong target."""
    n, k = probs.shape
    counts = {CORRECT: 0, CLOSE: 0, CONFUSED: 0, INCORRECT: 0}
    total = 0
    for i in range(n):
        targets = [int(y[i])] if genuine else [c for c in range(k) if c != y[i]]
        for c in targets:
            counts[grade_attempt(probs[i], c, cfg["CORRECT_MIN"], cfg["CLOSE_MIN"], cfg["CONFUSION_MIN"])] += 1
            total += 1
    return {g: v / total for g, v in counts.items()}


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------


def _fmt(x: float, pct: bool = True) -> str:
    if x != x:
        return "  n/a"
    return f"{x * 100:5.1f}%" if pct else f"{x:.4f}"


def print_tables(correct_rows, close_rows, confusion_rows) -> None:
    print("\nCORRECT_MIN  (top-1 must also have p >= t)")
    print(f"  {'t':>5}  {'correct accepted':>17}  {'false accept':>13}")
    for r in correct_rows:
        print(f"  {r['t']:>5.2f}  {_fmt(r['correct_accepted']):>17}  {_fmt(r['false_accept'], True):>13}")
    print("\nCLOSE_MIN  (target in top-3 and p >= t)")
    print(f"  {'t':>5}  {'close kept':>11}  {'false close':>12}")
    for r in close_rows:
        print(f"  {r['t']:>5.2f}  {_fmt(r['close_kept']):>11}  {_fmt(r['false_close']):>12}")
    print("\nCONFUSION_MIN  (wrong top-1 with p >= t is named: 'That looked like X')")
    print(f"  {'t':>5}  {'wrong shown':>12}  {'of all clips':>13}  {'naming precision':>17}  {'coverage':>9}")
    for r in confusion_rows:
        print(
            f"  {r['t']:>5.2f}  {_fmt(r['wrong_shown']):>12}  {_fmt(r['wrong_shown_share_of_all']):>13}  "
            f"{_fmt(r['naming_precision']):>17}  {_fmt(r['coverage']):>9}"
        )


def analyze(probs: np.ndarray, y: np.ndarray, args) -> dict:
    grid = threshold_grid(args.grid_min, args.grid_max, args.grid_step)
    correct_rows = sweep_correct_min(probs, y, grid)
    close_rows = sweep_close_min(probs, y, grid)
    confusion_rows = sweep_confusion_min(probs, y, grid)
    n = len(y)
    top1_acc = float((probs.argmax(1) == y).mean())
    top3_acc = float(np.mean([y[i] in np.argsort(-probs[i])[:3] for i in range(n)]))
    print(f"\nPooled held-out predictions: n={n}, classes={probs.shape[1]}, top-1={top1_acc:.4f}, top-3={top3_acc:.4f}")
    print_tables(correct_rows, close_rows, confusion_rows)
    sug = suggest(
        correct_rows, close_rows, confusion_rows,
        max_false_accept=args.max_false_accept,
        max_false_close=args.max_false_close,
        min_naming_precision=args.min_naming_precision,
    )
    print("\nSuggested thresholds (rules: CORRECT_MIN = lowest t with false-accept "
          f"<={args.max_false_accept:.1%}; CLOSE_MIN = lowest t with false-close "
          f"<={args.max_false_close:.1%}; CONFUSION_MIN = lowest t with naming precision >={args.min_naming_precision:.0%})")
    for k_ in ("CORRECT_MIN", "CLOSE_MIN", "CONFUSION_MIN"):
        print(f"  {k_:<14} {sug[k_]:.2f}")
    for note in sug["notes"]:
        print(f"  NOTE: {note}")
    for label, cfg in (("suggested", sug), ("current placeholders", {"CORRECT_MIN": 0.40, "CLOSE_MIN": 0.15, "CONFUSION_MIN": 0.60})):
        gen = outcome_distribution(probs, y, cfg, genuine=True)
        imp = outcome_distribution(probs, y, cfg, genuine=False)
        print(f"\nOutcome mix with {label} thresholds ({cfg['CORRECT_MIN']:.2f}/{cfg['CLOSE_MIN']:.2f}/{cfg['CONFUSION_MIN']:.2f}):")
        print("  genuine attempts (signed the target):  " + "  ".join(f"{g}={_fmt(v)}" for g, v in gen.items()))
        print("  impostor attempts (signed another):    " + "  ".join(f"{g}={_fmt(v)}" for g, v in imp.items()))
    return sug


# --------------------------------------------------------------------------
# Collecting predictions (needs torch + the extracted dataset)
# --------------------------------------------------------------------------


def _class_names(path: Optional[str]) -> dict[int, str]:
    p = Path(path) if path else REPO / "class_label.p"
    if p.exists():
        with open(p, "rb") as f:
            return {int(k): str(v) for k, v in pickle.load(f).items()}
    return {}


def collect_run(run_dir: Path, args, names: dict[int, str]):
    """Returns (rows, probs (N,K), y (N,), class_ids (K,)) for one signer_kfold run."""
    import torch  # noqa: F401  (imported lazily so --from-npz works without torch)

    sys.path.insert(0, str(REPO))
    import transforms
    from dataset import check_signer_partition, load_manifest_dataset, signer_kfold_splits
    from evaluate import _resolve_against_config, load_fold, predict_probs

    fold_dirs = sorted(run_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        raise SystemExit(f"No fold_* subdirectories under {run_dir}")
    first_config, _, _ = load_fold(fold_dirs[0])
    if first_config["split_mode"] != "signer_kfold":
        raise SystemExit(f"{run_dir}: split_mode={first_config['split_mode']!r}; calibration needs signer_kfold held-out folds")
    min_hand_frac = _resolve_against_config(None, first_config, "min_hand_frac", 0.0, fold_dirs[0])
    normalize_body = _resolve_against_config(None, first_config, "normalize_body", False, fold_dirs[0])
    trim_idle = _resolve_against_config(None, first_config, "trim_idle", False, fold_dirs[0])
    trim_thr = _resolve_against_config(None, first_config, "trim_idle_motion_threshold", transforms.DEFAULT_TRIM_MOTION_THRESHOLD, fold_dirs[0])

    features, mask, manifest = load_manifest_dataset(args.features, args.mask, args.manifest, min_hand_frac=min_hand_frac)
    labels = manifest["class_id"].to_numpy()
    signer_ids = manifest["signer_id"].to_numpy()
    if normalize_body:
        features = transforms.normalize_body_bulk(features, mask)
    if trim_idle:
        features, mask = transforms.trim_idle_bulk(features, mask, motion_threshold=trim_thr)

    splits = signer_kfold_splits(labels, signer_ids, n_splits=len(fold_dirs), seed=first_config["seed"])
    check_signer_partition(splits, signer_ids)

    rows, all_probs, all_y = [], [], []
    class_ids = None
    for fold_dir, split in zip(fold_dirs, splits):
        config, maps, model = load_fold(fold_dir)
        assert maps["label_map"] == split.label_map, f"{fold_dir}: label_map mismatch vs rebuilt split"
        ids = np.array([split.reverse_label_map[i] for i in range(len(split.reverse_label_map))])
        class_ids = ids if class_ids is None else class_ids
        assert (class_ids == ids).all(), "runs/folds disagree on the class set"
        probs = predict_probs(model, features, split.test_idx, mask=mask, tta_mirror=args.tta_mirror).numpy().astype(np.float64)
        y = np.array([split.label_map[int(labels[i])] for i in split.test_idx])
        order = np.argsort(-probs, axis=1, kind="stable")
        for j, idx in enumerate(split.test_idx):
            top = order[j, :3]
            row = {
                "run": run_dir.name, "fold": split.fold, "sample_idx": int(idx), "signer_id": str(signer_ids[idx]),
                "target_class_id": int(ids[y[j]]), "target_word": names.get(int(ids[y[j]]), ""),
                "target_prob": float(probs[j, y[j]]), "target_rank": int(np.where(order[j] == y[j])[0][0]) + 1,
                "correct": bool(top[0] == y[j]),
            }
            for r, c in enumerate(top, start=1):
                row[f"top{r}_class_id"] = int(ids[c])
                row[f"top{r}_word"] = names.get(int(ids[c]), "")
                row[f"top{r}_prob"] = float(probs[j, c])
            rows.append(row)
        all_probs.append(probs)
        all_y.append(y)
        print(f"  {run_dir.name}/{fold_dir.name}: n_test={len(y)}, top-1={(probs.argmax(1) == y).mean():.4f}")
    return rows, np.concatenate(all_probs), np.concatenate(all_y), class_ids


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features"); p.add_argument("--mask"); p.add_argument("--manifest")
    p.add_argument("--checkpoint-dir", nargs="+", help="One or more signer_kfold run dirs (each containing fold_*/)")
    p.add_argument("--tta-mirror", action="store_true", help="Average softmax over each clip and its mirror (matches the browser)")
    p.add_argument("--out-dir", default="calibration", help="Where predictions.csv and probs.npz are written")
    p.add_argument("--class-labels", default=None, help="Pickle {class_id: word} (default: class_label.p in the repo root)")
    p.add_argument("--from-npz", default=None, help="Skip model/data: analyse a probs.npz written by an earlier run")
    p.add_argument("--grid-min", type=float, default=0.05); p.add_argument("--grid-max", type=float, default=0.95); p.add_argument("--grid-step", type=float, default=0.05)
    p.add_argument("--max-false-accept", type=float, default=0.005)
    p.add_argument("--max-false-close", type=float, default=0.02)
    p.add_argument("--min-naming-precision", type=float, default=0.95)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.from_npz:
        z = np.load(args.from_npz)
        analyze(z["probs"], z["y"], args)
        return 0
    if not (args.features and args.mask and args.manifest and args.checkpoint_dir):
        print("Provide --features/--mask/--manifest and --checkpoint-dir, or --from-npz", file=sys.stderr)
        return 2
    names = _class_names(args.class_labels)
    all_rows, probs_l, y_l, class_ids = [], [], [], None
    for d in args.checkpoint_dir:
        print(f"Collecting held-out predictions for {d} (tta_mirror={args.tta_mirror})")
        rows, probs, y, ids = collect_run(Path(d), args, names)
        if class_ids is not None and not (class_ids == ids).all():
            raise SystemExit("Runs use different class sets; cannot pool")
        class_ids = ids
        all_rows += rows; probs_l.append(probs); y_l.append(y)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    probs, y = np.concatenate(probs_l), np.concatenate(y_l)
    np.savez_compressed(out / "probs.npz", probs=probs, y=y, class_ids=class_ids)
    print(f"Wrote {out/'predictions.csv'} ({len(all_rows)} rows) and {out/'probs.npz'}")
    analyze(probs, y, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
