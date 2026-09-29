#!/usr/bin/env python3
"""Export per-class held-out accuracy for the frontend ("tricky sign" tags).

Reads the probs.npz written by scripts/calibrate_thresholds.py (pooled held-out
signer_kfold predictions, TTA mirror) and writes
src/frontend/src/data/class_accuracy.json:

    {"source": ..., "n_predictions": N,
     "classes": {"<original class id>": {"accuracy": 0.53, "n": 57,
                                         "top_confusions": [[<class id>, <count>], ...]}}}

Usage:
    python scripts/export_class_accuracy.py --probs runs/calibration/probs.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def class_accuracy(probs: np.ndarray, y: np.ndarray, class_ids: np.ndarray, top_confusions: int = 3) -> dict:
    top1 = probs.argmax(1)
    out = {}
    for d, cid in enumerate(class_ids):
        m = y == d
        n = int(m.sum())
        wrong = np.bincount(top1[m][top1[m] != d], minlength=len(class_ids))
        conf = [[int(class_ids[o]), int(wrong[o])] for o in np.argsort(-wrong, kind="stable")[:top_confusions] if wrong[o] > 0]
        out[str(int(cid))] = {
            "accuracy": round(float((top1[m] == d).mean()), 4) if n else None,
            "n": n,
            "top_confusions": conf,
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probs", required=True, help="probs.npz from calibrate_thresholds.py")
    p.add_argument("--out", default=str(REPO / "src/frontend/src/data/class_accuracy.json"))
    args = p.parse_args(argv)
    z = np.load(args.probs)
    classes = class_accuracy(z["probs"], z["y"], z["class_ids"])
    payload = {
        "source": "scripts/calibrate_thresholds.py held-out signer_kfold predictions (config f, TTA mirror), pooled over runs",
        "n_predictions": int(len(z["y"])),
        "classes": classes,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    worst = sorted(classes.items(), key=lambda kv: kv[1]["accuracy"])[:5]
    print(f"Wrote {args.out}; weakest: " + ", ".join(f"{k}={v['accuracy']:.1%}" for k, v in worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
