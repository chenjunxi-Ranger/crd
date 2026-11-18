"""Evaluate the frozen single formula on the four benchmarks (chunked, resumable).

Row-level benches: AUROC + 95% bootstrap CI.
FOIL: within-pair accuracy (ties = 0.5) + pair bootstrap CI + exact tie rate.

Usage
-----
    python scripts/run_bench.py --bench coco  --split val
    python scripts/run_bench.py --bench foil  --split test      # one-shot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import constants_hash, constants_dict          # noqa: E402
from src.protocol import (RESULTS_DIR, THRESHOLDS, bootstrap_acc,  # noqa: E402
                          bootstrap_auc, load_foil_test, load_foil_val,
                          load_split, write_json)
from src.score import FrozenScorer                                # noqa: E402

CHUNK = 2000


def _row(args, scorer) -> dict:
    df = load_split(args.bench, args.split, args.limit)
    scores = np.concatenate([
        scorer.score(df["path"].iloc[i:i + CHUNK].tolist(),
                     df["caption"].iloc[i:i + CHUNK].tolist())
        for i in range(0, len(df), CHUNK)
    ])
    out = pd.DataFrame({"image_id": df["image_id"], "label": df["label"], "score": scores})
    path = RESULTS_DIR / "bench" / f"{args.bench}_{args.split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    auc, lo, hi = bootstrap_auc(out["label"].to_numpy(), scores, n_boot=args.n_boot)
    payload = {"bench": args.bench, "split": args.split, "n": int(len(out)),
               "auroc": auc, "auroc_ci95": [lo, hi],
               "threshold": THRESHOLDS[args.bench],
               "gap": float(THRESHOLDS[args.bench] - auc),
               "pass": bool(auc >= THRESHOLDS[args.bench]),
               "constants": constants_dict(), "constants_hash": constants_hash(),
               "csv": str(path)}
    write_json(RESULTS_DIR / "bench" / f"{args.bench}_{args.split}.json", payload)
    return payload


def _foil(args, scorer) -> dict:
    df = load_foil_val(args.limit) if args.split == "val" else load_foil_test(args.limit)
    n = len(df)
    sp = np.concatenate([
        scorer.score(df["path"].iloc[i:i + CHUNK].tolist(),
                     df["positive"].iloc[i:i + CHUNK].tolist())
        for i in range(0, n, CHUNK)
    ])
    sn = np.concatenate([
        scorer.score(df["path"].iloc[i:i + CHUNK].tolist(),
                     df["negative"].iloc[i:i + CHUNK].tolist())
        for i in range(0, n, CHUNK)
    ])
    d = sn - sp                       # higher = negative looks more suspicious
    correct = np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))
    acc, lo, hi = bootstrap_acc(correct, n_boot=args.n_boot)
    out = pd.DataFrame({"pair_id": df["pair_id"], "score_pos": sp,
                        "score_neg": sn, "diff": d, "correct": correct})
    path = RESULTS_DIR / "bench" / f"foil_{args.split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    payload = {"bench": "foil", "split": args.split, "n_pairs": int(n),
               "accuracy": acc, "accuracy_ci95": [lo, hi],
               "tie_rate": float((d == 0).mean()),
               "strict_accuracy": float((d > 0).mean()),
               "threshold": THRESHOLDS["foil"],
               "gap": float(THRESHOLDS["foil"] - acc),
               "pass": bool(acc >= THRESHOLDS["foil"]),
               "constants": constants_dict(), "constants_hash": constants_hash(),
               "csv": str(path)}
    write_json(RESULTS_DIR / "bench" / f"foil_{args.split}.json", payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    scorer = FrozenScorer(device=args.device)
    payload = _foil(args, scorer) if args.bench == "foil" else _row(args, scorer)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
