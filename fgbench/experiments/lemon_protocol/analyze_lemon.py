"""Paired test for the LEMoN-protocol experiment: CRD vs raw cosine.

Rows are not independent: several captions share one image, so the bootstrap
resamples IMAGES, not rows (the analogue of cycle clustering on FGBench).

    python fgbench/experiments/lemon_protocol/analyze_lemon.py --frac 0.2 --frac 0.4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parents[3] / "results" / "fgbench" / "lemon_protocol"
N_BOOT = 1000


def auc(y, x):
    return float(roc_auc_score(y, -x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, action="append", required=True)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for frac in a.frac:
        zs = sorted(OUT.glob(f"scores_frac{frac:.2f}_s*.npz"))
        if not zs:
            print(f"missing scores for frac={frac}")
            continue
        arrs = [np.load(z, allow_pickle=True) for z in zs]
        y = arrs[0]["y"]
        img = arrs[0]["image_id"]
        cos = np.mean([z["cos"] for z in arrs], axis=0)
        crd = np.mean([z["crd"] for z in arrs], axis=0)   # seed-averaged score

        by_img = pd.Series(np.arange(len(y))).groupby(img).apply(
            lambda s: s.to_numpy()).to_dict()
        ids = list(by_img)
        deltas = []
        for _ in range(N_BOOT):
            pick = rng.choice(ids, len(ids), replace=True)
            idx = np.concatenate([by_img[i] for i in pick])
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            deltas.append(auc(yy, crd[idx]) - auc(yy, cos[idx]))
        deltas = np.asarray(deltas)
        rows.append({
            "frac": frac, "n_test": int(len(y)), "n_images": int(len(ids)),
            "auroc_cos": round(auc(y, cos), 4),
            "auroc_crd": round(auc(y, crd), 4),
            "delta_pp": round((auc(y, crd) - auc(y, cos)) * 100, 2),
            "ci_lo": round(float(np.percentile(deltas, 2.5)) * 100, 2),
            "ci_hi": round(float(np.percentile(deltas, 97.5)) * 100, 2),
            "p_better": float((deltas <= 0).mean()),
        })
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bootstrap.csv", index=False, encoding="utf-8-sig")
    print()
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
