"""P0-4: translate AUROC into a cleaning decision.

For each facet, rank pairs by anomaly score and report precision and recall in
the most suspicious 1/5/10/20% of pairs, for CRD, cosine and CSLS. A paired
cycle-clustered bootstrap gives the CRD - cosine difference at 10%.

Output: results/fgbench/p0_baselines/p0_4_precision.csv
Run:    python fgbench/experiments/p0_baselines/p0_4_precision.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "p0_baselines"
FACETS = {"Entity": "COCO", "Quantity": "COCO", "Attribute": "VG",
          "Action": "VG", "Relation": "VG"}
KS = (0.01, 0.05, 0.10, 0.20)


def crd_anom(ds, facet):
    if facet == "Action":
        zs = [np.load(CACHE / f"crd_action_scores_s{s}.npz", allow_pickle=True)
              for s in (0, 1, 2)]
    else:
        zs = [np.load(CACHE / f"crdbb_scores_clip_{ds}_{facet}_both_row_s{s}.npz",
                      allow_pickle=True) for s in (0, 1, 2)]
    return np.mean([-z["crd"] for z in zs], axis=0), zs[0]["cycle_id"], zs[0]["y"]


def prec_recall(score, y, frac):
    n = max(1, int(round(frac * len(y))))
    top = np.argsort(-score)[:n]
    tp = int(y[top].sum())
    return tp / n, tp / max(1, int(y.sum()))


def main() -> int:
    rng = np.random.default_rng(0)
    rows = []
    for facet, ds in FACETS.items():
        crd, cyc, y = crd_anom(ds, facet)
        sb = np.load(CACHE / f"strict_base_{ds}_{facet}.npz", allow_pickle=True)
        scores = {"CRD": crd, "cos": sb["cos"], "CSLS": sb["csls"]}

        keys = pd.unique(cyc)
        idx_by = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
            lambda s: s.to_numpy()).to_dict()
        for name, sc in scores.items():
            for k in KS:
                p, r = prec_recall(sc, y, k)
                rows.append({"facet": facet, "method": name, "k": k,
                             "precision": round(p, 4), "recall": round(r, 4),
                             "n": int(len(y))})
            # 10% paired difference vs cosine
        d = []
        n10 = max(1, int(round(0.10 * len(y))))
        for _ in range(3000):
            pick = rng.choice(keys, len(keys), replace=True)
            ii = np.concatenate([idx_by[c] for c in pick])
            if y[ii].min() == y[ii].max():
                continue
            p_c = prec_recall(crd[ii], y[ii], 0.10)[0]
            p_b = prec_recall(sb["cos"][ii], y[ii], 0.10)[0]
            d.append(p_c - p_b)
        d = np.asarray(d)
        rows.append({"facet": facet, "method": "CRD-cos@10", "k": 0.10,
                     "precision": round(float(d.mean()), 4),
                     "recall": np.nan, "n": int(len(y)),
                     "ci_lo": round(float(np.percentile(d, 2.5)), 4),
                     "ci_hi": round(float(np.percentile(d, 97.5)), 4),
                     "p_one_sided": round(max(1 / 3000, float((d <= 0).mean())), 4)})
        print(facet, "CRD-cos@10", round(float(d.mean()), 4), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p0_4_precision.csv", index=False, encoding="utf-8-sig")
    print(df[df.method.isin(["CRD", "cos", "CSLS"])].pivot_table(
        index=["facet", "k"], columns="method", values="precision").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
