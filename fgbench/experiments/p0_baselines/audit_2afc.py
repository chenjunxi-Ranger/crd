"""P0-2: audit of the 2AFC metric actually used in this project.

Definition found in the code (fgbench/evaluate/strict_main_table.py::metrics and
fgbench/evaluate/cycle_2afc.py):

    for every 4-row reciprocal cycle, with the two clean rows C and the two
    noisy rows N:
        correct  <=>  mean_{c in C} s(c)  >  mean_{n in N} s(n)
    2AFC = fraction of cycles that are correct

This is a CYCLE-LEVEL group comparison, not a same-image candidate comparison.

Consequence, and what this script verifies numerically: any score of the form
z' = z - lam * [a(u) + b(v) + b0] leaves every cycle's diag-minus-cross margin
unchanged, because a cycle contains both images and both captions exactly once,
so a and b contribute identically to the clean and the noisy side:

    sum_C a(u) + sum_C b(v) == sum_N a(u) + sum_N b(v)      (same multisets)

Hence the metric is invariant under CRD and under CSLS (also additive), and is
NOT invariant under centred / ABTT cosine, which change the bilinear form rather
than adding a separable term.

Run:  python fgbench/experiments/p0_baselines/audit_2afc.py
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
FACETS = [("COCO", "Entity"), ("COCO", "Quantity"), ("VG", "Attribute"),
          ("VG", "Action"), ("VG", "Relation")]


def cyc_2afc(y: np.ndarray, s: np.ndarray, cyc: np.ndarray) -> float:
    df = pd.DataFrame({"c": cyc, "y": y, "s": -s})     # -s: higher = cleaner
    acc = [g[g.y == 0].s.sum() < g[g.y == 1].s.sum()
           for _, g in df.groupby("c") if len(g) == 4]
    return float(np.mean(acc)) if acc else float("nan")


def crd_seeds(ds, facet):
    if facet == "Action":
        return [np.load(CACHE / f"crd_action_scores_s{s}.npz", allow_pickle=True)["crd"]
                for s in (0, 1, 2)]
    return [np.load(CACHE / f"crdbb_scores_clip_{ds}_{facet}_both_row_s{s}.npz",
                    allow_pickle=True)["crd"] for s in (0, 1, 2)]


def main() -> int:
    rows = []
    for ds, facet in FACETS:
        b = np.load(OUT / f"{ds}_{facet}_scores.npz", allow_pickle=True)
        y, cyc = b["y"], b["cycle_id"]
        rec = {"facet": facet,
               "auroc_cos": round(float(roc_auc_score(y, -b["raw_cos"])), 4),
               "1_minus_auroc": round(1 - float(roc_auc_score(y, -b["raw_cos"])), 4),
               "2afc_cos": round(cyc_2afc(y, b["raw_cos"], cyc), 4),
               "2afc_centered": round(cyc_2afc(y, b["centered"], cyc), 4),
               "2afc_abtt": round(cyc_2afc(y, b["abtt"], cyc), 4)}
        seeds = [round(cyc_2afc(y, s, cyc), 4) for s in crd_seeds(ds, facet)]
        rec["2afc_crd_seeds"] = seeds
        rec["crd_invariant"] = len(set(seeds) | {rec["2afc_cos"]}) == 1
        rows.append(rec)
        print(rec, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "audit_2afc.csv", index=False, encoding="utf-8-sig")
    print()
    print(df.drop(columns=["2afc_crd_seeds"]).to_string(index=False))
    print("\nCRD invariance holds on every facet:",
          bool(df.crd_invariant.all()))
    print("centred / ABTT change 2AFC on:",
          [r["facet"] for _, r in df.iterrows()
           if abs(r["2afc_centered"] - r["2afc_cos"]) > 1e-9
           or abs(r["2afc_abtt"] - r["2afc_cos"]) > 1e-9])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
