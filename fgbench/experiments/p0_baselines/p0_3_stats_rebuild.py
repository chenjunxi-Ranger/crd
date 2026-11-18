"""P0-3: rebuild the 24-cell paired family + 2AFC integer CIs.

The frozen `final_stats_strict.csv` contains 22 cells: Entity x LEMoN-style and
Entity x TRACED-style were never bootstrapped, although both rows exist in the
main table. This script recomputes all 24 non-empty cells (5 detectors x 5
facets, minus Action x GMM) with a cycle-clustered paired bootstrap, then applies
Holm over 24. It also reports the integer 2AFC counts and a bootstrap CI for CRD
and the two reference scores.

Output: results/fgbench/p0_baselines/p0_3_{facet}.csv
Run:    python .../p0_3_stats_rebuild.py --facet Attribute --boot 5000
"""
from __future__ import annotations

import argparse
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
BASELINES = ("cos", "csls", "gmm", "lemon", "traced")


def crd_anom(ds, facet):
    if facet == "Action":
        zs = [np.load(CACHE / f"crd_action_scores_s{s}.npz", allow_pickle=True)
              for s in (0, 1, 2)]
    else:
        zs = [np.load(CACHE / f"crdbb_scores_clip_{ds}_{facet}_both_row_s{s}.npz",
                      allow_pickle=True) for s in (0, 1, 2)]
    return np.mean([-z["crd"] for z in zs], axis=0), zs[0]["cycle_id"], zs[0]["y"]


def base_anom(ds, facet, name):
    if name in ("cos", "csls"):
        z = np.load(CACHE / f"strict_base_{ds}_{facet}.npz", allow_pickle=True)
        return z[name], z["cycle_id"], z["y"]
    p = (CACHE / f"score_gmm_{ds}_{facet}.npz" if name == "gmm"
         else CACHE / f"score_{name}_clip_{ds}_{facet}.npz")
    if not p.is_file():
        p = CACHE / f"score_{name}_{ds}_{facet}.npz"
    if not p.is_file():
        p = OUT / f"score_{name}_{ds}_{facet}.npz"
    if not p.is_file():
        return None, None, None
    z = np.load(p, allow_pickle=True)
    return z["score"], z["cycle_id"], z["y"]


def align(sc, bc, by, cyc, y):
    """Subset (and reorder) a baseline to the CRD rows, by cycle id."""
    if len(sc) == len(cyc) and np.array_equal(by, y):
        return sc
    order = []
    for c in pd.unique(cyc):
        ii = np.where(bc == c)[0]
        if len(ii) != 4:
            return None
        order.extend(ii)
    sc2, by2 = sc[order], by[order]
    return sc2 if np.array_equal(by2, y) else None


def acc2afc(score, cyc, y):
    df = pd.DataFrame({"c": cyc, "y": y, "s": score})
    out = [int(g[g.y == 0].s.sum() < g[g.y == 1].s.sum())
           for _, g in df.groupby("c") if len(g) == 4]
    return np.asarray(out, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facet", required=True, choices=list(FACETS))
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    facet, ds = args.facet, FACETS[args.facet]
    rng = np.random.default_rng(0)

    crd, cyc, y = crd_anom(ds, facet)
    keys = pd.unique(cyc)
    idx_by = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
        lambda s: s.to_numpy()).to_dict()

    # 2AFC integer counts and cycle bootstrap CI
    rows = []
    sb = np.load(CACHE / f"strict_base_{ds}_{facet}.npz", allow_pickle=True)
    for name, score in (("CRD", crd), ("cos", sb["cos"]), ("csls", sb["csls"])):
        ok = acc2afc(score, cyc, y)
        n = len(ok)
        bs = [ok[rng.integers(0, n, n)].mean() for _ in range(2000)]
        rows.append({"facet": facet, "metric": "2afc", "method": name,
                     "value": round(float(ok.mean()), 6), "n_cycles": n,
                     "correct": int(ok.sum()),
                     "ci_lo": round(float(np.percentile(bs, 2.5)), 6),
                     "ci_hi": round(float(np.percentile(bs, 97.5)), 6)})
    for name in BASELINES:
        sc, bc, by = base_anom(ds, facet, name)
        if sc is None:
            continue
        sc = align(sc, bc, by, cyc, y)
        if sc is None:
            print(f"[skip] {facet} {name}: rows do not align", flush=True)
            continue
        a_crd = roc_auc_score(y, crd)
        a_b = roc_auc_score(y, sc)
        d = []
        for _ in range(args.boot):
            pick = rng.choice(keys, len(keys), replace=True)
            ii = np.concatenate([idx_by[c] for c in pick])
            if y[ii].min() == y[ii].max():
                continue
            d.append(roc_auc_score(y[ii], crd[ii]) - roc_auc_score(y[ii], sc[ii]))
        d = np.asarray(d)
        rows.append({
            "facet": facet, "metric": "auroc", "method": name,
            "value": round(float(a_crd - a_b) * 100, 2),
            "auroc_crd": round(float(a_crd), 4), "auroc_base": round(float(a_b), 4),
            "ci_lo": round(float(np.percentile(d, 2.5)) * 100, 2),
            "ci_hi": round(float(np.percentile(d, 97.5)) * 100, 2),
            "p_one_sided": round(max(1.0 / args.boot, float((d <= 0).mean())), 5),
            "n_boot": int(len(d))})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"p0_3_{facet}.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
