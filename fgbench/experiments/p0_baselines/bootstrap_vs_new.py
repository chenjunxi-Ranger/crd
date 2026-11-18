"""P0-1 (D): paired cycle-clustered bootstrap, CRD vs the two new baselines.

Ten core comparisons (5 facets x 2 baselines), one Holm family.
CRD test scores are averaged over the three training seeds before resampling,
so the comparison is CRD-as-deployed against a parameter-free baseline.

Run:  python fgbench/experiments/p0_baselines/bootstrap_vs_new.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "p0_baselines"
FACETS = [("COCO", "Entity"), ("COCO", "Quantity"), ("VG", "Attribute"),
          ("VG", "Action"), ("VG", "Relation")]
N_BOOT = 1000


def crd_scores(ds: str, facet: str) -> np.ndarray:
    """Mean over the three seeds of the calibrated score (similarity space)."""
    if facet == "Action":
        zs = [np.load(CACHE / f"crd_action_scores_s{s}.npz", allow_pickle=True)
              for s in (0, 1, 2)]
        return np.mean([z["crd"] for z in zs], axis=0), zs[0]["cycle_id"]
    zs = [np.load(CACHE / f"crdbb_scores_clip_{ds}_{facet}_both_row_s{s}.npz",
                  allow_pickle=True) for s in (0, 1, 2)]
    return np.mean([z["crd"] for z in zs], axis=0), zs[0]["cycle_id"]


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m, out, run = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def main() -> int:
    rng = np.random.default_rng(0)
    recs = []
    for ds, facet in FACETS:
        b = np.load(OUT / f"{ds}_{facet}_scores.npz", allow_pickle=True)
        crd, cyc = crd_scores(ds, facet)
        assert len(cyc) == len(b["cycle_id"]) and (cyc == b["cycle_id"]).all(), facet
        y = b["y"]
        for name, key in (("centered", "centered"), ("abtt", "abtt")):
            base = b[key]
            a_crd = roc_auc_score(y, -crd)
            a_base = roc_auc_score(y, -base)
            # resample cycles, not rows
            cyc_ids = pd.unique(b["cycle_id"])
            idx_by_cyc = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
                lambda s: s.to_numpy()).to_dict()
            deltas = []
            for _ in range(N_BOOT):
                pick = rng.choice(cyc_ids, len(cyc_ids), replace=True)
                idx = np.concatenate([idx_by_cyc[c] for c in pick])
                yy = y[idx]
                if yy.min() == yy.max():
                    continue
                deltas.append(roc_auc_score(yy, -crd[idx]) -
                              roc_auc_score(yy, -base[idx]))
            deltas = np.asarray(deltas)
            recs.append({
                "facet": facet, "baseline": name,
                "auroc_crd": round(float(a_crd), 4),
                "auroc_base": round(float(a_base), 4),
                "delta_pp": round(float(a_crd - a_base) * 100, 2),
                "ci_lo": round(float(np.percentile(deltas, 2.5)) * 100, 2),
                "ci_hi": round(float(np.percentile(deltas, 97.5)) * 100, 2),
                "p_one_sided": float((deltas <= 0).mean()),
                "n_boot": int(len(deltas)),
            })
            print(recs[-1], flush=True)

    df = pd.DataFrame(recs)
    df["p_holm"] = holm(df.p_one_sided.to_numpy())
    df["significant"] = df.p_holm < 0.05
    df.to_csv(OUT / "bootstrap_vs_new.csv", index=False, encoding="utf-8-sig")
    print()
    print(df.to_string(index=False))
    print("\nfamily size:", len(df), " significant after Holm:", int(df.significant.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
