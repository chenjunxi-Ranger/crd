"""P0-2 (B): paired cycle-clustered bootstrap, CRD vs the supervised probes.

Ten comparisons (5 facets x 2 probes: LR on [u;v;u*v], MLP on [u;v;u*v]),
one Holm family.  CRD test scores are the three-seed mean of the calibrated
score (the same convention as `bootstrap_vs_new.py` and the paper's statistics).

Run:  python fgbench/experiments/p0_baselines/bootstrap_probe.py
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
PROBES = [("lr_uvdot", "LR [u;v;u*v]"), ("mlp_uvdot", "MLP [u;v;u*v]")]


def crd_scores(ds: str, facet: str):
    """Three-seed mean CRD score (similarity space) and its cycle ids."""
    if facet == "Action":
        zs = [np.load(CACHE / f"crd_action_scores_s{s}.npz", allow_pickle=True)
              for s in (0, 1, 2)]
    else:
        zs = [np.load(CACHE / f"crdbb_scores_clip_{ds}_{facet}_both_row_s{s}.npz",
                      allow_pickle=True) for s in (0, 1, 2)]
    return np.mean([z["crd"] for z in zs], axis=0), zs[0]["cycle_id"], zs[0]["y"]


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
        z = np.load(OUT / f"probe_scores_{ds}_{facet}.npz", allow_pickle=True)
        cyc = z["cycle_id"].astype(str)
        crd_sim, crd_cyc, y = crd_scores(ds, facet)
        assert np.array_equal(cyc, crd_cyc.astype(str)), (ds, facet)
        assert np.array_equal(z["y"], y), (ds, facet)
        y = y.astype(int)
        cyc_ids = pd.unique(cyc)
        idx_by_cyc = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
            lambda s: s.to_numpy()).to_dict()
        for key, label in PROBES:
            probe = z[key]
            a_crd = roc_auc_score(y, -crd_sim)
            a_prb = roc_auc_score(y, probe)          # probe is an anomaly score
            deltas = []
            for _ in range(N_BOOT):
                pick = rng.choice(cyc_ids, len(cyc_ids), replace=True)
                idx = np.concatenate([idx_by_cyc[c] for c in pick])
                yy = y[idx]
                if yy.min() == yy.max():
                    continue
                deltas.append(roc_auc_score(yy, -crd_sim[idx]) -
                              roc_auc_score(yy, probe[idx]))
            deltas = np.asarray(deltas)
            p_left = float((deltas <= 0).mean())      # evidence that probe >= CRD
            p_right = float((deltas >= 0).mean())     # evidence that CRD >= probe
            recs.append({
                "facet": facet, "probe": label,
                "auroc_crd": round(float(a_crd), 4),
                "auroc_probe": round(float(a_prb), 4),
                "delta_pp": round(float(a_crd - a_prb) * 100, 2),
                "ci_lo": round(float(np.percentile(deltas, 2.5)) * 100, 2),
                "ci_hi": round(float(np.percentile(deltas, 97.5)) * 100, 2),
                "p_two_sided": round(min(1.0, 2 * min(p_left, p_right)), 4),
                "n_boot": int(len(deltas)),
            })
            print(recs[-1], flush=True)

    df = pd.DataFrame(recs)
    df["p_holm"] = holm(df.p_two_sided.to_numpy())
    df["significant"] = df.p_holm < 0.05
    df["winner"] = np.where(~df.significant, "tie",
                            np.where(df.delta_pp > 0, "CRD", "probe"))
    df.to_csv(OUT / "probe_bootstrap_vs_crd.csv", index=False, encoding="utf-8-sig")
    print()
    print(df.to_string(index=False))
    print("\nfamily size:", len(df),
          "| probe better:", int((df.winner == "probe").sum()),
          "| CRD better:", int((df.winner == "CRD").sum()),
          "| tie:", int((df.winner == "tie").sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
