"""P0-1: re-select CSLS's lambda on a wide grid (0-30) for all five facets.

The frozen main table used Lambda = {0, 0.1, ..., 2.0} for CSLS. The external
report showed that this grid was truncated on three external protocols, where
the optimum lies at 10-18 and CSLS overtakes CRD. This script repeats the check
inside FGBench: same features, same reference bank (8,000 training rows), same
validation split, but Lambda = {0, 0.5, ..., 30}.

Output: results/fgbench/p0_baselines/p0_1_csls_grid.csv
Run:    python fgbench/experiments/p0_baselines/p0_1_csls_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "experiments" / "p0_baselines"))

from run_baselines import FACETS, action_splits, load_cache  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "p0_baselines"
BANK = 8000
K = 10
GRID = np.round(np.arange(0.0, 30.01, 0.5), 2)
N_BOOT = 2000


def topk_mean(A, B, k=K, chunk=2048):
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        s = A[i:i + chunk] @ B.T
        kk = min(k, s.shape[1])
        out[i:i + chunk] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def crd_anom(ds, facet):
    if facet == "Action":
        zs = [np.load(CACHE / f"crd_action_scores_s{s}.npz", allow_pickle=True)
              for s in (0, 1, 2)]
    else:
        zs = [np.load(CACHE / f"crdbb_scores_clip_{ds}_{facet}_both_row_s{s}.npz",
                      allow_pickle=True) for s in (0, 1, 2)]
    return np.mean([-z["crd"] for z in zs], axis=0), zs[0]["cycle_id"], zs[0]["y"]


def main() -> int:
    rng = np.random.default_rng(0)
    rows = []
    for ds, facet, cycles in FACETS:
        if facet == "Action":
            (dtr, Utr, Vtr), (dva, Uva, Vva), (dte, Ute, Vte) = action_splits()
        else:
            dtr, Utr, Vtr = load_cache(ds, facet, "train", cycles)
            dva, Uva, Vva = load_cache(ds, facet, "val")
            dte, Ute, Vte = load_cache(ds, facet, "test")

        # use the same reference bank as the frozen main table (crd_bootstrap.py)
        bank = np.load(CACHE / f"bank_b16_{ds}.npz", allow_pickle=True)
        Ub, Vb = bank["img"], bank["txt"]
        cos_va, cos_te = (Uva * Vva).sum(1), (Ute * Vte).sum(1)
        ri_va, rc_va = topk_mean(Uva, Vb), topk_mean(Vva, Ub)
        ri_te, rc_te = topk_mean(Ute, Vb), topk_mean(Vte, Ub)
        yv, yt = dva.pair_label.to_numpy(), dte.pair_label.to_numpy()

        val_auc = [roc_auc_score(yv, -(cos_va - L * (ri_va + rc_va))) for L in GRID]
        lam = float(GRID[int(np.argmax(val_auc))])
        csls_te = cos_te - lam * (ri_te + rc_te)
        a_csls = roc_auc_score(yt, -csls_te)

        crd, cyc, yy = crd_anom(ds, facet)
        assert (cyc == dte.cycle_id.values).all() and (yy == yt).all(), facet
        a_crd = roc_auc_score(yt, crd)

        # paired cycle-clustered bootstrap: CRD - CSLS
        cid = dte.cycle_id.values
        keys = pd.unique(cid)
        idx_by = pd.Series(np.arange(len(cid))).groupby(cid).apply(
            lambda s: s.to_numpy()).to_dict()
        d = []
        for _ in range(N_BOOT):
            pick = rng.choice(keys, len(keys), replace=True)
            ii = np.concatenate([idx_by[c] for c in pick])
            if yt[ii].min() == yt[ii].max():
                continue
            d.append(roc_auc_score(yt[ii], crd[ii]) -
                     roc_auc_score(yt[ii], -csls_te[ii]))
        d = np.asarray(d)
        rec = {
            "facet": facet, "lambda_wide": lam,
            "val_auroc_csls": round(float(max(val_auc)), 4),
            "auroc_csls": round(float(a_csls), 4),
            "auroc_crd": round(float(a_crd), 4),
            "delta_crd_minus_csls_pp": round(float(a_crd - a_csls) * 100, 2),
            "ci_lo": round(float(np.percentile(d, 2.5)) * 100, 2),
            "ci_hi": round(float(np.percentile(d, 97.5)) * 100, 2),
            "p_two_sided": round(min(1.0, 2 * min(float((d <= 0).mean()),
                                                  float((d >= 0).mean()))), 4),
            "lambda_at_grid_edge": bool(lam >= GRID[-1] - 1e-9),
            "n_boot": int(len(d)),
        }
        rows.append(rec)
        print(rec, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p0_1_csls_grid.csv", index=False, encoding="utf-8-sig")
    print("\n", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
