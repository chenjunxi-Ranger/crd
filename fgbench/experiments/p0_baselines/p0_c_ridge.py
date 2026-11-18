"""P0-C: ridge linear heads + R^2 / shortcut attribution for g.

Questions:
  1. If a(u) and b(v) are linear (closed-form ridge) instead of MLPs, does the
     calibration still work?  If not, the gain is nonlinear, not "additive".
  2. How much variance of the trained g is explained by z (R^2), and how much by
     caption length / numeral count (shortcut check)?

Output: results/fgbench/p0_baselines/p0_c_ridge.csv
Run:    python .../p0_c_ridge.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "experiments" / "p0_baselines"))
from run_baselines import load_cache  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "p0_baselines"
FACETS = [("COCO", "Entity", 8000), ("COCO", "Quantity", 2500),
          ("VG", "Attribute", 8000), ("VG", "Relation", 2000)]
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
LAMS = np.round(np.arange(0.0, 2.01, 0.05), 2)


def ridge_fit(X, y, alpha):
    X1 = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    A = X1.T @ X1 + alpha * np.eye(X1.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, X1.T @ y)
    return w


def ridge_pred(X, w):
    return np.concatenate([X, np.ones((len(X), 1), np.float32)], 1) @ w


def r2(pred, target):
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum()) + 1e-12
    return 1 - ss_res / ss_tot


def main() -> int:
    recs = []
    for ds, facet, cycles in FACETS:
        dtr, Utr, Vtr = load_cache(ds, facet, "train", cycles)
        dva, Uva, Vva = load_cache(ds, facet, "val")
        dte, Ute, Vte = load_cache(ds, facet, "test")
        s_tr, s_va, s_te = (Utr * Vtr).sum(1), (Uva * Vva).sum(1), (Ute * Vte).sum(1)
        mu, sd = s_tr.mean(), s_tr.std() + 1e-8
        z_tr, z_va, z_te = (s_tr - mu) / sd, (s_va - mu) / sd, (s_te - mu) / sd
        yv, yt = dva.pair_label.to_numpy(), dte.pair_label.to_numpy()

        Xtr = np.concatenate([Utr, Vtr], 1)
        Xva = np.concatenate([Uva, Vva], 1)
        Xte = np.concatenate([Ute, Vte], 1)
        best = None
        for a in ALPHAS:
            w = ridge_fit(Xtr, z_tr, a)
            g_tr, g_va, g_te = (ridge_pred(Xtr, w), ridge_pred(Xva, w),
                                ridge_pred(Xte, w))
            for L in LAMS:
                auc = roc_auc_score(yv, -(z_va - L * g_va))
                if best is None or auc > best[0]:
                    best = (auc, a, L, g_tr, g_te)
        val_auc, alpha, lam, g_tr, g_te = best
        a_ridge = roc_auc_score(yt, -(z_te - lam * g_te))
        z0 = np.load(CACHE / f"crdbb_scores_clip_{ds}_{facet}_both_row_s0.npz",
                     allow_pickle=True)

        # shortcut features from captions
        cap = dte.caption.astype(str).to_numpy()
        length = np.array([len(c.split()) for c in cap], float)
        nums = np.array([len(re.findall(r"\d+", c)) for c in cap], float)
        Xs = np.stack([length, nums, np.ones_like(length)], 1)
        ws = np.linalg.lstsq(Xs, g_te, rcond=None)[0]
        r2_short = r2(Xs @ ws, g_te)
        rho_len = float(pd.Series(length).corr(pd.Series(g_te), method="spearman"))
        rho_num = float(pd.Series(nums).corr(pd.Series(g_te), method="spearman"))

        recs.append({
            "facet": facet, "alpha": alpha, "lam_ridge": lam,
            "val_auc_ridge": round(float(val_auc), 4),
            "auroc_ridge": round(float(a_ridge), 4),
            "auroc_crd": round(float(roc_auc_score(yt, -z0["crd"])), 4),
            "r2_train": round(float(r2(g_tr, z_tr)), 4),
            "r2_test": round(float(r2(g_te, z_te)), 4),
            "r2_shortcut_g": round(float(r2_short), 4),
            "spearman_len": round(rho_len, 4), "spearman_num": round(rho_num, 4),
        })
        print(recs[-1], flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "p0_c_ridge.csv", index=False, encoding="utf-8-sig")
    print("\n", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
