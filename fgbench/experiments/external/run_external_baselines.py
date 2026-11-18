"""CSLS and LEMoN-style on an external corpus (Flickr30k), plus the reference rows.

Ports the two bank-based detectors from the FGBench evaluation to the external
protocol, with everything refitted on the external train split:

    CSLS         cos - lam * (topk_mean(u, text_bank) + topk_mean(v, image_bank))
                 bank = train features, lam selected on val AUROC
    LEMoN-style  8 neighbour-agreement features over K=32 library entries,
                 logistic regression fitted on train labels
    GMM-based    not run: its posterior is a monotone function of the cosine, so
                 its AUROC is identical to the cosine by construction (see the
                 docstring of fgbench/models/gmm_filter.py) -- reported as such
    TRACED-style not run here: it needs a deletion trajectory and re-encoding of
                 every caption at each step, which is hours on this corpus

Run:  python fgbench/experiments/external/run_external_baselines.py --noise noun
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DATA = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\derived\flickr30k")
OUT = ROOT / "results" / "fgbench" / "external"
K_NN = 10          # CSLS k
K_LEMON = 32       # LEMoN library neighbours


def auc(y, x):
    """x is in 'higher = cleaner' space."""
    return float(roc_auc_score(y, -x))


def ap(y, x):
    return float(average_precision_score(y, -x))


def topk_mean(a: np.ndarray, b: np.ndarray, k: int = K_NN,
              chunk: int = 512) -> np.ndarray:
    """Mean of the top-k similarities between each row of `a` and all of `b`."""
    return np.asarray([
        b.T @ a[s:s + chunk].T for s in range(0, len(a), chunk)
    ][0]) if False else np.concatenate([
        np.sort(a[s:s + chunk] @ b.T, axis=1)[:, -k:].mean(1)
        for s in range(0, len(a), chunk)
    ])


def lemon_features(U, V, LI, LT, k=K_LEMON, chunk=512) -> np.ndarray:
    """Same 8 neighbour-agreement features as fgbench/evaluate/lemon_fgbench.py."""
    n = len(U)
    out = np.zeros((n, 8), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Si = U[s:e] @ LI.T
        St = V[s:e] @ LT.T
        ii = np.argpartition(-Si, k, axis=1)[:, :k]
        ti = np.argpartition(-St, k, axis=1)[:, :k]
        nbr_txt = LT[ii]
        nbr_img = LI[ti]
        sc_i = np.einsum("cd,ckd->ck", V[s:e], nbr_txt)
        sc_t = np.einsum("cd,ckd->ck", U[s:e], nbr_img)
        s_self = (U[s:e] * V[s:e]).sum(1)
        out[s:e, 0] = s_self
        out[s:e, 1] = Si[np.arange(e - s), ii[:, 0]]
        out[s:e, 2] = St[np.arange(e - s), ti[:, 0]]
        out[s:e, 3] = sc_i.mean(1)
        out[s:e, 4] = sc_t.mean(1)
        out[s:e, 5] = sc_i.max(1) - sc_i.min(1)
        out[s:e, 6] = sc_t.max(1) - sc_t.min(1)
        out[s:e, 7] = sc_i.mean(1) - sc_t.mean(1)
    return out


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--noise", default="noun", choices=["random", "noun"])
    ap_.add_argument("--level", default="0.40")
    a = ap_.parse_args()

    df = pd.read_csv(DATA / f"flickr30k__{a.noise}__lvl{a.level}__seed0.csv",
                     encoding="utf-8-sig")
    with open(OUT / "feat_flickr_images.pkl", "rb") as f:
        U = pickle.load(f)
    for cand in (OUT / f"feat_flickr_txt_{a.noise}{a.level}.pkl",
                 OUT / f"feat_flickr_{a.noise}{a.level}.pkl"):
        if cand.is_file():
            with open(cand, "rb") as f:
                obj = pickle.load(f)
            # the first run cached (U, V) under one tag; later runs split them
            U, V = obj if isinstance(obj, tuple) else (U, obj)
            print(f"  cache: {cand.name}", flush=True)
            break
    else:
        raise FileNotFoundError(
            f"no caption cache for {a.noise}{a.level}; run run_flickr30k.py first")
    S = np.einsum("ij,ij->i", U, V)
    y = df.is_mislabel.to_numpy()
    m_tr = (df.split == "train").to_numpy()
    m_va = (df.split == "val").to_numpy()
    m_te = (df.split == "test").to_numpy()
    y_va, y_te = y[m_va], y[m_te]

    mu, sd = float(S[m_tr].mean()), float(S[m_tr].std() + 1e-8)
    Z = (S - mu) / sd

    LI, LT = U[m_tr], V[m_tr]                      # library = train split
    rows = [("CLIP cosine", auc(y_te, Z[m_te]), ap(y_te, Z[m_te]), "")]
    print("cosine", rows[-1][1], flush=True)

    # ---------------- CSLS
    ri_va = topk_mean(U[m_va], LT)
    rc_va = topk_mean(V[m_va], LI)
    ri_te = topk_mean(U[m_te], LT)
    rc_te = topk_mean(V[m_te], LI)
    lam = max(np.round(np.arange(0, 1.51, 0.05), 2),
              key=lambda L: auc(y_va, Z[m_va] - L * (ri_va + rc_va)))
    s_csls = Z[m_te] - lam * (ri_te + rc_te)
    rows.append(("CLIP + CSLS", auc(y_te, s_csls), ap(y_te, s_csls),
                 f"lam={lam}"))
    print("csls", rows[-1][1], flush=True)

    # ---------------- LEMoN-style
    F = lemon_features(U, V, LI, LT)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(F[m_tr], y[m_tr])
    p_te = clf.predict_proba(F[m_te])[:, 1]        # P(mislabel)
    rows.append(("LEMoN-style", auc(y_te, -p_te), ap(y_te, -p_te), "LR/K=32"))
    print("lemon", rows[-1][1], flush=True)

    # ---------------- reference rows already measured
    rows.append(("GMM-based", rows[0][1], rows[0][2], "== cosine (identity)"))
    rows.append(("TRACED-style", float("nan"), float("nan"), "not run"))

    d = pd.DataFrame(rows, columns=["method", "auroc", "auprc", "note"])
    d.to_csv(OUT / f"external_methods_{a.noise}{a.level}.csv", index=False,
             encoding="utf-8-sig")
    print()
    print(d.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
