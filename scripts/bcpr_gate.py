"""BCPR diagnostic: bidirectional conditional pair residual (M1' embedding interface).

Learn two ridge maps on the official train split (cross-fitted):
    W_IT : u -> v    W_TI : v -> u
Residuals -> shrunk covariance -> Mahalanobis energy
    q = 0.5 [ e_IT' S_IT^-1 e_IT + e_TI' S_TI^-1 e_TI ]

Controls
    M0 cosine | M1 +CSLS | M2 q alone | M3 +q | M4 +||e|| (plain residual norm)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, load_split, write_json   # noqa: E402


def load_emb(bench: str, backbone: str, split: str, limit: int | None):
    p = RESULTS_DIR / "gate" / f"emb_{backbone}_{bench}_{split}.npz"
    if p.is_file():
        z = np.load(p, allow_pickle=True)
        return z["img"], z["txt"], z["label"], z["image_id"].astype(str)
    raise SystemExit(f"missing embeddings: {p} (run scripts/bcpr_encode.py first)")


def shrink_cov(E: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    S = np.cov(E, rowvar=False)
    d = S.shape[0]
    return (1 - alpha) * S + alpha * np.trace(S) / d * np.eye(d)


def maha(E: np.ndarray, S: np.ndarray) -> np.ndarray:
    Si = np.linalg.pinv(S)
    return np.einsum("ij,jk,ik->i", E, Si, E)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--backbone", default="clip", choices=["clip", "fg"])
    ap.add_argument("--pca", type=int, default=64)
    args = ap.parse_args()

    utr, vtr, _, _ = load_emb(args.bench, args.backbone, "train", None)
    uva, vva, y, gid = load_emb(args.bench, args.backbone, "val", None)

    pca = PCA(n_components=args.pca, random_state=0).fit(np.vstack([utr, vtr]))
    Utr, Vtr = pca.transform(utr), pca.transform(vtr)
    Uva, Vva = pca.transform(uva), pca.transform(vva)
    sc_u = StandardScaler().fit(Utr); sc_v = StandardScaler().fit(Vtr)
    Utr, Vtr = sc_u.transform(Utr), sc_v.transform(Vtr)
    Uva, Vva = sc_u.transform(Uva), sc_v.transform(Vva)

    # cross-fitted residuals on train (no self-fit leakage)
    E1 = np.zeros_like(Vtr); E2 = np.zeros_like(Utr)
    kf = KFold(5, shuffle=True, random_state=0)
    for tr, te in kf.split(Utr):
        w1 = Ridge(alpha=1.0).fit(Utr[tr], Vtr[tr])
        w2 = Ridge(alpha=1.0).fit(Vtr[tr], Utr[tr])
        E1[te] = Vtr[te] - w1.predict(Utr[te])
        E2[te] = Utr[te] - w2.predict(Vtr[te])
    S1, S2 = shrink_cov(E1), shrink_cov(E2)

    w1 = Ridge(alpha=1.0).fit(Utr, Vtr)
    w2 = Ridge(alpha=1.0).fit(Vtr, Utr)
    e1 = Vva - w1.predict(Uva)
    e2 = Uva - w2.predict(Vva)
    q = 0.5 * (maha(e1, S1) + maha(e2, S2))
    norm = np.sqrt((e1 ** 2).sum(1) + (e2 ** 2).sum(1))

    # CSLS baseline from the frozen bank (same rows)
    zb = np.load(RESULTS_DIR / "gate" / f"bank_{args.bench}_all.npz")
    sim_t = uva @ zb["txt"].T
    sim_i = vva @ zb["img"].T
    k = 10
    h = np.partition(sim_t, -k, axis=1)[:, -k:].mean(1) + \
        np.partition(sim_i, -k, axis=1)[:, -k:].mean(1)
    s = -(uva * vva).sum(1)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    def cv(X, yy):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        p = cross_val_predict(m, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
        return float(roc_auc_score(yy, p))

    out = {"bench": args.bench, "backbone": args.backbone, "n": int(len(y)),
           "M0_cos": float(roc_auc_score(y, s)),
           "M2_bcpr_alone": float(roc_auc_score(y, q)),
           "M4_norm_alone": float(roc_auc_score(y, norm))}
    out["M1_csls"] = cv(np.column_stack([s, h]), y)
    out["M3_csls_bcpr"] = cv(np.column_stack([s, h, q]), y)
    out["M4_csls_norm"] = cv(np.column_stack([s, h, norm]), y)
    out["delta_ind"] = out["M3_csls_bcpr"] - out["M1_csls"]
    out["delta_norm"] = out["M4_csls_norm"] - out["M1_csls"]
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"bcpr_{args.backbone}_{args.bench}.json", out)
    np.savez_compressed(RESULTS_DIR / "gate" / f"bcpr_{args.backbone}_{args.bench}.npz",
                        q=q, norm=norm, label=y, image_id=gid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
