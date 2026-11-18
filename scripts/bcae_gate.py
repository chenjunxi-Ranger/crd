"""BCAE admission: batch-conditional assignment evidence.

For a cohort of B pairs, build S_ij = s(I_i, c_j) and put a Gibbs distribution on
perfect matchings,  P(M) ∝ exp(tau * E(M)).  For B=8 the marginal inclusion
probability of the diagonal can be enumerated exactly (8! = 40,320).

Anomaly score: A = -log(p_i + eps)  (low retention -> suspicious)

Controls
    M0 cosine | M1 +CSLS | M2 +Hungarian margin | M3 +BCAE p_i
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, load_split, write_json   # noqa: E402
from src.scorer import build_scorer                            # noqa: E402

B = 8
TAU = 10.0


def perms(Bn: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(Bn))), dtype=np.int16)


def marginals(S: np.ndarray, tau: float, P: np.ndarray):
    idx = np.arange(S.shape[0])[None, :]
    E = S[idx, P].sum(1)                   # (n_perm,)
    w = np.exp(tau * (E - E.max()))
    w /= w.sum()
    p = np.array([w[P[:, i] == i].sum() for i in range(S.shape[0])])
    ent = float(-(w * np.log(w + 1e-12)).sum())
    return p, ent


def hungarian_margin(S: np.ndarray, P: np.ndarray):
    idx = np.arange(S.shape[0])[None, :]
    E = S[idx, P].sum(1)
    out = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        has = P[:, i] == i
        best_has = E[has].max()
        best_no = E[~has].max() if (~has).any() else best_has
        out[i] = best_has - best_no
    return out


def cv_auc(X, y, seed=0):
    w = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    p = cross_val_predict(w, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--backbone", default="clip", choices=["clip", "fg"])
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()

    df = load_split(args.bench, "val", args.limit).drop_duplicates("image_id")
    df = df.reset_index(drop=True)
    sc = build_scorer(args.backbone)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(df))
    P = perms(B)

    # real reference-bank CSLS for the M1/M2 controls (BCAE stays the only variable)
    if args.backbone == "clip":
        from rat_gate import topk_mean
        from src.encoders import JointEncoder
        from src.protocol import CLIP_DIR
        enc = JointEncoder(CLIP_DIR)
        vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
        vt = enc.encode_texts(df["caption"].astype(str).tolist())
        zb = np.load(RESULTS_DIR / "gate" / f"bank_{args.bench}_all.npz")
        bank_ri = topk_mean(vi, zb["txt"])
        bank_rc = topk_mean(vt, zb["img"])
    else:
        bank_ri = np.zeros(len(df)); bank_rc = np.zeros(len(df))

    # CSLS-style density from the same cohort (reference = cohort captions/images)
    rows = []
    for s0 in range(0, len(df) - B + 1, B):
        idx = order[s0:s0 + B]
        sub = df.iloc[idx]
        imgs = sub["path"].tolist(); caps = sub["caption"].astype(str).tolist()
        S = np.zeros((B, B))
        for i in range(B):
            S[i] = sc.score_many([imgs[i]] * B, caps)
        p, ent = marginals(S, args.tau, P)
        marg = hungarian_margin(S, P)
        diag = np.diag(S)
        # cohort-internal density (CSLS proxy): mean similarity to the other side
        ri = bank_ri[idx]
        rc = bank_rc[idx]
        for k in range(B):
            rows.append({"sample_id": int(idx[k]), "image_id": sub["image_id"].iloc[k],
                         "label": int(sub["label"].iloc[k]), "batch_id": s0 // B,
                         "diag": float(diag[k]), "p": float(p[k]), "entropy": ent,
                         "margin": float(marg[k]), "ri": float(ri[k]), "rc": float(rc[k])})
    R = pd.DataFrame(rows)
    y = R["label"].to_numpy()
    s = -R["diag"].to_numpy()
    h = (R["ri"] + R["rc"]).to_numpy()
    out = {"bench": args.bench, "backbone": args.backbone, "n": int(len(R)),
           "tau": args.tau, "batch": B,
           "auc_cos": float(roc_auc_score(y, s)),
           "auc_p_alone": float(roc_auc_score(y, -np.log(R["p"] + 1e-9))),
           "auc_margin_alone": float(roc_auc_score(y, -R["margin"]))}
    out["M1_csls"] = cv_auc(np.column_stack([s, h]), y)
    out["M2_hungarian"] = cv_auc(np.column_stack([s, h, R["margin"]]), y)
    out["M3_bcae"] = cv_auc(np.column_stack([s, h, R["margin"], -np.log(R["p"] + 1e-9)]), y)
    out["delta_bcae_over_M2"] = out["M3_bcae"] - out["M2_hungarian"]
    # stability of p across cohorts: re-draw batches and correlate
    order2 = np.random.default_rng(7).permutation(len(df))
    p2 = {}
    for s0 in range(0, len(df) - B + 1, B):
        idx = order2[s0:s0 + B]
        sub = df.iloc[idx]
        imgs = sub["path"].tolist(); caps = sub["caption"].astype(str).tolist()
        S2 = np.zeros((B, B))
        for i in range(B):
            S2[i] = sc.score_many([imgs[i]] * B, caps)
        pp, _ = marginals(S2, args.tau, P)
        for k in range(B):
            p2[int(idx[k])] = float(pp[k])
    merged = R[R["sample_id"].isin(p2)].copy()
    merged["p2"] = merged["sample_id"].map(p2)
    out["cohort_stability_spearman"] = float(merged["p"].corr(merged["p2"], method="spearman"))
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"bcae_{args.backbone}_{args.bench}.json", out)
    R.to_csv(RESULTS_DIR / "gate" / f"bcae_{args.backbone}_{args.bench}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
