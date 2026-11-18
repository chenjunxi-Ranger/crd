"""BCSC zero-training diagnostic.

R_{T->I} = min_{a in simplex} || t - sum_j a_j v_j ||^2
R_{I->T} = min_{b in simplex} || v - sum_k b_k t_k ||^2
R_BCSC   = (R_TI + R_IT) / 2

Controls
--------
* R_nearest : min_j ||t - v_j||^2   (single patch, no combination)
* R_rot     : same as R_TI after applying one fixed random orthogonal map to
              all patch vectors (preserves patch geometry, breaks the shared
              coordinate frame with the text side)
* direction ablation: R_TI only, R_IT only, both
* conditional MCA at small CLIP margins
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.features import content_mask                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402


def simplex_qp(V: np.ndarray, target: np.ndarray) -> float:
    """min_{a>=0, sum a=1} ||target - V^T a||^2 ."""
    G = V @ V.T
    b = V @ target
    n = len(V)

    def f(a):
        return float(a @ G @ a - 2 * a @ b + target @ target)

    def g(a):
        return 2 * (G @ a - b)

    a0 = np.full(n, 1.0 / n)
    cons = ({"type": "eq", "fun": lambda a: a.sum() - 1.0,
             "jac": lambda a: np.ones(n)},)
    res = minimize(f, a0, jac=g, bounds=[(0.0, 1.0)] * n, constraints=cons,
                   method="SLSQP", options={"maxiter": 60, "ftol": 1e-8})
    a = np.clip(res.x, 0, None)
    s = a.sum()
    a = a / s if s > 0 else a0
    return float(max(0.0, f(a)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco", choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    rng = np.random.default_rng(0)
    A0 = rng.normal(size=(512, 512))
    Q, _ = np.linalg.qr(A0)                    # random orthogonal map

    def feats(paths, caps):
        out = {k: [] for k in ("s", "r_ti", "r_it", "r_nearest", "r_rot")}
        for i in range(0, len(paths), args.chunk):
            st = enc.encode(paths[i:i + args.chunk], caps[i:i + args.chunk],
                            batch_size=64, num_workers=4)
            mask = content_mask(st, enc.tokenizer)
            for r in range(len(st.pooled_img)):
                V = st.patch[r].astype(np.float64)
                Tk = st.token[r][mask[r]].astype(np.float64)
                t = st.pooled_txt[r].astype(np.float64)
                v = st.pooled_img[r].astype(np.float64)
                out["s"].append(float(v @ t))
                out["r_ti"].append(simplex_qp(V, t))
                out["r_it"].append(simplex_qp(Tk, v) if len(Tk) else 0.0)
                out["r_nearest"].append(float(2 - 2 * (V @ t).max()))
                out["r_rot"].append(simplex_qp(V @ Q, t))
        return {k: np.asarray(v) for k, v in out.items()}

    if args.bench == "foil":
        from src.protocol import load_foil_val
        df = load_foil_val(args.limit)
        pos = feats(df["path"].tolist(), df["positive"].tolist())
        neg = feats(df["path"].tolist(), df["negative"].tolist())
        res = {"n_pairs": int(len(df))}
        for key in ("r_ti", "r_it", "r_nearest", "r_rot"):
            d = pos[key] - neg[key]
            res[key] = float(np.mean(np.where(d < 0, 1.0, np.where(d > 0, 0.0, 0.5))))
        res["bcsc"] = float(np.mean(np.where(
            0.5 * (pos["r_ti"] + pos["r_it"]) < 0.5 * (neg["r_ti"] + neg["r_it"]),
            1.0, 0.0)))
        res["clip_acc"] = float(np.mean(np.where(neg["s"] < pos["s"], 1.0,
                                                 np.where(neg["s"] > pos["s"], 0.0, 0.5))))
        payload = {"bench": "foil", "foil": res}
        print(json.dumps(res, ensure_ascii=False, indent=2))
        np.savez_compressed(RESULTS_DIR / "gate" / "bcsc_foil.npz",
                            s_pos=pos["s"], s_neg=neg["s"],
                            rti_pos=pos["r_ti"], rti_neg=neg["r_ti"],
                            rit_pos=pos["r_it"], rit_neg=neg["r_it"])
    else:
        df = load_split(args.bench, "val", args.limit)
        f = feats(df["path"].tolist(), df["caption"].tolist())
        y = df["label"].to_numpy()
        bcsc = 0.5 * (f["r_ti"] + f["r_it"])
        res = {"n": int(len(df)), "auc_cos": float(roc_auc_score(y, -f["s"])),
               "auc": {"bcsc": float(roc_auc_score(y, bcsc)),
                       "r_ti": float(roc_auc_score(y, f["r_ti"])),
                       "r_it": float(roc_auc_score(y, f["r_it"])),
                       "r_nearest": float(roc_auc_score(y, f["r_nearest"])),
                       "r_rot": float(roc_auc_score(y, f["r_rot"]))},
               "mean": {k: [float(f[k][y == 0].mean()), float(f[k][y == 1].mean())]
                        for k in ("r_ti", "r_it", "r_nearest", "r_rot")}}
        # conditional MCA on CLIP-near pairs
        clean, noisy = f["s"][y == 0], f["s"][y == 1]
        order = np.argsort(clean)
        cs = clean[order]
        p = np.searchsorted(cs, noisy)
        left = np.clip(p - 1, 0, len(cs) - 1)
        right = np.clip(p, 0, len(cs) - 1)
        choose = np.where(np.abs(cs[left] - noisy) <= np.abs(cs[right] - noisy), left, right)
        partner = order[choose]
        delta = np.abs(clean[partner] - noisy)
        cond = {}
        for eps in (0.005, 0.01):
            keep = delta < eps
            cond[str(eps)] = {"n": int(keep.sum())}
            for k in ("r_ti", "r_it", "r_nearest", "r_rot"):
                cond[str(eps)][k] = float(np.mean(f[k][y == 0][partner][keep]
                                                  < f[k][y == 1][keep]))
        res["conditional"] = cond
        payload = {"bench": args.bench, "row": res}
        print(json.dumps(res, ensure_ascii=False, indent=2))
        np.savez_compressed(RESULTS_DIR / "gate" / f"bcsc_{args.bench}.npz",
                            s=f["s"], rti=f["r_ti"], rit=f["r_it"],
                            rnear=f["r_nearest"], rrot=f["r_rot"], label=y)

    write_json(RESULTS_DIR / "gate" / f"bcsc_{args.bench}.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
