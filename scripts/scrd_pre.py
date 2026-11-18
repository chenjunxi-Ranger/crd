"""SCRD gate: does the residual DIRECTION q carry information beyond (s, h)?

q = normalize(v - t), orthogonal to m = normalize(v + t); ||v - t||^2 = 2(1 - s).
Therefore ||q|| carries no information at all -- only the direction can.

Test (per benchmark, 5-fold CV, PCA fitted inside each fold):
    [s, h]            vs   [s, h, q_pca]
    q-only
If adding q does not improve held-out AUROC, the density head has nothing to
model and SCRD must not be trained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402


def cv_auc(feat_fn, q, base, y, n_pca=16):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    vals = []
    for tr, te in skf.split(base, y):
        if q is None:
            xtr, xte = base[tr], base[te]
        else:
            pca = PCA(n_components=min(n_pca, len(tr) - 1, q.shape[1]), random_state=0)
            qtr = pca.fit_transform(q[tr])
            qte = pca.transform(q[te])
            if base.shape[1] == 0:
                xtr, xte = qtr, qte
            else:
                xtr = np.column_stack([base[tr], qtr])
                xte = np.column_stack([base[te], qte])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        m.fit(xtr, y[tr])
        vals.append(roc_auc_score(y[te], m.predict_proba(xte)[:, 1]))
    return float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--limit", type=int, default=1200)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    df = load_split(args.bench, "val", args.limit)
    n = len(df)
    v = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
    t = enc.encode_texts(df["caption"].tolist())
    y = df["label"].to_numpy()

    r = np.load(RESULTS_DIR / "gate" / f"rsg_{args.bench}.npz")
    n = min(n, len(r["cos"]))
    v, t, y = v[:n], t[:n], y[:n]
    s = (v * t).sum(1)
    h = r["h"][:n]
    diff = v - t
    q = diff / np.clip(np.linalg.norm(diff, axis=1, keepdims=True), 1e-8, None)

    base = np.column_stack([s, h])
    out = {"bench": args.bench, "n": int(n),
           "auc_s": float(roc_auc_score(y, -s)),
           "auc_h_s": float(roc_auc_score(y, h - 2 * s)),
           "cv_base(s,h)": cv_auc(None, None, base, y),
           "cv_plus_q": cv_auc(None, q, base, y),
           "cv_q_only": cv_auc(None, q, np.zeros((n, 0)), y)}
    out["delta_q"] = out["cv_plus_q"] - out["cv_base(s,h)"]

    # direction-only check: is q itself label-separable after removing s?
    # (rank-correlation of each sample with the mean clean-q, a cheap sanity probe)
    qc = q[y == 0].mean(0)
    qn = q[y == 1].mean(0)
    out["cos(qc,qn)"] = float(qc @ qn / (np.linalg.norm(qc) * np.linalg.norm(qn) + 1e-8))

    write_json(RESULTS_DIR / "gate" / f"scrd_{args.bench}.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
