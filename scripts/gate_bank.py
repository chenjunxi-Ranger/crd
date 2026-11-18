"""Gate: ceiling once reference-bank statistics are allowed.

This answers a different question from ``gate_combined.py``: if we allow the
official train split as an *unlabelled* reference bank, how much AUROC is
available at all?  It does not decide novelty — CSLS / CMC / LEMoN are known
operators and remain excluded from the final formula by C7.

Analysis only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                # noqa: E402
from src.protocol import (CLIP_DIR, RESULTS_DIR, THRESHOLDS, load_split,  # noqa: E402
                          write_json)


def _bank(bench: str, limit: int, seed: int, enc: JointEncoder):
    df = load_split(bench, "train")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(limit, len(df)), replace=False)
    sub = df.iloc[np.sort(idx)]
    img = enc.encode_images(sub["path"].tolist(), batch_size=64, num_workers=4)
    txt = enc.encode_texts(sub["caption"].tolist())
    return img, txt


def _bank_features(val_img, val_txt, bank_img, bank_txt, k: int = 10) -> dict:
    s_it = val_img @ bank_txt.T          # (N, B)
    s_ti = val_txt @ bank_img.T
    cos = (val_img * val_txt).sum(1)

    top_t = np.sort(s_it, axis=1)[:, -k:]
    top_i = np.sort(s_ti, axis=1)[:, -k:]
    rx, rc = top_t.mean(1), top_i.mean(1)
    mx_t, mx_i = s_it.max(1), s_ti.max(1)
    cmc = -0.5 * ((cos - mx_t) + (cos - mx_i)) / 2.0
    return {
        "cosine": cos,
        "rx": rx,
        "rc": rc,
        "csls": 2 * cos - rx - rc,
        "m_img": cos - mx_t,
        "m_txt": cos - mx_i,
        "cmc": cmc,
        "knn_density": 0.5 * (rx + rc),
        "peak_ratio": np.maximum(mx_t, mx_i) / np.maximum(cos, 1e-6),
    }


def _cv(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    lin, gbm = [], []
    for tr, te in skf.split(x, y):
        if len(np.unique(y[te])) < 2:
            continue
        p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        p.fit(x[tr], y[tr])
        lin.append(roc_auc_score(y[te], p.predict_proba(x[te])[:, 1]))
        g = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        g.fit(x[tr], y[tr])
        gbm.append(roc_auc_score(y[te], g.predict_proba(x[te])[:, 1]))
    return {"linear": float(np.mean(lin)), "gbm": float(np.mean(gbm))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco", choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--bank", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    val = load_split(args.bench, "val", args.limit)
    vi = enc.encode_images(val["path"].tolist(), batch_size=64, num_workers=4)
    vt = enc.encode_texts(val["caption"].tolist())
    bi, bt = _bank(args.bench, args.bank, args.seed, enc)
    feats = _bank_features(vi, vt, bi, bt)
    y = val["label"].to_numpy()

    out = {"bench": args.bench, "n_val": int(len(val)), "n_bank": int(len(bi)),
           "threshold": THRESHOLDS[args.bench], "single": {}, "cv": {}}
    for k, v in feats.items():
        a, b = roc_auc_score(y, v), roc_auc_score(y, -v)
        out["single"][k] = float(max(a, b))               # best orientation (analysis)
    names = list(feats)
    out["cv"]["cosine"] = _cv(feats["cosine"][:, None], y)
    out["cv"]["cosine+cmc"] = _cv(np.stack([feats["cosine"], feats["cmc"]], 1), y)
    out["cv"]["all_bank"] = _cv(np.stack([feats[n] for n in names], 1), y)
    out["best_cv"] = max(v["gbm"] for v in out["cv"].values())
    out["gap_best_cv"] = float(THRESHOLDS[args.bench] - out["best_cv"])
    write_json(RESULTS_DIR / "gate" / f"{args.bench}_bank_ceiling.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
