"""Combined feasibility ceiling: cosine + token-patch + leave-one-word-out.

This is the decisive Gate measurement.  If a supervised fusion of *all* allowed
features cannot reach a threshold, no training-free single formula built from
them can, and the family must be replaced (see session.json D001).

Analysis only; labels never enter ``src/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.counterfactual import LOO_KEYS, content_words, drop_word, loo_features  # noqa: E402
from src.encoders import JointEncoder                                              # noqa: E402
from src.features import FEATURE_KEYS, compute                                     # noqa: E402
from src.protocol import (CLIP_DIR, RESULTS_DIR, THRESHOLDS, load_foil_val,        # noqa: E402
                          load_split, write_json)


def _loo(captions: list[str], imgs: np.ndarray, enc: JointEncoder) -> dict[str, np.ndarray]:
    texts: list[str] = []
    spans: list[list] = []
    for c in captions:
        sp = content_words(c)
        spans.append(sp)
        texts.append(c)
        texts.extend(drop_word(c, s) for s in sp)
    emb = enc.encode_texts(texts)
    rows, cursor = [], 0
    cos = np.zeros(len(captions))
    for i, sp in enumerate(spans):
        full = emb[cursor]
        cursor += 1
        drops = emb[cursor:cursor + len(sp)]
        cursor += len(sp)
        cos[i] = float(imgs[i] @ full)
        rows.append(loo_features(cos[i], drops @ imgs[i]))
    return {k: np.asarray([r[k] for r in rows]) for k in LOO_KEYS}, cos


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


def _row(args, enc) -> dict:
    df = load_split(args.bench, "val", args.limit)
    st = enc.encode(df["path"].tolist(), df["caption"].tolist(),
                    batch_size=args.batch_size, num_workers=args.workers)
    patch = compute(st, enc.tokenizer, enc.grid)
    loo, cos = _loo(df["caption"].tolist(), st.pooled_img, enc)
    y = df["label"].to_numpy()

    blocks = {
        "cosine": [cos],
        "cosine+loo": [cos] + [loo[k] for k in LOO_KEYS],
        "cosine+patch": [cos] + [patch[k] for k in FEATURE_KEYS],
        "all": [cos] + [loo[k] for k in LOO_KEYS] + [patch[k] for k in FEATURE_KEYS],
    }
    out = {"bench": args.bench, "n": int(len(df)), "threshold": THRESHOLDS[args.bench],
           "prevalence": float(y.mean()), "cv": {}, "single": {}}
    for name, cols in blocks.items():
        out["cv"][name] = _cv(np.stack(cols, axis=1), y)
    for name, v in (("cosine", -cos), ("loo_mean", -loo["loo_mean"]),
                    ("loo_min", -loo["loo_min"]), ("support_mean", -patch["support_mean"]),
                    ("coverage_soft", -patch["coverage_soft"])):
        out["single"][name] = float(roc_auc_score(y, v))
    out["best_cv"] = max(v["gbm"] for v in out["cv"].values())
    out["gap_best_cv"] = float(THRESHOLDS[args.bench] - out["best_cv"])
    return out


def _foil(args, enc) -> dict:
    df = load_foil_val(args.limit)
    n = len(df)
    st = enc.encode(df["path"].tolist() * 2,
                    df["positive"].tolist() + df["negative"].tolist(),
                    batch_size=args.batch_size, num_workers=args.workers)
    patch = compute(st, enc.tokenizer, enc.grid)
    loo, cos = _loo(df["positive"].tolist() + df["negative"].tolist(), st.pooled_img, enc)

    def acc(diff: np.ndarray) -> float:
        return float(np.mean(np.where(diff < 0, 1.0, np.where(diff > 0, 0.0, 0.5))))

    out = {"bench": "foil", "n_pairs": int(n), "threshold": THRESHOLDS["foil"], "single": {}}
    out["single"]["cosine"] = acc(cos[n:] - cos[:n])
    for k in LOO_KEYS:
        out["single"][f"loo_{k}"] = acc(loo[k][n:] - loo[k][:n])
    for k in FEATURE_KEYS:
        out["single"][f"patch_{k}"] = acc(patch[k][n:] - patch[k][:n])
    out["best_single"] = max(out["single"].values())
    out["best_single_name"] = max(out["single"], key=out["single"].get)
    out["gap_best_single"] = float(THRESHOLDS["foil"] - out["best_single"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    enc = JointEncoder(CLIP_DIR)
    payload = _foil(args, enc) if args.bench == "foil" else _row(args, enc)
    write_json(RESULTS_DIR / "gate" / f"{args.bench}_ceiling.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
