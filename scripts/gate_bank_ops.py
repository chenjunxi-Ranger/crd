"""Gate for NEW bank-side operators (set / graph statistics, not margins).

Candidates measured here (all use only the unlabelled official train bank):
* BNSC  — bidirectional neighbour *set* consistency: overlap between the images
          behind this image's top-k captions and the images this caption ranks
          on top.
* RRP   — reciprocal rank product, log-scaled.
* STAB  — matching stability: does the pair stay top-1 after removing its best
          competitor (second-best margin, no averaging).
* COH   — neighbourhood coherence: mean text-text similarity among the image's
          top-k bank captions (a second-order, set-level statistic).

Analysis only.  Known margin means (CSLS/CMC) are computed as reference rows.
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

from src.encoders import JointEncoder                   # noqa: E402
from src.protocol import (CLIP_DIR, RESULTS_DIR, THRESHOLDS, load_split,  # noqa: E402
                          write_json)


def _load_bank(bench: str, limit: int, seed: int, enc: JointEncoder):
    if limit <= 0:
        full = RESULTS_DIR / "gate" / f"bank_{bench}_all.npz"
        if full.is_file():
            with np.load(full) as z:
                return z["img"], z["txt"]
        raise SystemExit(f"{full} missing; run scripts/prepare_bank.py first")
    path = RESULTS_DIR / "gate" / f"bank_{bench}_{limit}.npz"
    if path.is_file():
        with np.load(path) as z:
            return z["img"], z["txt"]
    df = load_split(bench, "train")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=min(limit, len(df)), replace=False))
    sub = df.iloc[idx]
    img = enc.encode_images(sub["path"].tolist(), batch_size=64, num_workers=4)
    txt = enc.encode_texts(sub["caption"].tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, img=img, txt=txt)
    return img, txt


def _ops(val_img, val_txt, bank_img, bank_txt, k: int = 10,
         chunk: int = 256) -> dict[str, np.ndarray]:
    """Chunked so a 113k bank never materialises a full NxB matrix twice."""
    n = len(val_img)
    keys = ["bnsc", "rrp", "stab", "coh", "minrank", "maxrank", "asym",
            "harm", "ratio", "cosine", "csls", "cmc"]
    buf = {key: [] for key in keys}
    iu = np.triu_indices(k, 1)

    for s0 in range(0, n, chunk):
        vi, vt = val_img[s0:s0 + chunk], val_txt[s0:s0 + chunk]
        s_it = vi @ bank_txt.T
        s_ti = vt @ bank_img.T
        cos = (vi * vt).sum(1)

        top_t = np.argpartition(-s_it, k, axis=1)[:, :k]
        top_i = np.argpartition(-s_ti, k, axis=1)[:, :k]
        bnsc = np.asarray([
            len(set(top_t[r].tolist()) & set(top_i[r].tolist())) / k
            for r in range(len(vi))
        ])
        coh = np.asarray([
            float((bank_txt[top_t[r]] @ bank_txt[top_t[r]].T)[iu].mean())
            for r in range(len(vi))
        ])

        rank_t = (s_it > cos[:, None]).sum(1).astype(np.float64) + 1.0
        rank_i = (s_ti > cos[:, None]).sum(1).astype(np.float64) + 1.0
        rrp = -np.log(rank_t * rank_i)
        minrank = -np.log(np.maximum(rank_t, rank_i))
        maxrank = -np.log(np.minimum(rank_t, rank_i))
        asym = -np.abs(np.log(rank_t) - np.log(rank_i))
        harm = -np.log(2.0 / (1.0 / rank_t + 1.0 / rank_i))
        ratio = np.log(rank_t / rank_i)

        top2t = np.partition(s_it, -2, axis=1)[:, -2:]
        top2i = np.partition(s_ti, -2, axis=1)[:, -2:]
        rx = np.partition(s_it, -k, axis=1)[:, -k:].mean(1)
        rc = np.partition(s_ti, -k, axis=1)[:, -k:].mean(1)
        m1, m2 = cos - top2t[:, -1], cos - top2t[:, -2]
        n1, n2 = cos - top2i[:, -1], cos - top2i[:, -2]
        stab = np.minimum(m2 - m1, n2 - n1)

        for key, val in (
            ("bnsc", bnsc), ("rrp", rrp), ("stab", stab), ("coh", coh),
            ("minrank", minrank), ("maxrank", maxrank), ("asym", asym),
            ("harm", harm), ("ratio", ratio), ("cosine", cos),
            ("csls", 2 * cos - rx - rc),
            ("cmc", -0.5 * ((cos - top2t[:, -1]) + (cos - top2i[:, -1]))),
        ):
            buf[key].append(val)
    return {key: np.concatenate(vals) for key, vals in buf.items()}


def _cv(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    lin, gbm = [], []
    for tr, te in skf.split(x, y):
        p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        p.fit(x[tr], y[tr])
        lin.append(roc_auc_score(y[te], p.predict_proba(x[te])[:, 1]))
        g = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        g.fit(x[tr], y[tr])
        gbm.append(roc_auc_score(y[te], g.predict_proba(x[te])[:, 1]))
    return {"linear": float(np.mean(lin)), "gbm": float(np.mean(gbm))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--bank", type=int, default=5000)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    val = load_split(args.bench, "val", args.limit)
    vi = enc.encode_images(val["path"].tolist(), batch_size=64, num_workers=4)
    vt = enc.encode_texts(val["caption"].tolist())
    bi, bt = _load_bank(args.bench, args.bank, 0, enc)
    feats = _ops(vi, vt, bi, bt, k=args.k)
    y = val["label"].to_numpy()

    out = {"bench": args.bench, "n_val": int(len(val)), "n_bank": int(len(bi)),
           "k": args.k, "threshold": THRESHOLDS[args.bench], "single": {}, "cv": {}}
    for name, v in feats.items():
        a, b = roc_auc_score(y, v), roc_auc_score(y, -v)
        out["single"][name] = float(max(a, b))
    new = ["bnsc", "rrp", "stab", "coh"]
    out["cv"]["cosine"] = _cv(feats["cosine"][:, None], y)
    out["cv"]["cos+new"] = _cv(np.stack([feats["cosine"]] + [feats[n] for n in new], 1), y)
    out["cv"]["new_only"] = _cv(np.stack([feats[n] for n in new], 1), y)
    out["cv"]["cos+csls"] = _cv(np.stack([feats["cosine"], feats["csls"]], 1), y)
    out["best_cv"] = max(v["gbm"] for v in out["cv"].values())
    out["gap_best_cv"] = float(THRESHOLDS[args.bench] - out["best_cv"])
    write_json(RESULTS_DIR / "gate" / f"{args.bench}_bank_ops.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
