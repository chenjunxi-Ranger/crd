"""TRACED-style trajectory baseline, implemented from the paper description.

Offline constraint: the official repo cannot be cloned, so this is a faithful
re-implementation of the *mechanism* (iterative token deletion guided by the
scorer + trajectory features + a lightweight classifier), not a bit-exact port.

Trajectory (T steps):
    at each step, try deleting every content token, keep the deletion that most
    improves s(I, c'), record the gain; stop early if no deletion improves.

Features (per pair): s0, total gain, per-step gains, #steps with gain > 0,
mean token-level best gain, fraction of tokens deleted, residual token overlap.
"""
from __future__ import annotations

import argparse
import re
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

STOP = set("a an the is are was were be of in on at to for with and or by from as it its this that".split())


def tokens(c):
    return [w for w in re.findall(r"[A-Za-z']+", c)]


def candidates(toks):
    idx = [i for i, w in enumerate(toks) if w.lower() not in STOP and len(w) > 2]
    return idx or list(range(len(toks)))


def trajectory(sc, img, cap, steps=4):
    toks = tokens(cap)
    cur, s_cur = list(toks), float(sc.score(img, cap))
    gains, feats = [], []
    for _ in range(steps):
        idx = candidates(cur)
        if not idx:
            gains.append(0.0); continue
        cands = [" ".join(cur[:i] + cur[i + 1:]) for i in idx]
        cands = [c if c.strip() else cap for c in cands]
        vals = sc.score_many([img] * len(cands), cands)
        best = int(np.argmax(vals))
        gain = float(vals[best] - s_cur)
        gains.append(gain)
        if gain <= 1e-6:
            break
        s_cur = float(vals[best])
        cur = tokens(cands[best])
    kept = set(w.lower() for w in cur)
    orig = set(w.lower() for w in toks)
    overlap = len(kept & orig) / max(1, len(orig))
    gp = (gains + [0.0, 0.0, 0.0, 0.0])[:4]
    return [float(s_cur), float(sum(gains)), *[float(g) for g in gp],
            float(sum(1 for g in gains if g > 1e-6)),
            float(np.mean(gains)) if gains else 0.0, float(1 - overlap)]


def build(sc, df, steps):
    X = np.array([trajectory(sc, df["path"].iloc[i], str(df["caption"].iloc[i]), steps)
                  for i in range(len(df))])
    return X


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco", choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--backbone", default="clip", choices=["clip", "fg"])
    args = ap.parse_args()

    sc = build_scorer(args.backbone)
    tr = load_split(args.bench, "train", args.train)
    va = load_split(args.bench, "val", args.val)
    print(f"train {len(tr)} val {len(va)} steps {args.steps}", flush=True)

    Xtr = build(sc, tr, args.steps)
    Xva = build(sc, va, args.steps)
    ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()

    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xva)[:, 1]
    auc_trace = float(roc_auc_score(yva, p))

    # baselines on the same rows
    cos = np.array([sc.score(va["path"].iloc[i], str(va["caption"].iloc[i]))
                    for i in range(len(va))])
    auc_cos = float(roc_auc_score(yva, -cos))

    out = {"bench": args.bench, "backbone": args.backbone, "n_train": int(len(tr)),
           "n_val": int(len(va)), "steps": args.steps,
           "auc_cosine": auc_cos, "auc_traced_repro": auc_trace,
           "delta": auc_trace - auc_cos,
           "calls_per_sample_max": 1 + args.steps * 8}
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"traced_{args.backbone}_{args.bench}.json", out)
    np.savez_compressed(RESULTS_DIR / "gate" / f"traced_{args.backbone}_{args.bench}.npz",
                        Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, cos=cos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
