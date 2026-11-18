"""Cycle-clustered paired bootstrap for the Action CRD vs cosine."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import glob

files = sorted(glob.glob("results/fgbench/crd_action_scores_s*.npz"))
rng = np.random.default_rng(0)
per_seed = []
for f in files:
    z = np.load(f, allow_pickle=True)
    y, cid = z["y"], z["cycle_id"]
    idx = pd.Series(np.arange(len(y))).groupby(cid).apply(lambda x: x.to_numpy()).to_dict()
    keys = list(idx)
    ds = []
    for _ in range(2000):
        pk = rng.choice(len(keys), len(keys), replace=True)
        ii = np.concatenate([idx[keys[t]] for t in pk])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        ds.append((roc_auc_score(yy, -z["crd"][ii]) - roc_auc_score(yy, -z["raw"][ii])) * 100)
    ds = np.array(ds)
    per_seed.append((float(ds.mean()), float(np.percentile(ds, 2.5)),
                     float(np.percentile(ds, 97.5)), float((ds > 0).mean())))
    print(f"{f.split(chr(92))[-1]}: delta {ds.mean():+.2f} "
          f"[{np.percentile(ds,2.5):+.2f}, {np.percentile(ds,97.5):+.2f}] "
          f"P(>0)={float((ds>0).mean()):.3f}")

m = np.mean([p[0] for p in per_seed])
lo = np.mean([p[1] for p in per_seed])
hi = np.mean([p[2] for p in per_seed])
print(f"\nmean over seeds: {m:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")
