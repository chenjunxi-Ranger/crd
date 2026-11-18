"""Recompute CRD cycle-mean R^2 cleanly from the cached tensors (index-free)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\mymodel7")
OUT = ROOT / "results" / "fgbench"


def load(split):
    z = np.load(OUT / f"crd_b16_{split}.npz", allow_pickle=True)
    d = pd.read_parquet(OUT / f"crd_b16_{split}.meta.parquet")
    return d, z["U"], z["V"]


for split in ("train", "val", "test"):
    d, U, V = load(split)
    s = (U * V).sum(1)
    d = d.assign(s=s).reset_index(drop=True)
    g = d.groupby("cycle_id")
    cyc = g.agg(m=("s", "mean"), n=("s", "size"))
    print(f"{split}: rows {len(d)} cycles {len(cyc)} "
          f"mean_of_means {cyc.m.mean():.4f} sd {cyc.m.std():.4f} "
          f"sizes {cyc.n.value_counts().to_dict()}")
