"""Which file did each CRD cell in the main table actually come from?"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = Path("results/fgbench")

print("=== 候选文件 ===")
for pat in ("crdbb_scores_clip_*", "crd_scores_b16_*both*"):
    for f in sorted(glob.glob(str(OUT / pat))):
        z = np.load(f, allow_pickle=True)
        lam = float(z["lam"][0]) if "lam" in z else float("nan")
        print(f"  {Path(f).name:62s} n={len(z['y']):6d} lam={lam:.2f} "
              f"auc={roc_auc_score(z['y'], -z['crd']):.4f}")

print("\n=== 按 facet 汇总（新命名 vs 旧命名） ===")
for ds, fac in (("VG", "Attribute"), ("VG", "Relation"), ("COCO", "Quantity")):
    new = sorted(glob.glob(str(OUT / f"crdbb_scores_clip_{ds}_{fac}_both_row_s*.npz")))
    old = sorted(glob.glob(str(OUT / f"crd_scores_b16_{ds}_{fac}_both_s*.npz")))
    for tag, files in (("crdbb(new)", new), ("offset(old)", old)):
        if not files:
            print(f"  {ds}/{fac:10s} {tag:12s} (none)")
            continue
        aucs = [roc_auc_score(np.load(f, allow_pickle=True)["y"],
                              -np.load(f, allow_pickle=True)["crd"]) for f in files]
        print(f"  {ds}/{fac:10s} {tag:12s} n_seeds={len(files)} "
              f"mean={np.mean(aucs):.4f} sd={np.std(aucs, ddof=1) if len(aucs)>1 else 0:.4f}")
