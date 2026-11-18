"""Which manifest does each backbone's Relation evaluation actually use?"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/fgbench")

print("=== VG/Relation test cache 行数（2824=strict / 3024=v1） ===")
for pref in ("crdbb_clip", "crdbb_siglip2", "crdbb_clipL", "crdbb_flava",
             "crdbb_blipitc", "crdbb_blipitm"):
    hits = sorted(glob.glob(str(OUT / f"{pref}_VG_Relation_test.npz")))
    if not hits:
        print(f"  {pref:16s} (no cache)")
        continue
    z = np.load(hits[0], allow_pickle=True)
    n = len(z["U"]) if "U" in z else "?"
    print(f"  {pref:16s} n={n}")

print("\n=== CLIP 的 CRD 分数文件（crd_offset 输出） ===")
for f in sorted(glob.glob(str(OUT / "crd_scores_b16_VG_Relation_both_row_s*.npz"))):
    z = np.load(f, allow_pickle=True)
    print(f"  {Path(f).name}: n={len(z['y'])}")

print("\n=== strict_main_table.csv 中的 Relation 行 ===")
p = OUT / "strict_main_table.csv"
if p.is_file():
    d = pd.read_csv(p)
    print(d[d.facet == "Relation"].to_string(index=False))
else:
    print("  missing")
