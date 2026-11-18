"""Copy the current (BLIP-ITC) LEMoN/TRACED score files to backbone-tagged names."""
from __future__ import annotations

import glob
import os
import shutil

OUT = "results/fgbench"
n = 0
for f in sorted(glob.glob(f"{OUT}/score_lemon_*_*.npz")
                + glob.glob(f"{OUT}/score_traced_*_*.npz")):
    b = os.path.basename(f)
    if "blipitc" in b or "clip_" in b:
        continue
    parts = b.split("_")                       # score_<name>_<ds>_<facet>.npz
    if len(parts) != 4:
        continue
    new = f"{OUT}/score_{parts[1]}_blipitc_{parts[2]}_{parts[3]}"
    shutil.copy(f, new)
    print("copied ->", os.path.basename(new))
    n += 1
print("n", n)
