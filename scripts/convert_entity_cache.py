"""Re-serialise the COCO Entity cache into the crdbb_ layout (U, V, S + meta)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

OUT = Path("results/fgbench")
CYCLES = 8000

for split in ("train", "val", "test"):
    src = OUT / f"crd_b16_COCO_Entity_{split}.npz"
    z = np.load(src, allow_pickle=True)
    U, V = z["U"], z["V"]
    S = (U * V).sum(1)
    tag = f"crdbb_clip_COCO_Entity" + (f"_c{CYCLES}" if split == "train" else "")
    dst = OUT / f"{tag}_{split}.npz"
    np.savez_compressed(dst, U=U, V=V, S=S)
    shutil.copy(OUT / f"crd_b16_COCO_Entity_{split}.meta.parquet",
                dst.with_suffix(".meta.parquet"))
    print(f"{split}: {U.shape} -> {dst.name}")
