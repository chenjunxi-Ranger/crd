"""Verify that the frozen FGBench splits resolve to images on this server.

Usage:
  export FGB_BENCH=/path/to/fgbench_v1_strict
  export FGB_COCO=/path/to/MS-COCO
  export FGB_VG=/path/to/VisualGenome
  export FGB_VG_INDEX=/path/to/image_index.parquet
  python verify_data.py

Writes missing_images.txt and exits 1 if anything is missing.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCH = Path(os.environ.get("FGB_BENCH", ROOT / "data" / "fgbench_v1_strict"))
COCO = Path(os.environ.get("FGB_COCO", "/path/to/MS-COCO"))
VG = Path(os.environ.get("FGB_VG", "/path/to/VisualGenome"))
VG_INDEX = Path(os.environ.get("FGB_VG_INDEX", ROOT / "data" / "image_index.parquet"))


def main() -> int:
    vg = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()
    missing = []
    for split in ("train", "val", "test"):
        d = pd.read_parquet(BENCH / f"{split}.parquet")
        print(f"\n[{split}]")
        for facet, g in d.groupby("facet"):
            if g.dataset.iloc[0] == "COCO":
                paths = [COCO / str(f) for f in g.target_file]
            else:
                ids = g.target_image_id.str.split(":").str[1].astype(int)
                paths = [VG / str(vg[i]) for i in ids if i in vg]
            bad = [str(p) for p in paths if not p.is_file()]
            print(f"  {facet:10s} rows={len(g):6d} missing={len(bad):5d}")
            missing.extend(bad)
            # cycle integrity: 4 rows per cycle, 2 clean
            per = g.groupby("cycle_id").size()
            clean = g[g.pair_label == 0].groupby("cycle_id").size()
            if not (per == 4).all() or not (clean.reindex(per.index).fillna(0) == 2).all():
                print(f"    [warn] {facet}: cycle shape is not 4 rows / 2 clean")
    if missing:
        out = ROOT / "missing_images.txt"
        out.write_text("\n".join(missing), encoding="utf-8")
        print(f"\n{len(missing)} missing images; list -> {out}")
        return 1
    print("\nOK: all images exist; every cycle has 4 rows and 2 clean pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
