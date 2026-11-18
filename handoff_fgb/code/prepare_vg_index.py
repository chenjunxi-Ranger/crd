"""Build the Visual Genome image_id -> relative path index if it is missing.

Usage:
  export FGB_VG=/path/to/VisualGenome
  python prepare_vg_index.py
Writes image_index.parquet next to this script's parent data/ directory, or to
$FGB_VG_INDEX if set.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VG = Path(os.environ.get("FGB_VG", "/path/to/VisualGenome"))
OUT = Path(os.environ.get("FGB_VG_INDEX", ROOT / "data" / "image_index.parquet"))


def main() -> int:
    rows = []
    for sub in ("VG_100K", "VG_100K_2"):
        d = VG / sub
        if not d.is_dir():
            print(f"[skip] {d} not found")
            continue
        for p in d.glob("*.jpg"):
            try:
                rows.append((int(p.stem), f"{sub}/{p.name}"))
            except ValueError:
                continue
    if not rows:
        raise SystemExit("no VG jpg files found; check FGB_VG")
    out = pd.DataFrame(rows, columns=["image_id", "rel_path"]).drop_duplicates("image_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {len(out)} entries to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
