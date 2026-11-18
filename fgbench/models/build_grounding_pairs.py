"""Recover (image, phrase, box) supervision for GFA from VG region descriptions.

Our VG cycles use region-description phrases as captions.  Only CLEAN rows have a
valid box (the phrase really describes that image).  This script matches each
clean row back to its region box and reports coverage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import BENCH, INTER, jdump  # noqa: E402

VG = INTER / "vg"


def main() -> int:
    reg = pd.read_parquet(VG / "regions.parquet")
    reg["phrase_n"] = reg.phrase.astype(str).str.lower().str.strip()
    key = {(int(i), p): (x, y, w, h) for i, p, x, y, w, h in
           zip(reg.image_id, reg.phrase_n, reg.x, reg.y, reg.w, reg.h)}

    d = pd.read_parquet(BENCH / "all.parquet")
    d = d[(d.dataset == "VG") & (d.pair_label == 0)].copy()   # clean rows only
    d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
    d["phrase_n"] = d.caption.astype(str).str.lower().str.strip()

    hit = [key.get((i, p)) for i, p in zip(d.iid, d.phrase_n)]
    d["box"] = hit
    ok = d.box.notna()
    print(f"clean VG rows {len(d)}, box recovered {int(ok.sum())} "
          f"({ok.mean()*100:.1f}%)", flush=True)
    print(d.groupby("facet").box.apply(lambda s: f"{s.notna().mean()*100:.1f}%"),
          flush=True)

    sel = d[ok]
    boxes = [b for b in sel.box]
    out = pd.DataFrame({
        "cycle_id": sel.cycle_id.values,
        "target_image_id": sel.target_image_id.values,
        "iid": sel.iid.values,
        "caption": sel.caption.values,
        "facet": sel.facet.values,
        "split": sel.split.values,
        "x": [b[0] for b in boxes],
        "y": [b[1] for b in boxes],
        "w": [b[2] for b in boxes],
        "h": [b[3] for b in boxes]})
    out.to_parquet(INTER / "vg_grounding_pairs.parquet", index=False)
    jdump({"clean_rows": int(len(d)), "with_box": int(ok.sum()),
           "coverage": float(ok.mean()),
           "by_split": {k: int(v) for k, v in out.split.value_counts().items()}},
          INTER / "vg_grounding_pairs_meta.json")
    print(out.head(3).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
