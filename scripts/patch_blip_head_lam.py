"""Add the val-selected lambda to BLIP checkpoints saved before that field existed.

The value is read from the run's own JSON (same seed / facet), so nothing is
recomputed or invented.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import torch

OUT = Path("results/fgbench")
n = 0
for f in sorted(glob.glob(str(OUT / "crdbb_blip*_head.pt"))):
    p = Path(f)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if "lam" in ck:
        continue
    # crdbb_blipitc_VG_Attribute_row_s0_head.pt -> crdbb_blipitc_VG_Attribute_row_s0.json
    js = p.with_name(p.name.replace("_head.pt", ".json"))
    if not js.is_file():
        print("no json for", p.name)
        continue
    lam = float(json.load(open(js, encoding="utf-8"))["lam"])
    ck["lam"] = lam
    torch.save(ck, p)
    n += 1
    print(f"{p.name}: lam={lam}")
print("patched", n)
