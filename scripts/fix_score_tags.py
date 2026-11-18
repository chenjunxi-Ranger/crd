"""Remove the two facet copies that were NOT actually re-run under BLIP-ITC."""
from __future__ import annotations

import os

OUT = "results/fgbench"
BAD = [
    "score_lemon_blipitc_COCO_Entity.npz",
    "score_lemon_blipitc_VG_Action.npz",
    "score_traced_blipitc_COCO_Entity.npz",
    "score_traced_blipitc_VG_Action.npz",
]
for b in BAD:
    p = os.path.join(OUT, b)
    if os.path.exists(p):
        os.remove(p)
        print("removed", b)
    else:
        print("missing", b)
