"""Audit 10: same-image paired test.

For every noisy pair we look for a clean pair on the SAME target image and ask
whether the model scores the true caption higher than the transplanted one:

    Acc = P[ s(I, c_clean) > s(I, c_noisy) ]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cache import cos, get            # noqa: E402
from fgbench.common import BENCH       # noqa: E402


def main(split="test") -> int:
    d, img, txt = get(split)
    d["score"] = -cos(d, img, txt)          # higher = more anomalous

    clean = d[d.pair_label == 0].groupby("target_image_id").first()
    out = {"split": split}
    for facet in ("Entity", "Quantity"):
        noisy = d[(d.facet == facet) & (d.pair_label == 1)]
        joined = noisy.merge(clean[["score", "caption"]], left_on="target_image_id",
                             right_index=True, suffixes=("", "_clean"))
        if not len(joined):
            continue
        acc = float((joined.score_clean < joined.score).mean())  # clean less anomalous
        out[facet] = {"n_pairs": int(len(joined)),
                      "paired_acc": acc,
                      "n_targets": int(joined.target_image_id.nunique())}
    out["overall_n_pairs"] = int(sum(v["n_pairs"] for k, v in out.items()
                                     if isinstance(v, dict)))
    (BENCH / "audit_10_paired.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
