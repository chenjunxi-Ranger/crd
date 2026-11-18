"""Audit 13: shuffled-image control.

Replace each row's image embedding with another row's image from the same facet.
If AUROC stays high, the score is not actually using image-text correspondence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cache import cos, get       # noqa: E402
from fgbench.common import BENCH  # noqa: E402


def main(seed: int = 0, reps: int = 5) -> int:
    d, img, txt = get("test")
    rng = np.random.default_rng(seed)
    out = {}
    for facet in ("Entity", "Quantity", "All"):
        m = np.ones(len(d), bool) if facet == "All" else (d.facet == facet).to_numpy()
        y = d.pair_label[m].to_numpy()
        base = roc_auc_score(y, -cos(d, img, txt)[m])
        vals = []
        for _ in range(reps):
            perm = rng.permutation(len(d))
            # keep facet composition by shuffling within the facet mask
            idx = np.where(m)[0]
            img_sh = img.copy()
            img_sh[idx] = img[rng.permutation(idx)]
            vals.append(roc_auc_score(y, -cos(d, img_sh, txt)[m]))
        out[facet] = {"real": float(base), "shuffled_mean": float(np.mean(vals)),
                      "shuffled_sd": float(np.std(vals)), "reps": reps}
    (BENCH / "audit_13_shuffled.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
           for k, v in out.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
