"""Merge the five per-facet P0-3 files and apply Holm over the 24-cell family."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[3] / "results" / "fgbench" / "p0_baselines"
FACETS = ["Entity", "Quantity", "Attribute", "Action", "Relation"]


def holm(p):
    order = np.argsort(p)
    m, out, run = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


df = pd.concat([pd.read_csv(CACHE / f"p0_3_{f}.csv", encoding="utf-8-sig")
                for f in FACETS], ignore_index=True)
au = df[df.metric == "auroc"].copy().reset_index(drop=True)
au["p_holm"] = np.round(holm(au.p_one_sided.to_numpy()), 4)
au["significant"] = au.p_holm < 0.05
au.to_csv(CACHE / "p0_3_final.csv", index=False, encoding="utf-8-sig")
print(au[["facet", "method", "auroc_base", "auroc_crd", "value",
          "ci_lo", "ci_hi", "p_one_sided", "p_holm", "significant"]].to_string(index=False))
print(f"\ncells={len(au)}  significant={int(au.significant.sum())}")
print("\n2AFC:")
print(df[df.metric == "2afc"][["facet", "method", "correct", "n_cycles",
                               "value", "ci_lo", "ci_hi"]].to_string(index=False))
