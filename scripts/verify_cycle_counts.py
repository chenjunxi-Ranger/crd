"""Verify FGBench v1 cycle counts and the per-rule breakdown of Relation/Action."""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd

d = pd.read_parquet("fgbench/data/fgbench_v1/all.parquet")

cyc = d.groupby(["dataset", "facet"]).cycle_id.nunique()
print("== cycles per facet ==")
print(cyc.to_string())
print("total cycles", d.cycle_id.nunique(), "rows", len(d))
print()

# one row per cycle keeps evidence unambiguous
one = d[d.pair_label == 0].drop_duplicates("cycle_id")

for facet in ("Relation", "Action"):
    sub = one[one.facet == facet]
    c = Counter()
    for ev in sub.evidence:
        if isinstance(ev, dict):
            p = ev.get("exclusive_pair")
            if p is None:
                c["<missing>"] += 1
            else:
                c[" | ".join(sorted(map(str, p)))] += 1
    print(f"== {facet} cycles by exclusive_pair ({len(sub)} cycles) ==")
    for k, v in c.most_common():
        print(f"   {k:28s} {v}")
    print()

print("== cycles per split per facet ==")
print(d.pivot_table(index=["dataset", "facet"], columns="split", values="cycle_id",
                    aggfunc="nunique", fill_value=0).to_string())
print()
print("== entity difficulty (cycles) ==")
e = one[one.facet == "Entity"]
print(e.difficulty.value_counts().to_string())
