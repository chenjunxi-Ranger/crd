"""Inspect the frozen FGBench v1 splits: sizes, structure, and example cycles."""
from __future__ import annotations

import pandas as pd

B = "fgbench/data/fgbench_v1"

d = pd.read_parquet(f"{B}/all.parquet")
print("columns:", list(d.columns))
print("rows", len(d), "cycles", d.cycle_id.nunique())
print()

p = d.pivot_table(index=["dataset", "facet"], columns=["split", "pair_label"],
                  values="pair_id", aggfunc="count", fill_value=0)
print("== rows by dataset/facet/split(label) ==")
print(p.to_string())
print()

print("== per split: unique images / captions / cycles ==")
g = d.groupby("split").agg(rows=("pair_id", "size"),
                           images=("target_image_id", "nunique"),
                           captions=("caption_id", "nunique"),
                           cycles=("cycle_id", "nunique"))
print(g.to_string())
print()

print("== text granularity ==")
print(pd.crosstab(d.dataset, d.text_granularity).to_string())
print()

for facet in ("Entity", "Quantity", "Attribute", "Action", "Relation"):
    sub = d[d.facet == facet]
    cid = sub[sub.split == "test"].cycle_id.iloc[0]
    print(f"===== {facet} | example cycle {cid} =====")
    cols = ["pair_id", "target_image_id", "caption", "pair_label",
            "split", "text_granularity", "fact"]
    print(sub[sub.cycle_id == cid][cols].to_string(index=False))
    print("evidence:", sub[sub.cycle_id == cid].evidence.iloc[0])
    print()
