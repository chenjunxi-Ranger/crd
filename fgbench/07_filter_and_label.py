"""Step 07: filter candidates, attach labels, add matched clean controls.

Filters
  - dedupe (target, donor, caption)
  - at most 2 pairs share one donor caption (prevents one sentence dominating)
  - at most 3 pairs share one target image
Quotas are applied per (split, facet) so the test split stays evaluable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgbench.common import BENCH, INTER, jdump  # noqa: E402

FACETS = ["Entity", "Quantity"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-split", default="train:4000,val:1500,test:2000")
    ap.add_argument("--clean-ratio", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    quota = {k: int(v) for k, v in (x.split(":") for x in args.per_split.split(","))}
    parts = []
    for f, fn in (("Entity", "entity_candidates.parquet"),
                  ("Quantity", "quantity_candidates.parquet")):
        d = pd.read_parquet(INTER / fn)
        d["facet"] = f
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates(subset=["target_image_id", "donor_image_id", "caption"])
    d = d.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    d = d.groupby(["facet", "donor_image_id", "caption"]).head(2)
    d = d.groupby(["facet", "target_image_id"]).head(3)

    kept = []
    for (s, f), grp in d.groupby(["split", "facet"]):
        q = quota.get(s, 1000)
        kept.append(grp.head(q))
    d = pd.concat(kept, ignore_index=True)

    caps = pd.read_parquet(INTER / "caption_facts.parquet")
    facts = pd.read_parquet(INTER / "image_facts.parquet")
    name = dict(zip(facts.image_id, facts.file_name))
    own = caps.groupby("image_id")["caption"].apply(list).to_dict()

    noisy = d[["pair_id", "target_image_id", "donor_image_id", "caption",
               "primary_facet", "fact", "evidence", "split", "facet"]].copy()
    noisy["pair_label"] = 1
    noisy["caption_source"] = "original_donor_caption"

    rows = []
    for (s, f), grp in d.groupby(["split", "facet"]):
        n = int(len(grp) * args.clean_ratio)
        tg = grp.target_image_id.unique()
        rng.shuffle(tg)
        for t in tg:
            if len(rows) >= 0 and len([r for r in rows if r["split"] == s and r["facet"] == f]) >= n:
                break
            c = own.get(t)
            if not c:
                continue
            rows.append({"pair_id": f"C{len(rows):07d}", "target_image_id": t,
                         "donor_image_id": t, "caption": c[rng.integers(len(c))],
                         "primary_facet": "Clean", "fact": "", "evidence": {},
                         "split": s, "facet": f, "pair_label": 0,
                         "caption_source": "original_own_caption"})
    clean = pd.DataFrame(rows)

    out = pd.concat([noisy, clean], ignore_index=True)
    out["target_file"] = out.target_image_id.map(name)
    out["donor_file"] = out.donor_image_id.map(name)
    BENCH.mkdir(parents=True, exist_ok=True)
    out.to_parquet(BENCH / "all.parquet", index=False)

    jdump({"rows": int(len(out)),
           "by_split_facet_label": out.groupby(["split", "facet", "pair_label"])
                                     .size().rename("n").reset_index()
                                     .to_dict("records"),
           "filter_rates": {"raw_entity": int(len(pd.read_parquet(INTER / "entity_candidates.parquet"))),
                            "raw_quantity": int(len(pd.read_parquet(INTER / "quantity_candidates.parquet"))),
                            "after_filter": int(len(d))}},
          BENCH / "filter_audit.json")
    print(out.groupby(["split", "facet", "pair_label"]).size().to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
