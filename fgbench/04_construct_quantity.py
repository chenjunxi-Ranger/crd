"""Step 04: Quantity candidates.

  donor caption asserts (c, n)  with  n == donor's instance count and n >= 2
  target contains c with a DIFFERENT positive count
  every other class named by the caption exists in both images
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgbench.common import INTER, jdump  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mentions", type=int, default=2)
    ap.add_argument("--max-per-caption", type=int, default=8)
    ap.add_argument("--cap", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    facts = pd.read_parquet(INTER / "image_facts.parquet")
    caps = pd.read_parquet(INTER / "caption_facts.parquet")
    caps = caps[caps.quantities.map(len) > 0]

    cnt, split, fname = {}, {}, {}
    for r in facts.itertuples():
        cnt.setdefault(r.image_id, {})[r.class_name] = r.count
        split[r.image_id] = r.split
        fname[r.image_id] = r.file_name

    idx = defaultdict(set)
    for im, cs in cnt.items():
        for c in cs:
            idx[c].add(im)
    by_split = {s: {c: {i for i in v if split[i] == s} for c, v in idx.items()}
                for s in ("train", "val", "test")}

    rows = []
    for r in caps.itertuples():
        d = r.image_id
        if d not in cnt:
            continue
        qty = dict(r.quantities)
        mentioned = list(r.mentioned)
        for c, n in qty.items():
            if cnt[d].get(c, 0) != n or n < 2:
                continue                     # caption number must match the donor image
            others = [x for x in mentioned if x != c]
            if not others:
                continue
            sidx = by_split[split[d]]
            pools = [sidx.get(o, set()) for o in others]
            if any(not p for p in pools):
                continue
            inter = set.intersection(*pools)
            inter = [t for t in inter if t != d and 0 < cnt[t].get(c, 0) != n]
            if not inter:
                continue
            rng.shuffle(inter)
            for t in inter[: args.max_per_caption]:
                rows.append({
                    "pair_id": f"Q{len(rows):07d}",
                    "target_image_id": t, "donor_image_id": d,
                    "target_file": fname[t], "donor_file": fname[d],
                    "caption": r.caption, "pair_label": 1,
                    "primary_facet": "Quantity", "fact": c,
                    "n_mentioned": len(mentioned), "mentioned": mentioned,
                    "evidence": {"fact": c, "caption_number": n,
                                 "donor_count": cnt[d][c], "target_count": cnt[t][c],
                                 "target_evidence": "coco_instance_annotations",
                                 "other_facts_supported_by_target": True},
                    "split": split[t], "caption_source": "original_donor_caption"})
        if len(rows) >= args.cap:
            break

    df = pd.DataFrame(rows)
    df.to_parquet(INTER / "quantity_candidates.parquet", index=False)
    jdump({"rows": int(len(df)),
           "unique_targets": int(df.target_image_id.nunique()) if len(df) else 0,
           "by_split": df.split.value_counts().to_dict() if len(df) else {},
           "facet_counts": df.fact.value_counts().head(20).to_dict() if len(df) else {}},
          INTER / "quantity_meta.json")
    print("quantity:", df.shape, df.split.value_counts().to_dict() if len(df) else {},
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
