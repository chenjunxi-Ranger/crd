"""Step 01: per-image fact table from COCO instance annotations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgbench.common import (COCO_ANN, INTER, jdump, load_instances,  # noqa: E402
                            split_of)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="instances_val2014.json")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    counts, boxes, cat, images = load_instances(COCO_ANN / args.instances)
    rows = []
    for im in images:
        iid = im["id"]
        if iid not in counts:
            continue
        for cid, n in counts[iid].items():
            rows.append({"image_id": iid, "file_name": im["file_name"],
                         "class_id": cid, "class_name": cat[cid], "count": n,
                         "boxes": boxes[iid][cid], "split": split_of(iid)})
        if args.max_images and len(rows) > args.max_images * 8:
            pass
    df = pd.DataFrame(rows)
    if args.max_images:
        keep = set(df["image_id"].drop_duplicates().sample(
            min(args.max_images, df["image_id"].nunique()), random_state=0))
        df = df[df["image_id"].isin(keep)]
    INTER.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTER / "image_facts.parquet", index=False)
    jdump({"images": int(df.image_id.nunique()), "classes": int(df.class_name.nunique()),
           "rows": int(len(df)),
           "split_sizes": df.drop_duplicates("image_id")["split"].value_counts().to_dict()},
          INTER / "image_facts_meta.json")
    print(df.shape, df.image_id.nunique(), "images",
          df.drop_duplicates("image_id")["split"].value_counts().to_dict(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
