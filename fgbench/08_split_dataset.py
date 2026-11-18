"""Step 08: materialise train/val/test parquet + leakage audit."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgbench.common import BENCH, jdump  # noqa: E402


def main() -> int:
    d = pd.read_parquet(BENCH / "all.parquet")
    for s in ("train", "val", "test"):
        d[d.split == s].to_parquet(BENCH / f"{s}.parquet", index=False)

    # leakage: an image id must live in exactly one split
    img_split = {}
    bad = 0
    has_donor = "donor_image_id" in d.columns
    for r in d.itertuples():
        ids = {r.target_image_id}
        if has_donor:
            ids.add(r.donor_image_id)
        for i in ids:
            if i in img_split and img_split[i] != r.split:
                bad += 1
            img_split[i] = r.split
    cross = (int((d.target_image_id != d.donor_image_id).sum())
             if has_donor else 0)
    jdump({"images": len(img_split), "rows": int(len(d)),
           "split_conflicts": bad, "noisy_pairs_with_donor": cross,
           "splits": d.split.value_counts().to_dict()},
          BENCH / "split_audit.json")
    print("split conflicts:", bad, "| splits:", d.split.value_counts().to_dict(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
