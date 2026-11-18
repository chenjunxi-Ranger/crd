"""Task-construction and label-semantics audit for the four benchmarks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import PROJECT, RESULTS_DIR, write_json   # noqa: E402

DERIVED = PROJECT / "数据集" / "derived"
FILES = {
    "coco": "mscoco/mscoco__cat__lvl0.40__seed1.csv",
    "flickr": "flickr30k/flickr30k__noun__lvl0.40__seed1.csv",
    "mmimdb": "mmimdb/mmimdb__cat__lvl0.40__seed0.csv",
}


def main() -> int:
    out = {}
    for bench, rel in FILES.items():
        d = pd.read_csv(DERIVED / rel, encoding="utf-8-sig")
        rec = {"rows": int(len(d)), "cols": len(d.columns)}
        rec["noise_rate"] = float(d["is_mislabel"].mean())
        rec["splits"] = d["split"].value_counts().to_dict()
        if "donor_image_id" in d.columns:
            m = d["is_mislabel"] == 1
            dn = d.loc[m, "donor_image_id"].astype(str)
            cur = d.loc[m, "image_id"].astype(str)
            rec["same_image_donor_rate"] = float((dn == cur).mean())
            rec["n_mislabel"] = int(m.sum())
        if "cat_labels" in d.columns:
            m = d["is_mislabel"] == 1
            multi = d.loc[m, "cat_labels"].astype(str).str.split()
            rec["genre_available"] = True
            rec["mean_genres_per_movie"] = float(np.mean([len(x) for x in multi])) if len(multi) else None
            # genre overlap between the movie and its donor (donor row lookup)
            ids = d["image_id"].astype(str)
            idx = {i: k for k, i in enumerate(ids)}
            same, overlap, tot = 0, 0, 0
            for i in np.where(m)[0][:5000]:
                did = str(d.loc[i, "donor_image_id"])
                if did in idx:
                    j = idx[did]
                    a = set(str(d.loc[i, "cat_labels"]).split())
                    b = set(str(d.loc[j, "cat_labels"]).split())
                    tot += 1
                    overlap += int(len(a & b) > 0)
                    same += int(a == b)
            if tot:
                rec["donor_genre_overlap_rate"] = overlap / tot
                rec["donor_genre_identical_rate"] = same / tot
                rec["donor_genre_compared"] = tot
        out[bench] = rec
        print(bench, rec, flush=True)
    write_json(RESULTS_DIR / "bench" / "task_audit.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
