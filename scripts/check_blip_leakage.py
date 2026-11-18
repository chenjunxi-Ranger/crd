"""Check whether FGBench test images of a facet ever occur in Karpathy train.

blip-itm-base-coco was fine-tuned on COCO retrieval (train+restval), so any
overlap with a COCO facet is a training-data leak, not a generalisation result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fgbench.common import BENCH  # noqa: E402

KP = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\karpathy_split\dataset_coco.json")


def main() -> int:
    split_of = {}
    for im in json.load(open(KP, encoding="utf-8"))["images"]:
        split_of[int(im["cocoid"])] = "train" if im["split"] in ("train", "restval") \
            else im["split"]
    train_ids = {k for k, v in split_of.items() if v == "train"}

    d = pd.read_parquet(BENCH / "test.parquet")
    d = d[d.dataset == "COCO"].copy()
    d["cocoid"] = d.target_image_id.str.split(":").str[1].astype(int)
    for facet in sorted(d.facet.unique()):
        sub = d[d.facet == facet]
        imgs = set(sub.cocoid)
        leak = imgs & train_ids
        print(f"{facet}: test images {len(imgs)}, in Karpathy train {len(leak)} "
              f"({len(leak)/max(1,len(imgs))*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
