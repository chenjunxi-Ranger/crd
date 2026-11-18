"""Inspect SugarCrepe and audit image overlap against FGBench COCO splits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SC = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\SugarCrepe\sugar-crepe-main")
IMG = SC / "data" / "coco" / "images" / "val2017"
KP = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\karpathy_split\dataset_coco.json")

SUBSETS = ["add_obj", "add_att", "replace_obj", "replace_att",
           "replace_rel", "swap_obj", "swap_att"]


def main() -> int:
    total = 0
    ids = set()
    for s in SUBSETS:
        raw = json.load(open(SC / "data" / f"{s}.json", encoding="utf-8"))
        rows = list(raw.values()) if isinstance(raw, dict) else raw
        total += len(rows)
        for r in rows:
            ids.add(int(Path(r["filename"]).stem))     # filenames are zero-padded
        print(f"{s}: {len(rows)}  keys={list(rows[0].keys())}")
        if s == "replace_rel":
            print("   sample:", {k: str(v)[:70] for k, v in rows[0].items()})
    print(f"\ntotal pairs {total}, unique images {len(ids)}")
    have = {p.stem for p in IMG.glob("*.jpg")}
    print(f"images on disk {len(have)}, missing {len(ids - have)}")

    split_of = {}
    for im in json.load(open(KP, encoding="utf-8"))["images"]:
        split_of[int(im["cocoid"])] = "train" if im["split"] in ("train", "restval") \
            else im["split"]
    tr = {k for k, v in split_of.items() if v == "train"}
    te = {k for k, v in split_of.items() if v == "test"}
    print(f"\nSugarCrepe images in Karpathy TRAIN : {len(ids & tr)}")
    print(f"SugarCrepe images in Karpathy TEST  : {len(ids & te)}")

    d = pd.read_parquet(ROOT / "fgbench" / "data" / "fgbench_v1_strict" / "train.parquet")
    fb = d[d.dataset == "COCO"]
    fbi = set(fb.target_image_id.str.split(":").str[1])
    print(f"SugarCrepe images in FGBench-strict TRAIN: {len(ids & fbi)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
