"""Recover official VG150 train/val/test image ids from VG-SGG-with-attri.h5.

Row i of the h5 corresponds to the i-th entry of image_data.json after removing
the four corrupted images {1592, 1722, 4616, 4617}.  split: 0 = train+val pool,
2 = test.  Val = the first 5,000 *valid* images (>=1 box) of the train pool,
matching the Neural-Motifs loader.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import h5py

H5 = Path(r"D:\迅雷下载\VG-SGG-with-attri.h5")
VG = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
OUT = Path(r"D:\PyCharm\PythonProject\异常标签检测\mymodel7\fgbench\splits")
CORRUPTED = {1592, 1722, 4616, 4617}
TARGET = {"train": 68_538, "val": 5_000, "test": 31_876}


def main() -> int:
    with h5py.File(H5, "r") as f:
        split = f["split"][:]
        first_box = f["img_to_first_box"][:]

    meta = json.load(open(VG / "image_data.json", encoding="utf-8"))
    ordered = [int(m["image_id"]) for m in meta
               if int(m["image_id"]) not in CORRUPTED]
    print(f"image_data {len(meta)} -> ordered {len(ordered)} ; h5 rows {len(split)}")
    assert len(ordered) == len(split), "row order assumption broken"
    print("split values:", dict(Counter(split.tolist())))

    valid = first_box >= 0
    print("valid (>=1 box):", int(valid.sum()))

    test_ids = [ordered[i] for i in range(len(ordered)) if split[i] == 2 and valid[i]]
    pool = [ordered[i] for i in range(len(ordered)) if split[i] == 0 and valid[i]]
    print("test candidates", len(test_ids), "| train+val pool", len(pool))

    val_ids = pool[:TARGET["val"]]
    train_ids = pool[TARGET["val"]:]

    got = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
    print("recovered:", got)
    print("target   :", TARGET)
    ok = got == TARGET
    print("MATCH" if ok else "MISMATCH")

    smap = {}
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        for i in ids:
            smap[str(i)] = name
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "vg150_split_ids.json"
    p.write_text(json.dumps(smap), encoding="utf-8")
    print("wrote", p, f"({len(smap)} ids)")
    if not ok:
        print("NOTE: counts differ from the quoted 68,538/5,000/31,876; "
              "recorded as-is instead of forcing numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
