"""Recover the VG150 official split from VG-SGG-with-attri.h5.

The h5 stores `split` per image *index* but no image ids, so the index->id
ordering must be recovered.  We fingerprint candidate orderings by comparing
the per-image object counts implied by (img_to_first_box, img_to_last_box)
against the counts computed from objects.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

H5 = Path(r"D:\迅雷下载\VG-SGG-with-attri.h5")
VG = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
OUT = Path(r"D:\PyCharm\PythonProject\异常标签检测\mymodel7\fgbench\data\intermediate\vg")


def main() -> int:
    with h5py.File(H5, "r") as f:
        split = f["split"][:]
        first = f["img_to_first_box"][:]
        last = f["img_to_last_box"][:]
    print("split counts:", dict(Counter(split.tolist())), flush=True)
    n_img = len(split)
    print("n images in h5:", n_img, flush=True)

    counts = (last - first + 1).astype(np.int64)
    counts[first < 0] = 0                       # images without annotations
    print("images with 0 boxes:", int((counts == 0).sum()),
          "| total boxes:", int(counts.sum()), flush=True)

    meta = json.load(open(VG / "image_data.json", encoding="utf-8"))
    ids_meta = [m["image_id"] for m in meta]
    print("image_data.json images:", len(ids_meta), flush=True)

    objs = json.load(open(VG / "objects.json", encoding="utf-8"))
    obj_count = {im["image_id"]: len(im["objects"]) for im in objs}
    print("objects.json images:", len(obj_count),
          "| with >=1 box:", sum(1 for v in obj_count.values() if v > 0), flush=True)

    candidates = {
        "sorted_meta": sorted(ids_meta),
        "meta_order": ids_meta,
        "sorted_with_boxes": sorted(k for k, v in obj_count.items() if v > 0),
        "objects_order": [im["image_id"] for im in objs],
    }
    best = None
    for name, order in candidates.items():
        if len(order) != n_img:
            print(f"  {name:20s} len={len(order)} != {n_img} -> skip", flush=True)
            continue
        c = np.array([obj_count.get(i, 0) for i in order], dtype=np.int64)
        match = int((c == counts).sum())
        rate = match / n_img
        print(f"  {name:20s} per-image count match {match}/{n_img} = {rate:.4f}",
              flush=True)
        if best is None or rate > best[0]:
            best = (rate, name, order)

    if best is None:
        print("no candidate ordering matched; abort", flush=True)
        return 1
    rate, name, order = best
    print(f"\nchosen ordering: {name} (match {rate:.4f})", flush=True)
    if rate < 0.99:
        print("WARNING: ordering not verified; refusing to write split", flush=True)
        return 1

    lab = {0: "train", 1: "val", 2: "test"}
    smap = {int(i): lab.get(int(s), f"unk{s}") for i, s in zip(order, split)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vg150_split.json").write_text(json.dumps(smap), encoding="utf-8")
    print("wrote", OUT / "vg150_split.json",
          dict(Counter(smap.values())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
