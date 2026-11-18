"""Recover index->image_id for the VG150 h5 by aligning per-image object counts.

Hypothesis: the h5 was built from image_data.json sorted by image_id, with a few
images dropped (those that failed to load).  We align the count sequences and
allow a small number of deletions.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

H5 = Path(r"D:\迅雷下载\VG-SGG-with-attri.h5")
VG = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")


def align(C: list[int], H: list[int], max_skip: int = 200):
    """Return (indices of C kept, skip positions) or None."""
    i = j = 0
    kept = []
    skips = []
    while i < len(C) and j < len(H):
        if C[i] == H[j]:
            kept.append(i); i += 1; j += 1
        else:
            # allow skipping entries in C (images absent from h5)
            if len(skips) >= max_skip:
                return None, None
            skips.append(i); i += 1
    if j != len(H):
        return None, None
    return kept, skips


def main() -> int:
    with h5py.File(H5, "r") as f:
        split = f["split"][:]
        first = f["img_to_first_box"][:]
        last = f["img_to_last_box"][:]
    H = (last - first + 1).astype(int)
    H[first < 0] = 0

    meta = json.load(open(VG / "image_data.json", encoding="utf-8"))
    objs = json.load(open(VG / "objects.json", encoding="utf-8"))
    cnt = {im["image_id"]: len(im["objects"]) for im in objs}
    # every image in image_data must appear in objects (verified earlier)
    ids_sorted = sorted(m["image_id"] for m in meta)
    C = [cnt.get(i, 0) for i in ids_sorted]
    print(f"C={len(C)} (sorted ids with counts), H={len(H)} (h5)", flush=True)

    kept, skips = align(C, H)
    if kept is None:
        print("alignment FAILED under sorted(ids) hypothesis", flush=True)
        return 1
    print(f"aligned: kept={len(kept)} skipped={len(skips)}", flush=True)
    if len(skips) > 10:
        print("WARNING: many skips, alignment may be loose", flush=True)

    order = [ids_sorted[k] for k in kept]
    lab = {0: "train", 1: "val", 2: "test"}
    smap = {int(i): lab.get(int(s), f"unk{int(s)}") for i, s in zip(order, split)}
    print("split:", dict(Counter(smap.values())), flush=True)
    print("skipped ids:", [ids_sorted[k] for k in skips][:10], flush=True)

    out = VG.parent / "vg150_split.json"
    out.write_text(json.dumps({str(k): v for k, v in smap.items()}), encoding="utf-8")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
