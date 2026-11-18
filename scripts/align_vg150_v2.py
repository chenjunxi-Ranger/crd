"""Retry the index->id alignment, now emulating VG150's 150-class filter."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import h5py

H5 = Path(r"D:\迅雷下载\VG-SGG-with-attri.h5")
VG = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")


def align(C, H, max_skip=3000):
    i = j = 0
    kept, skips = [], []
    while i < len(C) and j < len(H):
        if C[i] == H[j]:
            kept.append(i); i += 1; j += 1
        else:
            if len(skips) >= max_skip:
                return None, None
            skips.append(i); i += 1
    if j != len(H):
        return None, None
    return kept, skips


def try_order(name, order, per_img_count, H):
    C = [per_img_count.get(i, 0) for i in order]
    kept, skips = align(C, H)
    if kept is None:
        print(f"  {name:28s} FAILED", flush=True)
        return None
    rate = len(kept) / len(H)
    print(f"  {name:28s} kept {len(kept)}/{len(H)} = {rate:.4f}  skips {len(skips)}",
          flush=True)
    return (rate, order, kept, skips) if rate > 0.99 else None


def main() -> int:
    with h5py.File(H5, "r") as f:
        split = f["split"][:]
        first = f["img_to_first_box"][:]
        last = f["img_to_last_box"][:]
    H = (last - first + 1).astype(int)
    H[first < 0] = 0

    meta = json.load(open(VG / "image_data.json", encoding="utf-8"))
    objs = json.load(open(VG / "objects.json", encoding="utf-8"))
    ids = [m["image_id"] for m in meta]

    freq = Counter()
    per_img = {}
    for im in objs:
        names = [o["names"][0] for o in im["objects"] if o.get("names")]
        per_img[im["image_id"]] = names
        freq.update(names)

    best = None
    for k in (150, 200, 300, 500, 1000):
        top = {n for n, _ in freq.most_common(k)}
        cnt = {i: sum(1 for n in ns if n in top) for i, ns in per_img.items()}
        for label, order in (("sorted", sorted(ids)), ("file_order", ids)):
            r = try_order(f"top{k}_{label}", order, cnt, H)
            if r and (best is None or r[0] > best[0]):
                best = r

    if best is None:
        print("\nno ordering matched; h5 does not carry enough info to recover ids",
              flush=True)
        return 1

    rate, order, kept, skips = best
    lab = {0: "train", 1: "val", 2: "test"}
    smap = {int(order[k]): lab.get(int(s), f"unk{int(s)}")
            for k, s in zip(kept, split)}
    print("\nbest ordering rate", round(rate, 4), "split", dict(Counter(smap.values())),
          flush=True)
    out = VG.parent / "vg150_split.json"
    out.write_text(json.dumps({str(k): v for k, v in smap.items()}), encoding="utf-8")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
