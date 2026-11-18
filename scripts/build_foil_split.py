"""Build the FOIL validation split (image-disjoint) from the official train file.

Protocol (CONSTRAINTS.md C6)
----------------------------
* source: ``数据集/FOIL-COCO/pairs_train.csv`` (197,788 pairs / 65,697 images)
* target: 10,000 pairs
* disjointness: whole images are assigned, never pairs, so an image appears on
  exactly one side of the split
* seed: fixed (default 0); the manifest records the image-set hash so later runs
  can prove they used the same split

The remaining train images are *not* used by any training procedure (C2); they
are kept only so the manifest can document the pool size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv"
DEFAULT_OUT = PROJECT / "mymodel7" / "data"


def image_hash(ids: np.ndarray) -> str:
    joined = ",".join(sorted(str(i) for i in ids.tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def build(src: Path, out_dir: Path, n_pairs: int, seed: int) -> dict:
    df = pd.read_csv(src, encoding="utf-8-sig")
    needed = {"pair_id", "image_id", "path", "positive_caption", "negative_caption"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{src} missing columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    images = df["image_id"].drop_duplicates().to_numpy()
    rng.shuffle(images)

    counts = df.groupby("image_id").size()
    picked: list = []
    total = 0
    # capacity-aware greedy: never overshoot, so the split is exactly n_pairs
    # (whole images only, which keeps the two sides image-disjoint)
    for img in images:
        c = int(counts[img])
        if total + c <= n_pairs:
            picked.append(img)
            total += c
        if total == n_pairs:
            break
    val = df[df["image_id"].isin(picked)].copy()
    rest = df[~df["image_id"].isin(picked)]

    out_dir.mkdir(parents=True, exist_ok=True)
    val_path = out_dir / f"foil_val_{n_pairs // 1000}k.csv"
    val.to_csv(val_path, index=False, encoding="utf-8-sig")

    manifest = {
        "source": str(src),
        "output": str(val_path),
        "seed": int(seed),
        "requested_pairs": int(n_pairs),
        "val_pairs": int(len(val)),
        "val_images": int(val["image_id"].nunique()),
        "train_pool_images": int(df["image_id"].nunique()),
        "held_out_pairs": int(len(rest)),
        "held_out_images": int(rest["image_id"].nunique()),
        "image_disjoint": bool(not set(val["image_id"]).intersection(set(rest["image_id"]))),
        "val_image_hash": image_hash(val["image_id"].unique()),
        "val_pair_hash": hashlib.sha256(
            ",".join(sorted(val["pair_id"].astype(str))).encode("utf-8")
        ).hexdigest()[:32],
        "mean_caption_len": {
            "positive": float(val["positive_caption"].str.split().str.len().mean()),
            "negative": float(val["negative_caption"].str.split().str.len().mean()),
        },
    }
    (out_dir / "foil_val_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pairs", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    manifest = build(args.src, args.out_dir, args.pairs, args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
