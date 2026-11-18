"""Rebuild a reference bank with the LoRA-adapted CLIP (Stage 1b).

Same rules as the frozen banks: official train only, no labels, FOIL excludes
every image used by the 10k validation split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lora_stage1 import inject                              # noqa: E402
from src.encoders import JointEncoder                       # noqa: E402
from src.protocol import (CLIP_DIR, PROJECT, RESULTS_DIR, load_split,  # noqa: E402
                          load_foil_val)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--limit", type=int, default=15000)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    model = enc.model
    inject(model)
    ck = torch.load(RESULTS_DIR / "gate" / "lora_weights.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    model.to(enc.device).eval()
    print("loaded lora; missing", len(missing), "unexpected", len(unexpected), flush=True)

    if args.bench == "foil":
        val = load_foil_val(10000)
        val_ids = set(val["image_id"].astype(str))
        df = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                         encoding="utf-8-sig")
        df = df[~df["image_id"].astype(str).isin(val_ids)]
        df = df.sample(min(args.limit, len(df)), random_state=0)
        paths, caps = df["path"].tolist(), df["positive_caption"].astype(str).tolist()
        image_key = df["image_id"].to_numpy()
    else:
        df = load_split(args.bench, "train")
        df = df.sample(min(args.limit, len(df)), random_state=0)
        paths, caps = df["path"].tolist(), df["caption"].astype(str).tolist()
        image_key = df["image_id"].to_numpy()

    img = enc.encode_images(paths, batch_size=64, num_workers=4)
    txt = enc.encode_texts(caps)
    out = RESULTS_DIR / "gate" / f"lora_bank_{args.bench}.npz"
    np.savez_compressed(out, img=img, txt=txt, image_key=image_key)
    print(f"{args.bench}: {len(img)} rows, {len(np.unique(image_key))} images -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
