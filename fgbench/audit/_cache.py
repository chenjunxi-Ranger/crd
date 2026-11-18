"""Cache CLIP embeddings for a benchmark split (image / text), reused by all audits."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import BENCH, PROJ   # noqa: E402

IMG_ROOT = Path(os.environ.get("COCO_IMG", str(PROJ / "数据集" / "MS-COCO" / "val2014")))
COCO_ROOT = PROJ / "数据集" / "MS-COCO"


def resolve(fname: str) -> str:
    """v0.2x stores bare filenames; v0.3 stores 'train2014/xxx.jpg'."""
    for root in (IMG_ROOT, COCO_ROOT):
        p = root / fname
        if p.exists():
            return str(p)
    return str(IMG_ROOT / fname)


def get(split: str, scorer: str = "clip"):
    p = BENCH / f"emb_{scorer}_{split}.npz"
    if p.is_file():
        z = np.load(p, allow_pickle=True)
        return pd.read_parquet(BENCH / f"{split}.parquet"), z["img"], z["txt"]

    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    enc = JointEncoder(CLIP_DIR)
    d = pd.read_parquet(BENCH / f"{split}.parquet")
    paths = [resolve(f) for f in d.target_file]
    img = enc.encode_images(paths, batch_size=64, num_workers=4)
    txt = enc.encode_texts(d.caption.astype(str).tolist())
    np.savez_compressed(p, img=img, txt=txt)
    return d, img, txt


def cos(d, img, txt):
    return (img * txt).sum(1)
