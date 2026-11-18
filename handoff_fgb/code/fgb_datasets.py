"""FGBench adapter for the official TRACED pipeline.

Each row is one (image, caption) pair. `gold_sentence` is the caption that belongs
to the image in the same reciprocal cycle; `sentence` is the observed caption;
`is_mislabel` is FGBench's pair label. A 500/100/100 subset keeps the official
BLIP trajectory tractable on one 8 GB GPU.
"""
import os
from pathlib import Path

import pandas as pd

from lib.datasets.dataloader import CaptioningDataset
from lib.datasets.utils import generic_transform

BENCH = Path(os.environ.get(
    "FGB_BENCH",
    "/mnt/d/PyCharm/PythonProject/异常标签检测/mymodel7/fgbench/data/fgbench_v1_strict"))
MSCOCO = Path(os.environ.get(
    "FGB_COCO", "/mnt/d/PyCharm/PythonProject/异常标签检测/数据集/MS-COCO"))
VG_ROOT = Path(os.environ.get(
    "FGB_VG", "/mnt/d/PyCharm/PythonProject/异常标签检测/数据集/VisualGenome"))
VG_INDEX = Path(os.environ.get(
    "FGB_VG_INDEX",
    "/mnt/d/PyCharm/PythonProject/异常标签检测/mymodel7/fgbench/data/intermediate/vg/image_index.parquet"))
FACETS = {"fgb_entity": ("COCO", "Entity"), "fgb_quantity": ("COCO", "Quantity"),
          "fgb_attr": ("VG", "Attribute"), "fgb_action": ("VG", "Action"),
          "fgb_relation": ("VG", "Relation")}
_VG = None


def _vg():
    global _VG
    if _VG is None:
        _VG = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()
    return _VG


def _rows(split, name):
    dataset, facet = FACETS[name]
    d = pd.read_parquet(BENCH / f"{split}.parquet")
    d = d[(d.dataset == dataset) & (d.facet == facet)].copy()
    if dataset == "VG":
        vg = _vg()
        d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
        d = d[d.iid.isin(vg)]
        d["path"] = [str(VG_ROOT / vg[i]) for i in d.iid]
    else:
        d["path"] = [str(MSCOCO / str(f)) for f in d.target_file]
    own = (d[d.pair_label == 0].set_index(["cycle_id", "target_image_id"])
           .caption.to_dict())
    d["gold_sentence"] = [own[(c, i)] for c, i in
                          zip(d.cycle_id, d.target_image_id)]
    d["sentence"] = d.caption.astype(str)
    d["is_mislabel"] = d.pair_label.astype(int)
    d["filepath"] = ""
    d["filename"] = d.path.map(lambda p: Path(p).name)
    return d.reset_index(drop=True)


def load_fgb(name, n_train=0, n_val=0, n_test=0):
    """n=0 means the full split; a positive n is a stratified subset."""
    out = []
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        base = _rows(split, name)
        d = (base.sample(n=min(n, len(base)), random_state=0)
             if n and n < len(base) else base)
        out.append(CaptioningDataset(d.reset_index(drop=True), generic_transform,
                                     name, False))
    return tuple(out)
