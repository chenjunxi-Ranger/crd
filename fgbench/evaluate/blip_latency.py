"""Latency of BLIP ITC vs ITM, each with and without the CRD head (batch = 1)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import BENCH, jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
WDIR = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\blip-itm-base-coco")


def timeit(fn, warm=5, n=30):
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main() -> int:
    from PIL import Image
    from transformers import BlipForImageTextRetrieval, BlipProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = BlipForImageTextRetrieval.from_pretrained(str(WDIR)).to(dev).eval()
    proc = BlipProcessor.from_pretrained(str(WDIR))
    idx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()

    d = pd.read_parquet(BENCH / "test.parquet")
    r = d[(d.dataset == "VG") & (d.facet == "Attribute")].iloc[0]
    iid = int(r.target_image_id.split(":")[1])
    img = Image.open(str(VG_ROOT / idx[iid])).convert("RGB")
    cap = str(r.caption)
    px = proc(images=[img], return_tensors="pt")["pixel_values"].to(dev)
    tk = proc(text=[cap], return_tensors="pt", padding=True, truncation=True,
              max_length=64).to(dev)

    head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)).to(dev)

    @torch.no_grad()
    def itc():
        h = m.vision_model(pixel_values=px, return_dict=True).last_hidden_state[:, 0, :]
        g = m.text_encoder(input_ids=tk["input_ids"],
                           attention_mask=tk["attention_mask"],
                           return_dict=True).last_hidden_state[:, 0, :]
        return (h * g).sum()

    @torch.no_grad()
    def itm():
        return m(pixel_values=px, input_ids=tk["input_ids"],
                 attention_mask=tk["attention_mask"], use_itm_head=True).itm_score

    with torch.no_grad():
        h = m.vision_model(pixel_values=px, return_dict=True).last_hidden_state[:, 0, :]
        g = m.text_encoder(input_ids=tk["input_ids"],
                           attention_mask=tk["attention_mask"],
                           return_dict=True).last_hidden_state[:, 0, :]
        if hasattr(m, "vision_proj"):          # CRD head consumes the PROJ features
            h = m.vision_proj(h)
        if hasattr(m, "text_proj"):
            g = m.text_proj(g)

    res = {"device": dev,
           "itc_ms": round(timeit(itc), 2),
           "itm_ms": round(timeit(itm), 2),
           "crd_head_ms": round(timeit(lambda: head(h) + head(g)), 3),
           "crd_params": int(sum(p.numel() for p in head.parameters()) * 2 + 1)}
    res["itc_crd_ms"] = round(res["itc_ms"] + res["crd_head_ms"], 2)
    res["itm_crd_ms"] = round(res["itm_ms"] + res["crd_head_ms"], 2)
    print(res, flush=True)
    jdump(res, OUT / "blip_latency.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
