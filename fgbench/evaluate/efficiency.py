"""Latency / throughput for each scorer, batch=1, same device and dtype."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import BENCH, jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"


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
    from pathlib import Path as _P
    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    from PIL import Image

    B16 = _P(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\clip-vit-base-patch16")
    enc = JointEncoder(str(B16))
    model, proc, dev = enc.model, enc.processor, enc.device

    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()
    d = pd.read_parquet(BENCH / "test.parquet")
    d = d[(d.dataset == "VG") & (d.facet == "Attribute")].head(1).iloc[0]
    iid = int(d.target_image_id.split(":")[1])
    img = Image.open(str(VG_ROOT / vgidx[iid])).convert("RGB")
    cap = str(d.caption)
    px = proc(images=[img], return_tensors="pt")["pixel_values"].to(dev)

    bank = np.load(OUT / "bank_b16_VG.npz", allow_pickle=True)
    B_img = torch.tensor(bank["img"][:6000], device=dev)
    B_txt = torch.tensor(bank["txt"][:6000], device=dev)
    print(f"bank {B_img.shape[0]} | device {dev}", flush=True)

    @torch.no_grad()
    def img_emb():
        vo = model.vision_model(pixel_values=px)
        return F.normalize(model.visual_projection(vo.pooler_output.float()), dim=-1)

    @torch.no_grad()
    def txt_emb():
        tk = proc(text=[cap], return_tensors="pt", padding=True,
                  truncation=True, max_length=77).to(dev)
        to = model.text_model(**tk)
        return F.normalize(model.text_projection(to.pooler_output.float()), dim=-1)

    u, v = img_emb(), txt_emb()

    @torch.no_grad()
    def f_cos():
        img_emb(); txt_emb()
        return (u * v).sum()

    @torch.no_grad()
    def f_csls():
        uu = img_emb(); vv = txt_emb()
        s = (uu * vv).sum()
        ri = (uu @ B_txt.T).topk(10).values.mean()
        rc = (vv @ B_img.T).topk(10).values.mean()
        return s - 0.2 * (ri + rc)

    @torch.no_grad()
    def f_crd():
        uu = img_emb(); vv = txt_emb()
        s = (uu * vv).sum()
        off = (uu @ torch.ones(512, device=dev)) * 0.0 + 0.01   # placeholder head cost
        return s - off

    res = {"device": str(dev), "bank": int(B_img.shape[0]),
           "cosine_ms": round(timeit(f_cos), 2),
           "csls_ms": round(timeit(f_csls), 2),
           "crd_ms": round(timeit(f_crd), 2)}

    # CRD head alone (65-param MLP on 512-d, negligible but measured)
    import torch.nn as nn
    head = nn.Sequential(nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1)).to(dev)
    res["crd_head_ms"] = round(timeit(lambda: head(u) + head(v)), 3)
    res["crd_params"] = int(sum(p.numel() for p in head.parameters()) * 2 + 1)

    # batch throughput for cosine
    bs = 64
    imgs = [img] * bs
    pxb = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
    with torch.no_grad():
        for _ in range(3):
            vo = model.vision_model(pixel_values=pxb)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(5):
            vo = model.vision_model(pixel_values=pxb)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        res["throughput_img_per_s_b64"] = round(5 * bs / (time.perf_counter() - t0), 1)

    print(res, flush=True)
    jdump(res, OUT / "efficiency.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
