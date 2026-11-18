"""Cache CLIP B/16 embeddings for the COCO-Entity test split (needed by the strict table)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\mymodel7")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH  # noqa: E402

OUT = ROOT / "results" / "fgbench"
B16 = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\clip-vit-base-patch16")


def main() -> int:
    from PIL import Image
    from _cache import resolve
    from src.encoders import JointEncoder

    enc = JointEncoder(str(B16))
    model, proc, dev = enc.model, enc.processor, enc.device

    d = pd.read_parquet(BENCH / "test.parquet")
    d = d[(d.dataset == "COCO") & (d.facet == "Entity")].reset_index(drop=True)
    d["k"] = d.target_image_id.astype(str)
    paths = {}
    for k, f in zip(d.k, d.target_file):
        paths.setdefault(k, resolve(f))

    @torch.no_grad()
    def enc_i(ps, bs=64):
        out = []
        for i in range(0, len(ps), bs):
            px = proc(images=[Image.open(p).convert("RGB") for p in ps[i:i + bs]],
                      return_tensors="pt")["pixel_values"].to(dev)
            vo = model.vision_model(pixel_values=px)
            out.append(F.normalize(model.visual_projection(vo.pooler_output).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def enc_t(caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            tk = proc(text=list(caps[i:i + bs]), return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(dev)
            to = model.text_model(**tk)
            out.append(F.normalize(model.text_projection(to.pooler_output).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    ids = list(paths)
    U = enc_i([paths[i] for i in ids])
    pos = {i: k for k, i in enumerate(ids)}
    U = U[[pos[k] for k in d.k]]
    V = enc_t(d.caption.astype(str).tolist())
    np.savez_compressed(OUT / "crd_b16_COCO_Entity_test.npz", U=U, V=V)
    d.to_parquet(OUT / "crd_b16_COCO_Entity_test.meta.parquet", index=False)
    print(f"cached COCO-Entity test rows {len(d)} images {len(ids)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
