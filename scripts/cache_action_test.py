"""Encode the VG-Action test split once so subset metrics can reuse it."""
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

VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
IDX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
OUT = ROOT / "results" / "fgbench"
B16 = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\clip-vit-base-patch16")


def main() -> int:
    from PIL import Image
    from src.encoders import JointEncoder

    enc = JointEncoder(str(B16))
    model, proc, dev = enc.model, enc.processor, enc.device
    idx = pd.read_parquet(IDX).set_index("image_id").rel_path.to_dict()

    d = pd.read_parquet(BENCH / "test.parquet")
    d = d[(d.dataset == "VG") & (d.facet == "Action")].reset_index(drop=True)
    d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
    d = d[d.iid.isin(idx)].reset_index(drop=True)
    imgs = d.iid.unique()

    @torch.no_grad()
    def enc_i(paths, bs=64):
        out = []
        for i in range(0, len(paths), bs):
            px = proc(images=[Image.open(p).convert("RGB") for p in paths[i:i + bs]],
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

    U = enc_i([str(VG_ROOT / idx[i]) for i in imgs])
    pos = {i: k for k, i in enumerate(imgs)}
    U = U[[pos[i] for i in d.iid]]
    V = enc_t(d.caption.astype(str).tolist())
    np.savez_compressed(OUT / "crd_b16_VG_Action_test.npz", U=U, V=V)
    d.to_parquet(OUT / "crd_b16_VG_Action_test.meta.parquet", index=False)
    print(f"cached Action test rows {len(d)} images {len(imgs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
