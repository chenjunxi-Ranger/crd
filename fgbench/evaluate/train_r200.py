"""Train the R200 recipe (LoRA + clean/noisy ranking + anchor) on a v0.2x train split."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fgbench.common import BENCH, PROJ   # noqa: E402

IMG_ROOT = Path(os.environ.get("COCO_IMG", str(PROJ / "数据集" / "MS-COCO" / "val2014")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=5.0)
    ap.add_argument("--out", default="results/fgbench/r200_v02c.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    import lora_stage1 as L
    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    enc = JointEncoder(CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device
    for p in model.parameters():
        p.requires_grad = False
    nv = len(model.vision_model.encoder.layers)
    nt = len(model.text_model.encoder.layers)
    L.BLOCKS = (min(nv, nt) - 2, min(nv, nt) - 1)
    L.inject(model)
    model.to(dev)

    tr = pd.read_parquet(BENCH / "train.parquet")
    cl = tr[tr.pair_label == 0].reset_index(drop=True)
    no = tr[tr.pair_label == 1].reset_index(drop=True)

    def prep(df):
        px = proc(images=[Image.open(IMG_ROOT / f).convert("RGB") for f in df.target_file],
                  return_tensors="pt")["pixel_values"].to(dev)
        tk = proc(text=df.caption.astype(str).tolist(), return_tensors="pt",
                  padding=True, truncation=True, max_length=77).to(dev)
        return px, tk

    def s_of(px, tk):
        vo = model.vision_model(pixel_values=px)
        to = model.text_model(**tk)
        v = F.normalize(model.visual_projection(vo.pooler_output).float(), dim=-1)
        t = F.normalize(model.text_projection(to.pooler_output).float(), dim=-1)
        return (v * t).sum(1), v, t

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    model.train()
    for step in range(args.steps):
        ic = rng.integers(0, len(cl), args.batch)
        iN = rng.integers(0, len(no), args.batch)
        px, tk = prep(cl.iloc[ic])
        with torch.no_grad():
            _, v0, t0 = s_of(px, tk)
        sp, v, t = s_of(px, tk)
        px2, tk2 = prep(no.iloc[iN])
        sn, _, _ = s_of(px2, tk2)
        loss = F.softplus(0.1 - sp.mean() + sn.mean())
        loss = loss + args.anchor * ((1 - (v * v0).sum(1)).mean()
                                     + (1 - (t * t0).sum(1)).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1} loss {float(loss):.4f}", flush=True)

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
          if k.endswith((".A", ".B")) and ("q_proj" in k or "v_proj" in k)}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out)
    print("saved", len(sd), "->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
