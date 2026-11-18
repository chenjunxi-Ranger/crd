"""Full cross-depth correspondence matrix for CDUS (mechanism audit).

M[l, m] = mean over samples of cos(dz_I[l], dz_T[m])  for l, m in 1..L

Compares
  * diagonal             (true same-depth pairing)
  * off-diagonal mean
  * anti-diagonal        (the previous Drev control)
  * random permutation   (10 draws)
with a bootstrap CI over samples for (diagonal - off-diagonal).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco", choices=["coco", "foil"])
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device
    L = model.text_model.config.num_hidden_layers

    @torch.no_grad()
    def deltas(images, caps):
        px = proc(images=images, return_tensors="pt")["pixel_values"].to(dev)
        tx = proc(text=caps, return_tensors="pt", padding=True, truncation=True,
                  max_length=77)
        tx = {k: v.to(dev) for k, v in tx.items()}
        vo = model.vision_model(pixel_values=px, output_hidden_states=True)
        to = model.text_model(input_ids=tx["input_ids"],
                              attention_mask=tx["attention_mask"],
                              output_hidden_states=True)
        eos = tx["input_ids"].argmax(-1)
        zI, zT = [], []
        for l in range(L + 1):
            zi = F.normalize(model.visual_projection(
                model.vision_model.post_layernorm(vo.hidden_states[l][:, 0, :])).float(), -1)
            zt = F.normalize(model.text_projection(model.text_model.final_layer_norm(
                to.hidden_states[l][torch.arange(len(eos)), eos])).float(), -1)
            zI.append(zi)
            zT.append(zt)
        dI = torch.stack([zI[l] - zI[l - 1] for l in range(1, L + 1)], 1)   # (B,L,d)
        dT = torch.stack([zT[l] - zT[l - 1] for l in range(1, L + 1)], 1)
        dI = F.normalize(dI, dim=-1)
        dT = F.normalize(dT, dim=-1)
        return torch.einsum("bld,bmd->blm", dI, dT).cpu().numpy()           # (B,L,L)

    if args.bench == "foil":
        from src.protocol import load_foil_val
        df = load_foil_val(args.limit)
        rows = df["positive"].tolist()
    else:
        df = load_split(args.bench, "val", args.limit)
        rows = df["caption"].tolist()

    mats = []
    for i in range(0, len(df), args.batch):
        imgs = []
        for p in df["path"].iloc[i:i + args.batch]:
            from PIL import Image
            with Image.open(p) as im:
                im2 = im.convert("RGB")
                if im2.width * im2.height > 20_000_000:
                    im2.draft("RGB", (1024, 1024))
                imgs.append(im2.copy())
        mats.append(deltas(imgs, rows[i:i + args.batch]))
    M = np.concatenate(mats)                     # (N,L,L)
    mean = M.mean(0)
    diag = np.diagonal(mean)
    off = (mean.sum() - diag.sum()) / (L * L - L)
    anti = np.mean([mean[l, L - 1 - l] for l in range(L)])
    rng = np.random.default_rng(0)
    perm = []
    for _ in range(10):
        p = rng.permutation(L)
        perm.append(np.mean([mean[l, p[l]] for l in range(L)]))
    per_sample = np.diagonal(M, axis1=1, axis2=2).mean(1) - \
        (M.sum((1, 2)) - np.diagonal(M, axis1=1, axis2=2).sum(1)) / (L * L - L)
    boot = [per_sample[rng.integers(0, len(per_sample), len(per_sample))].mean()
            for _ in range(2000)]
    out = {"bench": args.bench, "n": int(len(M)), "layers": L,
           "diag_mean": float(diag.mean()), "offdiag_mean": float(off),
           "anti_diag": float(anti), "random_perm_mean": float(np.mean(perm)),
           "diag_minus_offdiag": float(diag.mean() - off),
           "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "matrix": mean.round(4).tolist()}
    write_json(RESULTS_DIR / "gate" / f"cdus_matrix_{args.bench}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "matrix"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
