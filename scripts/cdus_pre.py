"""CDUS zero-training diagnostic.

For each pair, take the per-layer CLS (vision) and EOS (text) states, push them
through the SAME readout used at the last layer (post_layernorm/final_layer_norm
+ projection), normalise, and measure the mean cosine between consecutive-layer
deltas over the last K layers:

    D(I,c) = mean_l cos(dz_I[l], dz_T[l]),  dz[l] = z[l] - z[l-1]

Controls
--------
* K in {4, 6, 12}            (uniform, no per-dataset choice)
* depth-misaligned pairing   (text layer l paired with image layer L-l)
* conditional MCA-like check (only pairs whose CLIP margin is small)

No label is used to build the score; labels only score the diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco",
                    choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device
    L = model.text_model.config.num_hidden_layers

    def prepare_image(im):
        if im.width * im.height > 20_000_000:
            im.draft("RGB", (1024, 1024))
        return proc(images=[im], return_tensors="pt")["pixel_values"].to(dev)

    @torch.no_grad()
    def run(paths, caps):
        """Return s (B,), D for K in {4,6,12}, and D with reversed depth pairing."""
        px = proc(images=paths, return_tensors="pt")["pixel_values"].to(dev)
        tx = proc(text=caps, return_tensors="pt", padding=True,
                  truncation=True, max_length=77)
        tx = {k: v.to(dev) for k, v in tx.items()}
        vo = model.vision_model(pixel_values=px, output_hidden_states=True)
        to = model.text_model(input_ids=tx["input_ids"],
                              attention_mask=tx["attention_mask"],
                              output_hidden_states=True)
        eos = tx["input_ids"].argmax(dim=-1)
        zI, zT = [], []
        for l in range(L + 1):
            hi = model.vision_model.post_layernorm(vo.hidden_states[l][:, 0, :])
            zi = model.visual_projection(hi)
            zi = F.normalize(zi.float(), dim=-1)
            ht = model.text_model.final_layer_norm(
                to.hidden_states[l][torch.arange(len(eos)), eos])
            zt = model.text_projection(ht)
            zt = F.normalize(zt.float(), dim=-1)
            zI.append(zi)
            zT.append(zt)
        s = (zI[-1] * zT[-1]).sum(1)
        dI = [zI[l] - zI[l - 1] for l in range(1, L + 1)]
        dT = [zT[l] - zT[l - 1] for l in range(1, L + 1)]
        out = {"s": s.cpu().numpy()}
        for K in (4, 6, 12):
            vals = []
            for l in range(L - K, L):
                a, b = dI[l], dT[l]
                vals.append(F.cosine_similarity(a, b, dim=-1))
            out[f"D{K}"] = torch.stack(vals).mean(0).cpu().numpy()
        # depth-misaligned control: text layer l paired with image layer L-1-l
        vals = []
        for l in range(L - 6, L):
            a, b = dI[l], dT[L - 1 - l]
            vals.append(F.cosine_similarity(a, b, dim=-1))
        out["Drev"] = torch.stack(vals).mean(0).cpu().numpy()
        return out

    def collect(df, cap_col):
        acc = {}
        paths = df["path"].tolist()
        caps = df[cap_col].tolist()
        for i in range(0, len(df), args.batch):
            batch = []
            for p, c in zip(paths[i:i + args.batch], caps[i:i + args.batch]):
                from PIL import Image
                with Image.open(p) as im:
                    batch.append(im.convert("RGB").copy())
            r = run(batch, caps[i:i + args.batch])
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
        return {k: np.concatenate(v) for k, v in acc.items()}

    payload = {"bench": args.bench, "limit": args.limit, "layers": L}

    if args.bench == "foil":
        from src.protocol import load_foil_val
        df = load_foil_val(args.limit)
        pos = collect(df, "positive")
        neg = collect(df, "negative")
        res = {"n_pairs": int(len(df))}
        for k in ("D4", "D6", "D12", "Drev"):
            d = pos[k] - neg[k]
            res[k] = {"acc": float(np.mean(np.where(d > 0, 1.0,
                                                    np.where(d < 0, 0.0, 0.5)))),
                      "mean_pos": float(pos[k].mean()), "mean_neg": float(neg[k].mean())}
        res["clip_acc"] = float(np.mean(np.where(neg["s"] < pos["s"], 1.0,
                                                 np.where(neg["s"] > pos["s"], 0.0, 0.5))))
        payload["foil"] = res
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        df = load_split(args.bench, "val", args.limit)
        f = collect(df, "caption")
        y = df["label"].to_numpy()
        res = {"n": int(len(df)),
               "auc_cos": float(roc_auc_score(y, -f["s"])),
               "by_D": {}}
        for k in ("D4", "D6", "D12", "Drev"):
            res["by_D"][k] = {"auc": float(roc_auc_score(y, -f[k])),
                              "mean_clean": float(f[k][y == 0].mean()),
                              "mean_noisy": float(f[k][y == 1].mean())}
        # conditional check on CLIP-near pairs
        clean, noisy = f["s"][y == 0], f["s"][y == 1]
        order = np.argsort(clean)
        cs = clean[order]
        pos_ = np.searchsorted(cs, noisy)
        left = np.clip(pos_ - 1, 0, len(cs) - 1)
        right = np.clip(pos_, 0, len(cs) - 1)
        choose = np.where(np.abs(cs[left] - noisy) <= np.abs(cs[right] - noisy), left, right)
        partner = order[choose]
        delta = np.abs(clean[partner] - noisy)
        cond = {}
        for eps in (0.005, 0.01):
            keep = delta < eps
            cond[str(eps)] = {}
            for k in ("D4", "D6", "D12", "Drev"):
                ec = f[k][y == 0][partner][keep]
                en = f[k][y == 1][keep]
                cond[str(eps)][k] = {"n": int(keep.sum()),
                                     "mca": float(np.mean(ec > en))}
        res["conditional"] = cond
        payload["row"] = res
        print(json.dumps(res, ensure_ascii=False, indent=2))

    write_json(RESULTS_DIR / "gate" / f"cdus_{args.bench}.json", payload)
    if args.bench == "foil":
        np.savez_compressed(RESULTS_DIR / "gate" / "cdus_scores_foil.npz",
                            s_pos=pos["s"], s_neg=neg["s"], d_pos=pos["D4"],
                            d_neg=neg["D4"], drev_pos=pos["Drev"], drev_neg=neg["Drev"])
    else:
        np.savez_compressed(RESULTS_DIR / "gate" / f"cdus_scores_{args.bench}.npz",
                            s=f["s"], d4=f["D4"], d6=f["D6"], d12=f["D12"],
                            drev=f["Drev"], label=y)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
