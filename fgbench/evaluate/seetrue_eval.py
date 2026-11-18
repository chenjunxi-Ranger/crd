"""SeeTRUE: single-pair alignment detection (AUROC / AUPRC).

label semantics are verified empirically (a correct model must give aligned
pairs a higher score); the script prints both orientations so the sign can be
checked instead of assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fgbench"
SC = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\SeeTRUE")
IMG = SC / "images"
WDIR = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Off(nn.Module):
    def __init__(self, d, width=64):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
        self.b = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u, v):
        return (self.a(u) + self.b(v) + self.b0).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="blipitm",
                    choices=["clip", "siglip2", "blipitc", "blipitm"])
    ap.add_argument("--head", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(SC / "SeeTRUE.csv")
    if args.limit:
        df = df.head(args.limit)
    df = df[df.image.map(lambda f: (IMG / f).is_file())].reset_index(drop=True)
    print(f"pairs {len(df)}  sources {df.dataset_source.nunique()}", flush=True)

    from PIL import Image
    files = sorted(df.image.unique())
    idx = {f: i for i, f in enumerate(files)}
    paths = [str(IMG / f) for f in files]
    ui = np.array([idx[f] for f in df.image])
    texts = df.text.astype(str).tolist()

    if args.model == "clip":
        from src.encoders import JointEncoder
        e = JointEncoder(str(WDIR / "clip-vit-base-patch16"))
        m, proc, dev = e.model, e.processor, e.device

        def enc_im(ps):
            o = []
            with torch.no_grad():
                for i in range(0, len(ps), 64):
                    px = proc(images=[Image.open(x).convert("RGB") for x in ps[i:i + 64]],
                              return_tensors="pt")["pixel_values"].to(dev)
                    v = m.vision_model(pixel_values=px)
                    o.append(torch.nn.functional.normalize(
                        m.visual_projection(v.pooler_output.float()), dim=-1).cpu().numpy())
            return np.concatenate(o)

        def enc_tx(ts):
            o = []
            with torch.no_grad():
                for i in range(0, len(ts), 128):
                    tk = proc(text=ts[i:i + 128], return_tensors="pt", padding=True,
                              truncation=True, max_length=77).to(dev)
                    v = m.text_model(**tk)
                    o.append(torch.nn.functional.normalize(
                        m.text_projection(v.pooler_output.float()), dim=-1).cpu().numpy())
            return np.concatenate(o)

        U, V = enc_im(paths), enc_tx(texts)
        Up = U[ui]
        s = (Up * V).sum(1)
    elif args.model == "siglip2":
        from transformers import AutoModel, AutoProcessor
        p = WDIR / "siglip2-base-patch16-224"
        m = AutoModel.from_pretrained(str(p)).to(DEV).eval()
        proc = AutoProcessor.from_pretrained(str(p))
        sc = m.logit_scale.exp().item()
        bias = m.logit_bias.item() if getattr(m, "logit_bias", None) is not None else 0.0

        def enc_im(ps):
            o = []
            with torch.no_grad():
                for i in range(0, len(ps), 64):
                    px = proc(images=[Image.open(x).convert("RGB") for x in ps[i:i + 64]],
                              return_tensors="pt")["pixel_values"].to(DEV)
                    o.append(torch.nn.functional.normalize(
                        m.get_image_features(pixel_values=px).float(), dim=-1).cpu().numpy())
            return np.concatenate(o)

        def enc_tx(ts):
            o = []
            with torch.no_grad():
                for i in range(0, len(ts), 128):
                    tk = proc(text=ts[i:i + 128], return_tensors="pt",
                              padding="max_length", truncation=True,
                              max_length=64).to(DEV)
                    o.append(torch.nn.functional.normalize(
                        m.get_text_features(input_ids=tk["input_ids"]).float(),
                        dim=-1).cpu().numpy())
            return np.concatenate(o)

        U, V = enc_im(paths), enc_tx(texts)
        Up = U[ui]
        s = sc * (Up * V).sum(1) + bias
    else:
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        p = WDIR / "blip-itm-base-coco"
        m = BlipForImageTextRetrieval.from_pretrained(str(p)).to(DEV).eval()
        proc = BlipProcessor.from_pretrained(str(p))

        def enc_im(ps):
            o = []
            with torch.no_grad():
                for i in range(0, len(ps), 48):
                    px = proc(images=[Image.open(x).convert("RGB") for x in ps[i:i + 48]],
                              return_tensors="pt")["pixel_values"].to(DEV)
                    h = m.vision_model(pixel_values=px, return_dict=True).last_hidden_state[:, 0, :]
                    if hasattr(m, "vision_proj"):
                        h = m.vision_proj(h)
                    o.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
            return np.concatenate(o)

        def enc_tx(ts):
            o = []
            with torch.no_grad():
                for i in range(0, len(ts), 64):
                    tk = proc(text=ts[i:i + 64], return_tensors="pt",
                              padding=True, truncation=True, max_length=64).to(DEV)
                    h = m.text_encoder(input_ids=tk["input_ids"],
                                       attention_mask=tk["attention_mask"],
                                       return_dict=True).last_hidden_state[:, 0, :]
                    if hasattr(m, "text_proj"):
                        h = m.text_proj(h)
                    o.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
            return np.concatenate(o)

        U, V = enc_im(paths), enc_tx(texts)
        Up = U[ui]
        if args.model == "blipitc":
            s = (Up * V).sum(1)
        else:
            cache = OUT / "seetrue_itm_scores.npz"
            if cache.is_file():
                z = np.load(cache, allow_pickle=True)
                s = z["s"]
                print("  loaded ITM score cache", flush=True)
            else:
                o, t0 = [], time.time()
                with torch.no_grad():
                    for i in range(0, len(df), 32):
                        px = proc(images=[Image.open(x).convert("RGB")
                                          for x in [str(IMG / f) for f in df.image[i:i + 32]]],
                                  return_tensors="pt")["pixel_values"].to(DEV)
                        tk = proc(text=texts[i:i + 32], return_tensors="pt",
                                  padding=True, truncation=True, max_length=64).to(DEV)
                        lg = m(pixel_values=px, input_ids=tk["input_ids"],
                               attention_mask=tk["attention_mask"],
                               use_itm_head=True).itm_score
                        o.append((lg[:, 1] - lg[:, 0]).float().cpu().numpy())
                        if (i // 32) % 40 == 0:
                            print(f"   {i}/{len(df)} {time.time()-t0:.0f}s", flush=True)
                s = np.concatenate(o)
                np.savez_compressed(cache, s=s, image=df.image.values)

    if args.head:
        ck = torch.load(args.head, map_location=DEV)
        head = Off(ck["dim"]).to(DEV)
        head.load_state_dict(ck["state"]); head.eval()
        lam, mu, sd = ck.get("lam", 1.0), ck["mu"], ck["sd"]
        with torch.no_grad():
            mh = head(torch.tensor(Up, device=DEV, dtype=torch.float32),
                      torch.tensor(V, device=DEV, dtype=torch.float32)).cpu().numpy()
        s = (s - mu) / sd - lam * mh

    y = df.label.to_numpy()
    res = {"model": args.model + ("+crd" if args.head else ""), "n": int(len(df)),
           "auroc_pos_is_aligned": round(float(roc_auc_score(y, s)), 4),
           "auroc_neg_is_aligned": round(float(roc_auc_score(y, -s)), 4),
           "auprc_pos_is_aligned": round(float(average_precision_score(y, s)), 4)}
    # per source
    for src, g in df.groupby("dataset_source"):
        m = g.index.to_numpy()
        if len(set(y[m])) < 2:
            continue
        res[f"auc_{src}"] = round(float(roc_auc_score(y[m], s[m])), 4)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    tag = args.model + ("_crd" if args.head else "")
    (OUT / f"seetrue_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
