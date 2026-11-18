"""Backbone probe: does a fine-grained VL backbone raise the cosine ceiling?

Compares the frozen CLIP ViT-B/32 baseline with FG-CLIP (ViT-B/16, fine-grained
pretraining), cosine only, on the same rows. No training, no fusion, no bank.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, CLIPImageProcessor, CLIPTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, load_foil_val, load_split, write_json  # noqa: E402

FG = r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\fg-clip-base"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--limit", type=int, default=1500)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(FG, trust_remote_code=True,
                                                 local_files_only=True).to(dev).eval()
    tok = CLIPTokenizerFast.from_pretrained(FG, local_files_only=True)
    iproc = CLIPImageProcessor.from_pretrained(FG, local_files_only=True)

    @torch.no_grad()
    def enc_images(paths, bs=48):
        out = []
        for i in range(0, len(paths), bs):
            ims = []
            for p in paths[i:i + bs]:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    if im.width * im.height > 20_000_000:
                        im.draft("RGB", (1024, 1024))
                    ims.append(im.copy())
            px = iproc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
            v = model.get_image_features(pixel_values=px)
            out.append(F.normalize(v.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def enc_texts(caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            t = tok(list(caps[i:i + bs]), return_tensors="pt", padding=True,
                    truncation=True, max_length=77)
            t = {k: v.to(dev) for k, v in t.items()}
            e = model.get_text_features(**t)
            out.append(F.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    res = {"bench": args.bench}
    if args.bench == "foil":
        df = load_foil_val(args.limit)
        vi = enc_images(df["path"].tolist())
        tp = enc_texts(df["positive"].astype(str).tolist())
        tn = enc_texts(df["negative"].astype(str).tolist())
        sp, sn = (vi * tp).sum(1), (vi * tn).sum(1)
        res.update({"n": int(len(df)),
                    "cos_acc": float(np.mean(np.where(sn < sp, 1.0,
                                                      np.where(sn > sp, 0.0, 0.5)))),
                    "mean_pos": float(sp.mean()), "mean_neg": float(sn.mean())})
    else:
        df = load_split(args.bench, "val", args.limit)
        vi = enc_images(df["path"].tolist())
        vt = enc_texts(df["caption"].astype(str).tolist())
        s = (vi * vt).sum(1)
        y = df["label"].to_numpy()
        res.update({"n": int(len(df)), "auc_cos": float(roc_auc_score(y, -s)),
                    "mean_clean": float(s[y == 0].mean()),
                    "mean_noisy": float(s[y == 1].mean())})
    print(res, flush=True)
    write_json(RESULTS_DIR / "gate" / f"backbone_fgclip_{args.bench}.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
