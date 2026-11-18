"""FG-CLIP + bank-side CSLS: same formula, per-bench evaluation.

A = -cos + lam_I * r_I + lam_c * r_c
Evaluates the fixed lam=0.8 case plus a grid and the image-only variant, and
saves per-sample components so per-dataset weights can be fitted later (Exp B).
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
K = 10


def topk_mean(q, bank, k=K):
    bt = torch.as_tensor(bank, device="cuda" if torch.cuda.is_available() else "cpu")
    out = np.empty(len(q))
    for i in range(0, len(q), 256):
        sim = torch.as_tensor(q[i:i + 256], device=bt.device) @ bt.T
        out[i:i + 256] = torch.topk(sim, k, dim=1).values.mean(1).cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--bank", type=int, default=15000)
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
            out.append(F.normalize(model.get_image_features(pixel_values=px).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def enc_texts(caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            t = tok(list(caps[i:i + bs]), return_tensors="pt", padding=True,
                    truncation=True, max_length=77)
            t = {k: v.to(dev) for k, v in t.items()}
            out.append(F.normalize(model.get_text_features(**t).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    bp = RESULTS_DIR / "gate" / f"fgbank_{args.bench}.npz"
    if args.bench == "foil":
        val = load_foil_val(10000)
        vids = set(val["image_id"].astype(str))
        import pandas as pd
        from src.protocol import PROJECT
        tr = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                         encoding="utf-8-sig")
        tr = tr[~tr["image_id"].astype(str).isin(vids)]
        tr = tr.sample(min(args.bank, len(tr)), random_state=0)
        bank_paths = tr["path"].tolist()
        bank_caps = tr["positive_caption"].astype(str).tolist()
    else:
        tr = load_split(args.bench, "train")
        tr = tr.sample(min(args.bank, len(tr)), random_state=0)
        bank_paths = tr["path"].tolist()
        bank_caps = tr["caption"].astype(str).tolist()

    if bp.is_file():
        z = np.load(bp)
        bi, bt = z["img"], z["txt"]
    else:
        bi = enc_images(bank_paths)
        bt = enc_texts(bank_caps)
        np.savez_compressed(bp, img=bi, txt=bt)
    print(f"{args.bench}: bank {len(bi)} rows", flush=True)

    out = {"bench": args.bench, "bank": int(len(bi))}
    if args.bench == "foil":
        df = load_foil_val(10000)
        vi = enc_images(df["path"].tolist())
        tp = enc_texts(df["positive"].astype(str).tolist())
        tn = enc_texts(df["negative"].astype(str).tolist())
        sp, sn = (vi * tp).sum(1), (vi * tn).sum(1)
        ri = topk_mean(vi, bt)
        rcp, rcn = topk_mean(tp, bi), topk_mean(tn, bi)
        gid = df["image_id"].astype(str).to_numpy()
        np.savez_compressed(RESULTS_DIR / "gate" / "fg_csls_foil.npz",
                            s_pos=sp, s_neg=sn, ri=ri, rc_pos=rcp, rc_neg=rcn,
                            image_id=gid)
        res = {"cos": float(np.mean(sn < sp))}
        for li, lc, tag in ((0.8, 0.8, "full0.8"), (0.8, 0.0, "img_only"),
                            (0.8, -0.1, "neg_rc")):
            ap_ = -sp + li * ri + lc * rcp
            an_ = -sn + li * ri + lc * rcn
            res[tag] = float(np.mean(an_ > ap_))
        out.update(res)
    else:
        df = load_split(args.bench, "val")
        vi = enc_images(df["path"].tolist())
        vt = enc_texts(df["caption"].astype(str).tolist())
        cos = (vi * vt).sum(1)
        ri = topk_mean(vi, bt)
        rc = topk_mean(vt, bi)
        y = df["label"].to_numpy()
        np.savez_compressed(RESULTS_DIR / "gate" / f"fg_csls_{args.bench}.npz",
                            cos=cos, ri=ri, rc=rc, label=y,
                            image_id=df["image_id"].astype(str).to_numpy())
        res = {"cos": float(roc_auc_score(y, -cos))}
        for li, lc, tag in ((0.8, 0.8, "full0.8"), (0.8, 0.0, "img_only"),
                            (0.0, 0.8, "txt_only")):
            res[tag] = float(roc_auc_score(y, -cos + li * ri + lc * rc))
        out.update(res)
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"fg_csls_{args.bench}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
