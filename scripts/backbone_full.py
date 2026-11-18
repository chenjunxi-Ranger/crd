"""Full-val FG-CLIP evaluation with image-clustered bootstrap CI.

Saves per-sample scores so later method comparisons reuse exactly the same rows.
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
N_BOOT = 1000


def cluster_ci(score, y, groups, seed=0):
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by[g] for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
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

    out = {"bench": args.bench, "backbone": "FG-CLIP-B/16", "split": "val_full"}
    if args.bench == "foil":
        df = load_foil_val(10000)
        vi = enc_images(df["path"].tolist())
        tp = enc_texts(df["positive"].astype(str).tolist())
        tn = enc_texts(df["negative"].astype(str).tolist())
        sp, sn = (vi * tp).sum(1), (vi * tn).sum(1)
        gid = df["image_id"].astype(str).to_numpy()
        ok = (sn < sp).astype(float)
        rng = np.random.default_rng(0)
        uniq = np.unique(gid)
        by = {g: np.where(gid == g)[0] for g in uniq}
        vals = []
        for _ in range(N_BOOT):
            pick = rng.choice(uniq, len(uniq), replace=True)
            vals.append(ok[np.concatenate([by[g] for g in pick])].mean())
        out.update({"n_pairs": int(len(df)), "cos_acc": float(ok.mean()),
                    "ci95": [float(np.percentile(vals, 2.5)),
                             float(np.percentile(vals, 97.5))]})
        np.savez_compressed(RESULTS_DIR / "gate" / "fg_foil_scores.npz",
                            s_pos=sp, s_neg=sn, image_id=gid)
    else:
        df = load_split(args.bench, "val")
        vi = enc_images(df["path"].tolist())
        vt = enc_texts(df["caption"].astype(str).tolist())
        s = (vi * vt).sum(1)
        y = df["label"].to_numpy()
        gid = df["image_id"].astype(str).to_numpy()
        auc = float(roc_auc_score(y, -s))
        lo, hi = cluster_ci(-s, y, gid)
        out.update({"n": int(len(df)), "auc_cos": auc, "ci95": [lo, hi]})
        np.savez_compressed(RESULTS_DIR / "gate" / f"fg_scores_{args.bench}.npz",
                            cos=s, label=y, image_id=gid)
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"fg_full_{args.bench}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
