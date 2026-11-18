"""Admission test: does FG-CLIP local evidence add information beyond global cosine?

Phrase extraction is deliberately simple (content words) so that the test measures
the representation, not a parser. q_local = -Q25(phrase support), support =
top-k mean similarity of a phrase against the 196 patch features.
"""
from __future__ import annotations

import argparse
import re
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
STOP = set("""a an the is are was were be being been of in on at to for with and or
by from as it its this that these those there here has have had do does did can could
will would may might must not no some any many much more most very just also while
when where who whom which what how his her their our your my he she they we you i
over under near next beside behind front top above below into out up down off""".split())


def phrases(cap: str, max_n: int = 6):
    toks = [w.lower() for w in re.findall(r"[A-Za-z']+", cap)]
    keep = [w for w in toks if w not in STOP and len(w) > 2]
    out = []
    for w in keep:
        if w not in out:
            out.append(w)
    return out[:max_n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(FG, trust_remote_code=True,
                                                 local_files_only=True).to(dev).eval()
    tok = CLIPTokenizerFast.from_pretrained(FG, local_files_only=True)
    iproc = CLIPImageProcessor.from_pretrained(FG, local_files_only=True)

    @torch.no_grad()
    def dense(paths, bs=32):
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
            d = model.get_image_dense_features(pixel_values=px)
            out.append(F.normalize(d.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def txt(caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            t = tok(list(caps[i:i + bs]), return_tensors="pt", padding=True,
                    truncation=True, max_length=77)
            t = {k: v.to(dev) for k, v in t.items()}
            out.append(F.normalize(model.get_text_features(**t).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    def support(D, q, k):
        sim = np.einsum("nd,pd->np", q, D)          # (n_phrases, 196)
        kk = min(k, sim.shape[1])
        part = np.partition(sim, -kk, axis=1)[:, -kk:]
        return part.mean(1)

    @torch.no_grad()
    def gimg(paths, bs=48):
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

    def run(paths, caps):
        D = dense(paths)
        g = txt(caps)                                 # global text
        # global image = mean-pooled patches (proxy) is not the model head; use model
        ph_all, owner = [], []
        for i, c in enumerate(caps):
            for w in phrases(c):
                ph_all.append(w); owner.append(i)
        P = txt(ph_all) if ph_all else np.zeros((0, D.shape[-1]))
        a = np.zeros(len(caps)); nph = np.zeros(len(caps))
        for i in range(len(caps)):
            idx = [k for k, o in enumerate(owner) if o == i]
            if not idx:
                a[i] = np.nan; continue
            s = support(D[i], P[idx], args.topk)
            a[i] = np.quantile(s, 0.25)
            nph[i] = len(idx)
        return g, a, nph

    out = {"bench": args.bench, "limit": args.limit}
    if args.bench == "foil":
        df = load_foil_val(args.limit)
        gp, ap, np_ = run(df["path"].tolist(), df["positive"].astype(str).tolist())
        gn, an, nn = run(df["path"].tolist(), df["negative"].astype(str).tolist())
        # global cosine needs the pooled image embedding; approximate with dense mean
        out.update({"mean_support_pos": float(np.nanmean(ap)),
                    "mean_support_neg": float(np.nanmean(an)),
                    "acc_local": float(np.nanmean(an < ap)),
                    "acc_both": float(np.nanmean((an < ap) & (np_ == nn))),
                    "n": int(len(df))})
    else:
        df = load_split(args.bench, "val", args.limit)
        paths = df["path"].tolist()
        caps = df["caption"].astype(str).tolist()
        _, a, nph = run(paths, caps)
        gi = gimg(paths)
        gt = txt(caps)
        cos = (gi * gt).sum(1)
        y = df["label"].to_numpy()
        keep = ~np.isnan(a)
        q = -a[keep]                                  # low support -> suspicious
        out.update({"n": int(keep.sum()),
                    "auc_local": float(roc_auc_score(y[keep], q)),
                    "auc_cos": float(roc_auc_score(y[keep], -cos[keep])),
                    "mean_support_clean": float(np.nanmean(a[y == 0])),
                    "mean_support_noisy": float(np.nanmean(a[y == 1])),
                    "mean_phrases": float(nph.mean())})
        np.savez_compressed(RESULTS_DIR / "gate" / f"local_{args.bench}.npz",
                            q=q, cos=cos[keep], label=y[keep],
                            image_id=df["image_id"].astype(str).to_numpy()[keep])
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"local_{args.bench}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
