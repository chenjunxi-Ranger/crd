"""TRACED mechanism on FGBench v1 (training-free except a tiny LR head).

Per sample: greedily delete content words 3 times, each deletion scored by the
frozen CLIP image-text similarity; features = trajectory of scores + word overlap.
A LogisticRegression is fitted on the TRAIN split and evaluated on TEST.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH, jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
STOP = set("a an the is are was were be been of on in at to for with and or "
           "this that these those it its his her their there near by from as "
           "while very some two three one".split())
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def content_idx(words):
    return [i for i, w in enumerate(words)
            if w.lower().strip(".,") not in STOP and len(w.strip(".,")) > 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="VG")
    ap.add_argument("--facet", default="Attribute")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=12000)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--backbone", default="clip", choices=["clip", "blipitc"])
    args = ap.parse_args()

    from _cache import resolve
    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()

    if args.backbone == "blipitc":
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        BP = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights"
                  r"\blip-itm-base-coco")
        model = BlipForImageTextRetrieval.from_pretrained(str(BP)).to(DEV).eval()
        proc = BlipProcessor.from_pretrained(str(BP))

        @torch.no_grad()
        def enc_t(texts, bs=256):
            out = []
            for i in range(0, len(texts), bs):
                tk = proc(text=list(texts[i:i + bs]), return_tensors="pt", padding=True,
                          truncation=True, max_length=64).to(DEV)
                h = model.text_encoder(input_ids=tk["input_ids"],
                                       attention_mask=tk["attention_mask"],
                                       return_dict=True).last_hidden_state[:, 0, :]
                if hasattr(model, "text_proj"):
                    h = model.text_proj(h)
                out.append(F.normalize(h.float(), dim=-1).cpu().numpy())
            return np.concatenate(out)
    else:
        from src.encoders import JointEncoder
        from src.protocol import CLIP_DIR
        enc = JointEncoder(args.model_dir or CLIP_DIR)
        model, proc = enc.model, enc.processor

        @torch.no_grad()
        def enc_t(texts, bs=256):
            out = []
            for i in range(0, len(texts), bs):
                tk = proc(text=list(texts[i:i + bs]), return_tensors="pt", padding=True,
                          truncation=True, max_length=77).to(DEV)
                to = model.text_model(**tk)
                out.append(F.normalize(model.text_projection(to.pooler_output).float(),
                                       dim=-1).cpu().numpy())
            return np.concatenate(out)

    def load(split, limit):
        d = pd.read_parquet(BENCH / f"{split}.parquet")
        d = d[(d.dataset == args.dataset) & (d.facet == args.facet)].copy()
        if args.dataset == "VG":
            d["k"] = d.target_image_id.str.split(":").str[1].astype(str)
            d = d[d.k.astype(int).isin(vgidx)]
        else:
            d["k"] = d.target_image_id.astype(str)
        if limit:
            d = d.groupby("pair_label", group_keys=False).head(limit // 2)
        d = d.reset_index(drop=True)
        if args.backbone == "blipitc":
            import glob as _glob
            hits = sorted(_glob.glob(str(OUT / f"crdbb_blipitc_{args.dataset}_"
                                                 f"{args.facet}*_{split}.npz")))
            if not hits:
                raise FileNotFoundError(f"no blipitc cache for {split}")
            zc = np.load(hits[0], allow_pickle=True)
            d0 = pd.read_parquet(Path(hits[0]).with_suffix(".meta.parquet"))
            if split == "train":
                # the cache holds a TRAIN subset (e.g. 8k rows); use exactly those rows
                return d0.reset_index(drop=True), zc["U"]
            assert len(d0) == len(d), (len(d0), len(d))
            return d, zc["U"]
        z = np.load(OUT / f"emb_{args.tag}_{args.dataset}_{split}.npz", allow_pickle=True) \
            if (OUT / f"emb_{args.tag}_{args.dataset}_{split}.npz").is_file() else None
        if z is None:
            if args.dataset == "VG":
                paths = [str(VG_ROOT / vgidx[int(k)]) for k in d.k]
            else:
                paths = [resolve(f) for f in d.target_file]
            from PIL import Image
            U = []
            with torch.no_grad():
                for i in range(0, len(paths), 64):
                    px = proc(images=[Image.open(p).convert("RGB") for p in paths[i:i + 64]],
                              return_tensors="pt")["pixel_values"].to(DEV)
                    vo = model.vision_model(pixel_values=px)
                    U.append(F.normalize(model.visual_projection(vo.pooler_output).float(),
                                         dim=-1).cpu().numpy())
            U = np.concatenate(U)
        else:
            emb = {k: v for k, v in zip(z["ids"].astype(str), z["vec"])}
            U = np.stack([emb[k] for k in d.k])
        return d, U

    args.tag = "b16"
    dtr, Utr = load("train", args.train)
    dte, Ute = load("test", args.test)
    print(f"train {len(dtr)} test {len(dte)}", flush=True)

    def trajectory(d, U):
        words = [str(c).split() for c in d.caption]
        cur = [(w, 1.0) for w in words]           # placeholder, replaced below
        cur_text = [" ".join(w) for w in words]
        cur_score = None
        feats = []
        n = len(d)
        s0 = np.zeros(n); sf = np.zeros(n)
        cursc = np.zeros(n)                      # GLOBAL index, filled per chunk
        gains = np.zeros((n, args.steps))
        for i in range(0, n, args.chunk):
            sl = slice(i, min(i + args.chunk, n))
            idxs = list(range(sl.start, sl.stop))
            tex = [cur_text[j] for j in idxs]
            sc = (enc_t(tex) * U[idxs]).sum(1)
            s0[idxs] = sc
            cursc[idxs] = sc
            for step in range(args.steps):
                variants, owner, base = [], [], []
                for jj, j in enumerate(idxs):
                    w = cur_text[j].split()
                    for k in content_idx(w):
                        variants.append(" ".join(w[:k] + w[k + 1:]))
                        owner.append(jj); base.append(k)
                if not variants:
                    continue
                # score every deletion variant against its own image
                img_repeat = U[[idxs[o] for o in owner]]
                vsc = (enc_t(variants) * img_repeat).sum(1)
                best = {}
                for oi, owner_jj in enumerate(owner):
                    if owner_jj not in best or vsc[oi] > best[owner_jj][0]:
                        best[owner_jj] = (vsc[oi], variants[oi])
                for owner_jj, (bs, btext) in best.items():
                    j = idxs[owner_jj]
                    if bs > cursc[j]:
                        gains[j, step] = bs - cursc[j]
                        cursc[j] = bs
                        cur_text[j] = btext
                print(f"   chunk {i} step {step+1} done", flush=True) if step == 0 else None
            sf[idxs] = cursc[idxs]
        overlap = np.array([1 - len(set(cur_text[j].split())) /
                            max(1, len(set(str(d.caption.iat[j]).split())))
                            for j in range(n)])
        return pd.DataFrame({
            "s0": s0, "sf": sf, "gain": sf - s0,
            "g1": gains[:, 0], "g2": gains[:, 1] if args.steps > 1 else 0,
            "g3": gains[:, 2] if args.steps > 2 else 0,
            "n_gain": (gains > 0).sum(1), "overlap": overlap})

    t0 = time.time()
    ftr = trajectory(dtr, Utr)
    fte = trajectory(dte, Ute)
    print(f"trajectory done {time.time()-t0:.0f}s", flush=True)

    cols = ["s0", "sf", "gain", "g1", "g2", "g3", "n_gain", "overlap"]
    lr = LogisticRegression(max_iter=2000).fit(ftr[cols].to_numpy(),
                                               dtr.pair_label.to_numpy())
    p = lr.predict_proba(fte[cols].to_numpy())[:, 1]
    y = dte.pair_label.to_numpy()
    auc = float(roc_auc_score(y, p))
    cos_auc = float(roc_auc_score(y, -fte.s0.to_numpy()))

    # 2AFC on the same rows
    d2 = dte.assign(s=p)
    accs = []
    for cid, g in d2.groupby("cycle_id"):
        if len(g) != 4:
            continue
        accs.append(g[g.pair_label == 0].s.sum() < g[g.pair_label == 1].s.sum())
    res = {"dataset": args.dataset, "facet": args.facet,
           "n_train": int(len(dtr)), "n_test": int(len(dte)),
           "auc_cosine": round(cos_auc, 4), "auc_traced": round(auc, 4),
           "delta": round(auc - cos_auc, 4),
           "acc2afc_traced": round(float(np.mean(accs)), 4) if accs else None}
    print(res, flush=True)
    np.savez_compressed(OUT / f"score_traced_{args.backbone}_{args.dataset}_"
                              f"{args.facet}.npz",
                        cycle_id=dte.cycle_id.values, y=y, score=p)
    jdump(res, OUT / f"traced_{args.dataset}_{args.facet}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
