"""Select the CSLS lambda on VALIDATION, then report TEST with that lambda locked."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH, jdump  # noqa: E402

VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
OUT = ROOT / "results" / "fgbench"
LAMS = np.round(np.arange(0.0, 1.21, 0.1), 2)


def topk_mean(a, b, k=10):
    out = np.empty(len(a))
    for i in range(0, len(a), 1024):
        s = a[i:i + 1024] @ b.T
        kk = min(k, s.shape[1])
        out[i:i + 1024] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--tag", default="b16")
    ap.add_argument("--bank", type=int, default=6000)
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    from _cache import resolve

    enc = JointEncoder(args.model_dir or CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device

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

    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()
    tr = pd.read_parquet(BENCH / "train.parquet")

    def load_split(split):
        d = pd.read_parquet(BENCH / f"{split}.parquet")
        out = []
        for ds in ("COCO", "VG"):
            s = d[d.dataset == ds].copy()
            s["k"] = (s.target_image_id.str.split(":").str[1].astype(str)
                      if ds == "VG" else s.target_image_id.astype(str))
            s = s[s.k.astype(int).isin(vgidx)] if ds == "VG" else s
            pmap = ({k: str(VG_ROOT / vgidx[int(k)]) for k in s.k.unique()}
                    if ds == "VG" else
                    {str(i): resolve(f) for i, f in zip(s.target_image_id, s.target_file)})
            cache = OUT / f"emb_{args.tag}_{ds}_{split}.npz"
            if cache.is_file():
                z = np.load(cache, allow_pickle=True)
                emb = {k: v for k, v in zip(z["ids"].astype(str), z["vec"])}
            else:
                ids = list(pmap)
                t0, vecs, emb = time.time(), [], {}
                for i in range(0, len(ids), 2000):
                    ch = ids[i:i + 2000]
                    v = enc_i([pmap[x] for x in ch])
                    vecs.append(v)
                    for x, e in zip(ch, v):
                        emb[x] = e
                    print(f"  {ds}/{split} enc {i+len(ch)}/{len(ids)} "
                          f"{time.time()-t0:.0f}s", flush=True)
                np.savez_compressed(cache, ids=np.array(ids), vec=np.concatenate(vecs))
            s = s[s.k.isin(emb)].reset_index(drop=True)
            Vi = np.stack([emb[k] for k in s.k])
            T = enc_t(s.caption.astype(str).tolist())
            bank = tr[tr.dataset == ds].drop_duplicates("target_image_id").head(args.bank)
            if ds == "VG":
                bid = bank.target_image_id.str.split(":").str[1].astype(int)
                keep = bid.isin(vgidx).to_numpy()
                bank, bid = bank[keep], bid[keep]
                bpaths = [str(VG_ROOT / vgidx[i]) for i in bid]
            else:
                bpaths = [resolve(f) for f in bank.target_file]
            cache_b = OUT / f"bank_{args.tag}_{ds}.npz"
            if cache_b.is_file():
                z = np.load(cache_b, allow_pickle=True)
                bi, bt = z["img"], z["txt"]
            else:
                bi, bt = enc_i(bpaths), enc_t(bank.caption.astype(str).tolist())
                np.savez_compressed(cache_b, img=bi, txt=bt)
            cos = (Vi * T).sum(1)
            ri, rc = topk_mean(Vi, bt), topk_mean(T, bi)
            out.append((ds, s, cos, ri, rc))
        return out

    val = load_split("val")
    test = load_split("test")

    locked = {}
    for ds, s, cos, ri, rc in val:
        for facet in sorted(s.facet.unique()):
            m = (s.facet == facet).to_numpy()
            y = s.pair_label.to_numpy()[m]
            if len(y) < 50 or len(np.unique(y)) < 2:
                print(f"  VAL {ds}/{facet}: n={len(y)} too small -> skip lambda fit",
                      flush=True)
                continue
            best = max(LAMS, key=lambda L: roc_auc_score(y, -cos[m] + L * (ri + rc)[m]))
            locked[(ds, facet)] = float(best)
            print(f"  VAL {ds}/{facet}: lambda={best}", flush=True)

    default = float(np.median([v for v in locked.values()])) if locked else 0.5
    print("fallback lambda (median of fitted):", default, flush=True)

    rows = []
    rng = np.random.default_rng(0)
    for ds, s, cos, ri, rc in test:
        for facet in sorted(s.facet.unique()):
            m = (s.facet == facet).to_numpy()
            y = s.pair_label.to_numpy()[m]
            if len(np.unique(y)) < 2:
                continue
            lam = locked.get((ds, facet), default)
            sm = (-cos + lam * (ri + rc))[m]
            uq = s.target_image_id.values[m]
            idx = np.where(m)[0]
            by = {c: idx[uq == c] for c in np.unique(uq)}
            cl = list(by)
            vals = []
            for _ in range(args.boot):
                pk = rng.choice(cl, len(cl), replace=True)
                ii = np.concatenate([by[c] for c in pk])
                yy = s.pair_label.to_numpy()[ii]
                if len(np.unique(yy)) < 2:
                    continue
                vals.append(roc_auc_score(yy, (-cos + lam * (ri + rc))[ii]))
            base = float(roc_auc_score(y, -cos[m]))
            rows.append({"dataset": ds, "facet": facet, "lambda_from_val": lam,
                         "n_noisy": int((y == 1).sum()),
                         "auc_cosine": base, "auc_csls_val_selected": float(roc_auc_score(y, sm)),
                         "ci_lo": float(np.percentile(vals, 2.5)) if vals else None,
                         "ci_hi": float(np.percentile(vals, 97.5)) if vals else None})
            print(f"{ds}/{facet} lam={lam:.1f} cos {rows[-1]['auc_cosine']:.4f} "
                  f"-> csls {rows[-1]['auc_csls_val_selected']:.4f}", flush=True)

    out = pd.DataFrame(rows)
    p = OUT / f"csls_val_selected_{args.tag}.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    jdump({"tag": args.tag, "locked": {f"{k[0]}|{k[1]}": v for k, v in locked.items()},
           "fallback": default, "rows": rows}, p.with_suffix(".json"))
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
