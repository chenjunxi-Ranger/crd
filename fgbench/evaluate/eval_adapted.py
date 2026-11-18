"""Evaluate an adapted (LoRA) encoder on FGBench v1: AUROC + 2AFC per facet.

Uses the SAME test cycles as the frozen-CLIP baseline, and asserts that the
adapter's training images do not overlap the test images.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH, jdump  # noqa: E402

VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
OUT = ROOT / "results" / "fgbench"


def topk_mean(a, b, k=10):
    out = np.empty(len(a))
    for i in range(0, len(a), 1024):
        s = a[i:i + 1024] @ b.T
        kk = min(k, s.shape[1])
        out[i:i + 1024] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tag", default="r200")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bank", type=int, default=6000)
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import lora_stage1 as L
    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR

    enc = JointEncoder(CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for tower in (model.vision_model, model.text_model):
        Ll = len(tower.encoder.layers)
        for li in range(Ll - 2, Ll):
            attn = tower.encoder.layers[li].self_attn
            dv = attn.q_proj.weight.device
            attn.q_proj = L.LoRALinear(attn.q_proj).to(dv)
            attn.v_proj = L.LoRALinear(attn.v_proj).to(dv)
            n += 2
    sd = torch.load(ROOT / args.adapter, map_location=dev, weights_only=True)
    missing = model.load_state_dict(sd, strict=False)
    print(f"injected {n} lora layers, loaded {len(sd)} tensors; "
          f"missing={len(missing.missing_keys)}", flush=True)
    model.eval()

    from PIL import Image

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

    te = pd.read_parquet(BENCH / f"{args.split}.parquet")
    tr = pd.read_parquet(BENCH / "train.parquet")
    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()
    from _cache import resolve

    results = []
    for ds in ("COCO", "VG"):
        d = te[te.dataset == ds].copy()
        if ds == "VG":
            d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
            d = d[d.iid.isin(vgidx)].reset_index(drop=True)
            paths = {str(i): str(VG_ROOT / vgidx[i]) for i in d.iid.unique()}
            key = d.iid.astype(str)
        else:
            paths = {}
            keys = []
            for i, f in zip(d.target_image_id, d.target_file):
                paths.setdefault(str(i), resolve(f))
                keys.append(str(i))
            d["iid"] = keys
            key = d.iid

        # ---- overlap check: adapter trained on COCO train images only ----
        train_keys = set(tr[tr.dataset == ds].target_image_id.astype(str))
        ov = len(set(key) & train_keys)
        print(f"{ds}: test rows {len(d)}, unique imgs {len(paths)}, "
              f"train-image overlap {ov}", flush=True)
        if ov:
            raise SystemExit(f"train/test image overlap in {ds}: {ov}")

        cache = OUT / f"emb_{args.tag}_{ds}_{args.split}.npz"
        if cache.is_file():
            z = np.load(cache, allow_pickle=True)
            emb = {k: v for k, v in zip(z["ids"].astype(str), z["vec"])}
        else:
            ids = list(paths)
            t0, vecs, emb = time.time(), [], {}
            for k in range(0, len(ids), 2000):
                chunk = ids[k:k + 2000]
                v = enc_i([paths[i] for i in chunk])
                vecs.append(v)
                for i, e in zip(chunk, v):
                    emb[i] = e
                print(f"  {ds} encoded {k+len(chunk)}/{len(ids)} {time.time()-t0:.0f}s",
                      flush=True)
            np.savez_compressed(cache, ids=np.array(ids), vec=np.concatenate(vecs))

        Vi = np.stack([emb[k] for k in key])
        T = enc_t(d.caption.astype(str).tolist())
        cos = (Vi * T).sum(1)

        bank = tr[(tr.dataset == ds)].drop_duplicates("target_image_id").head(args.bank)
        if ds == "VG":
            bid = bank.target_image_id.str.split(":").str[1].astype(int)
            keep = bid.isin(vgidx)
            bank = bank[keep]
            bpaths = [str(VG_ROOT / vgidx[i]) for i in bid[keep]]
        else:
            bpaths = [resolve(f) for f in bank.target_file]
        bi = enc_i(bpaths)
        bt = enc_t(bank.caption.astype(str).tolist())
        ri = topk_mean(Vi, bt)
        rc = topk_mean(T, bi)
        print(f"{ds} bank {bi.shape}", flush=True)

        rng = np.random.default_rng(0)
        for facet in sorted(d.facet.unique()):
            m = (d.facet == facet).to_numpy()
            y = d.pair_label.to_numpy()[m]
            if len(np.unique(y)) < 2:
                continue
            uq = d.target_image_id.values[m]
            for name, s in (("cosine", -cos), ("csls", -cos + 0.8 * (ri + rc))):
                sm = s[m]
                idx = np.where(m)[0]
                clusters = np.unique(uq)
                by = {c: idx[uq == c] for c in clusters}
                vals = []
                for _ in range(args.boot):
                    pk = rng.choice(clusters, len(clusters), replace=True)
                    ii = np.concatenate([by[c] for c in pk])
                    if len(np.unique(d.pair_label.to_numpy()[ii])) < 2:
                        continue
                    vals.append(roc_auc_score(d.pair_label.to_numpy()[ii], s[ii]))
                results.append({
                    "dataset": ds, "facet": facet, "detector": name,
                    "n_noisy": int((y == 1).sum()),
                    "auc": float(roc_auc_score(y, sm)),
                    "auprc": float(average_precision_score(y, sm)),
                    "ci_lo": float(np.percentile(vals, 2.5)) if vals else None,
                    "ci_hi": float(np.percentile(vals, 97.5)) if vals else None})

        # ---- 2AFC on the same cycles ----
        d["s"] = cos
        recs = []
        for cid, g in d.groupby("cycle_id"):
            if len(g) != 4:
                continue
            cl, no = g[g.pair_label == 0], g[g.pair_label == 1]
            if len(cl) != 2 or len(no) != 2:
                continue
            recs.append({"facet": g.facet.iat[0], "margin": cl.s.sum() - no.s.sum()})
        c = pd.DataFrame(recs)
        for facet in sorted(c.facet.unique()):
            sub = c[c.facet == facet]
            acc = float((sub.margin > 0).mean())
            vals = []
            arr = sub.margin.to_numpy()
            for _ in range(args.boot):
                pk = rng.integers(0, len(arr), len(arr))
                vals.append(float((arr[pk] > 0).mean()))
            results.append({"dataset": ds, "facet": facet, "detector": "2afc",
                            "n_noisy": int(len(sub)), "auc": acc, "auprc": None,
                            "ci_lo": float(np.percentile(vals, 2.5)),
                            "ci_hi": float(np.percentile(vals, 97.5))})
            print(f"  {ds}/{facet} 2AFC {acc:.4f}", flush=True)

    out = pd.DataFrame(results)
    p = OUT / f"adapted_{args.tag}_{args.split}.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    jdump({"adapter": args.adapter, "rows": results}, p.with_suffix(".json"))
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
