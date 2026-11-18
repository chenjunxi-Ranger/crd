"""Training-free baselines on FGBench v1: cosine, CSLS(lambda grid), 2AFC."""
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
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH, jdump  # noqa: E402

VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
OUT = ROOT / "results" / "fgbench"
LAMS = (0.2, 0.4, 0.6, 0.8, 1.0)


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
    ap.add_argument("--tag", default="clip")
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
    te = pd.read_parquet(BENCH / "test.parquet")
    tr = pd.read_parquet(BENCH / "train.parquet")
    rng = np.random.default_rng(0)
    rows = []

    for ds in ("COCO", "VG"):
        d = te[te.dataset == ds].copy()
        if ds == "VG":
            d["k"] = d.target_image_id.str.split(":").str[1].astype(str)
            d = d[d.k.astype(int).isin(vgidx)].reset_index(drop=True)
            pmap = {k: str(VG_ROOT / vgidx[int(k)]) for k in d.k.unique()}
        else:
            d["k"] = d.target_image_id.astype(str)
            pmap = {}
            for i, f in zip(d.target_image_id, d.target_file):
                pmap.setdefault(str(i), resolve(f))

        cache = OUT / f"emb_{args.tag}_{ds}_test.npz"
        if cache.is_file():
            z = np.load(cache, allow_pickle=True)
            emb = {k: v for k, v in zip(z["ids"].astype(str), z["vec"])}
            miss = set(d.k) - set(emb)
            if miss:
                print(f"{ds}: cache missing {len(miss)}, re-encoding", flush=True)
                cache = None
        if not cache.is_file():
            ids = list(pmap)
            t0, vecs, emb = time.time(), [], {}
            for i in range(0, len(ids), 2000):
                ch = ids[i:i + 2000]
                v = enc_i([pmap[k] for k in ch])
                vecs.append(v)
                for k, e in zip(ch, v):
                    emb[k] = e
                print(f"  {ds} enc {i+len(ch)}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
            np.savez_compressed(cache, ids=np.array(ids), vec=np.concatenate(vecs))

        Vi = np.stack([emb[k] for k in d.k])
        T = enc_t(d.caption.astype(str).tolist())
        cos = (Vi * T).sum(1)

        bank = tr[tr.dataset == ds].drop_duplicates("target_image_id").head(args.bank)
        if ds == "VG":
            bid = bank.target_image_id.str.split(":").str[1].astype(int)
            keep = bid.isin(vgidx).to_numpy()
            bank = bank[keep]; bid = bid[keep]
            bpaths = [str(VG_ROOT / vgidx[i]) for i in bid]
        else:
            bpaths = [resolve(f) for f in bank.target_file]
        bi = enc_i(bpaths)
        bt = enc_t(bank.caption.astype(str).tolist())
        ri = topk_mean(Vi, bt)
        rc = topk_mean(T, bi)
        print(f"{ds} bank {bi.shape} test {Vi.shape}", flush=True)

        for facet in sorted(d.facet.unique()):
            m = (d.facet == facet).to_numpy()
            y = d.pair_label.to_numpy()[m]
            if len(np.unique(y)) < 2:
                continue
            uq = d.target_image_id.values[m]
            idx = np.where(m)[0]
            by = {c: idx[uq == c] for c in np.unique(uq)}
            clusters = list(by)

            def report(name, score, lam=None):
                sm = score[m]
                vals = []
                for _ in range(args.boot):
                    pk = rng.choice(clusters, len(clusters), replace=True)
                    ii = np.concatenate([by[c] for c in pk])
                    yy = d.pair_label.to_numpy()[ii]
                    if len(np.unique(yy)) < 2:
                        continue
                    vals.append(roc_auc_score(yy, score[ii]))
                accs = []
                # sm is an ANOMALY score (-cos): clean rows must sum LOWER
                sub = d[m].copy(); sub["s"] = sm
                for cid, g in sub.groupby("cycle_id"):
                    if len(g) != 4:
                        continue
                    accs.append(g[g.pair_label == 0].s.sum() < g[g.pair_label == 1].s.sum())
                rows.append({"dataset": ds, "facet": facet, "baseline": name,
                             "lam": lam, "n_noisy": int((y == 1).sum()),
                             "auc": float(roc_auc_score(y, sm)),
                             "auprc": float(average_precision_score(y, sm)),
                             "ci_lo": float(np.percentile(vals, 2.5)) if vals else None,
                             "ci_hi": float(np.percentile(vals, 97.5)) if vals else None,
                             "acc2afc": float(np.mean(accs)) if accs else None})
                print(f"{ds}/{facet}/{name}{'' if lam is None else f'@{lam}'} "
                      f"AUC {rows[-1]['auc']:.4f} 2AFC {rows[-1]['acc2afc']:.4f}", flush=True)

            report("cosine", -cos)
            for lam in LAMS:
                report("csls", -cos + lam * (ri + rc), lam)

    out = pd.DataFrame(rows)
    p = OUT / f"tf_{args.tag}.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    jdump({"tag": args.tag, "rows": len(rows)}, p.with_suffix(".json"))
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
