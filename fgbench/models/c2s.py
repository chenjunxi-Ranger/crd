"""C2S: Cycle-to-Single scoring.

Frozen CLIP backbone; a light interaction head maps (u, v) -> s in R.
Two training objectives:
    L_global = BCEWithLogits(s, y)                  # cross-cycle comparability
    L_cycle  = softplus(-(s_AA + s_BB - s_AB - s_BA))# preserve within-cycle ranking
Inference uses a single (image, caption) pair only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH, jdump  # noqa: E402

VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
OUT = ROOT / "results" / "fgbench"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    def __init__(self, d=512, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d, h), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(h, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, u, v):
        z = torch.cat([u, v, u * v], dim=-1)
        return self.net(z).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cycle", choices=["global", "cycle"])
    ap.add_argument("--cycles", type=int, default=1500, help="cycles per facet")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    enc = JointEncoder(CLIP_DIR)
    from PIL import Image

    @torch.no_grad()
    def enc_i(paths, bs=64):
        out = []
        for i in range(0, len(paths), bs):
            px = enc.processor(images=[Image.open(p).convert("RGB") for p in paths[i:i + bs]],
                               return_tensors="pt")["pixel_values"].to(enc.device)
            vo = enc.model.vision_model(pixel_values=px)
            out.append(F.normalize(enc.model.visual_projection(vo.pooler_output).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def enc_t(caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            tk = enc.processor(text=list(caps[i:i + bs]), return_tensors="pt", padding=True,
                               truncation=True, max_length=77).to(enc.device)
            to = enc.model.text_model(**tk)
            out.append(F.normalize(enc.model.text_projection(to.pooler_output).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()
    from _cache import resolve

    tr = pd.read_parquet(BENCH / "train.parquet")
    # ---- sample balanced cycles per (dataset, facet) ----
    picked = []
    for (ds, facet), g in tr.groupby(["dataset", "facet"]):
        cids = g.cycle_id.unique()
        rng.shuffle(cids)
        picked.append(g[g.cycle_id.isin(cids[: args.cycles])])
    tr = pd.concat(picked, ignore_index=True)
    tr = tr[tr.groupby("cycle_id").target_image_id.transform("nunique") == 2]
    print("train rows", len(tr), "cycles", tr.cycle_id.nunique(),
          tr.groupby("facet").cycle_id.nunique().to_dict(), flush=True)

    def path_of(ds, tid, tf):
        if ds == "VG":
            i = int(str(tid).split(":")[1])
            return str(VG_ROOT / vgidx[i])
        return resolve(tf)

    keys = [f"{d}|{t}" for d, t in zip(tr.dataset, tr.target_image_id)]
    paths = {}
    for k, d, t, tf in zip(keys, tr.dataset, tr.target_image_id, tr.target_file):
        paths.setdefault(k, path_of(d, t, tf))

    cache = OUT / f"emb_c2s_train_{args.cycles}.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        emb = {k: v for k, v in zip(z["ids"].astype(str), z["vec"])}
        print("train cache hit", len(emb), flush=True)
    else:
        ids = list(paths)
        t0, vecs, emb = time.time(), [], {}
        for k in range(0, len(ids), 2000):
            chunk = ids[k:k + 2000]
            v = enc_i([paths[i] for i in chunk])
            vecs.append(v)
            for i, e in zip(chunk, v):
                emb[i] = e
            print(f"  encoded {k+len(chunk)}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(cache, ids=np.array(ids), vec=np.concatenate(vecs))

    U = np.stack([emb[k] for k in keys])
    T = enc_t(tr.caption.astype(str).tolist())
    print("train tensors", U.shape, T.shape, flush=True)

    # cycle index -> 4 row positions
    cyc = {}
    for cid, g in tr.groupby("cycle_id"):
        if len(g) != 4:
            continue
        cl = g.index[g.pair_label == 0].to_numpy()
        no = g.index[g.pair_label == 1].to_numpy()
        if len(cl) != 2 or len(no) != 2:
            continue
        cyc[cid] = (cl, no)
    cyc_items = list(cyc.items())
    print("usable cycles", len(cyc_items), flush=True)

    pos_of = {idx: i for i, idx in enumerate(tr.index)}
    Uc = torch.tensor(U, device=DEV); Tc = torch.tensor(T, device=DEV)
    # s is a CLEANLINESS score (higher = better match); pair_label 1 = noisy
    y = torch.tensor((1 - tr.pair_label).to_numpy().astype(np.float32), device=DEV)

    head = Head().to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    for step in range(args.steps):
        batch = [cyc_items[i] for i in rng.integers(0, len(cyc_items), args.batch)]
        rows = np.concatenate([np.concatenate([c[0], c[1]]) for _, c in batch])
        r = torch.tensor([pos_of[i] for i in rows], device=DEV)
        s = head(Uc[r], Tc[r])
        loss = F.binary_cross_entropy_with_logits(s, y[r])
        if args.variant == "cycle":
            q = s.view(len(batch), 4)          # [c1, c2, n1, n2]
            # rows are ordered: clean pair (A,cA),(B,cB) then noisy (B,cA),(A,cB)
            delta = q[:, 0] + q[:, 1] - q[:, 2] - q[:, 3]
            loss = loss + args.lam * F.softplus(-delta).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1} loss {float(loss):.4f}", flush=True)

    # ---------------- test ----------------
    te = pd.read_parquet(BENCH / "test.parquet")
    results = []
    for ds in ("COCO", "VG"):
        d = te[te.dataset == ds].copy()
        z = np.load(OUT / f"emb_clip_{ds}_test.npz", allow_pickle=True)
        emb_t = {k: v for k, v in zip(z["ids"].astype(str), z["vec"])}
        if ds == "VG":
            d["k"] = d.target_image_id.str.split(":").str[1].astype(str)
        else:
            d["k"] = d.target_image_id.astype(str)
        d = d[d.k.isin(emb_t)].reset_index(drop=True)
        Vi = np.stack([emb_t[k] for k in d.k])
        Ti = enc_t(d.caption.astype(str).tolist())
        with torch.no_grad():
            s = head(torch.tensor(Vi, device=DEV), torch.tensor(Ti, device=DEV)).cpu().numpy()
        d["s"] = -s                       # higher = more anomalous
        for facet in sorted(d.facet.unique()):
            m = (d.facet == facet).to_numpy()
            yy = d.pair_label.to_numpy()[m]
            if len(np.unique(yy)) < 2:
                continue
            auc = float(roc_auc_score(yy, d.s.to_numpy()[m]))
            sub = d[m]
            accs = []
            for cid, g in sub.groupby("cycle_id"):
                if len(g) != 4:
                    continue
                # d.s is an ANOMALY score: the clean rows must sum LOWER
                accs.append(g[g.pair_label == 0].s.sum() < g[g.pair_label == 1].s.sum())
            acc = float(np.mean(accs)) if accs else float("nan")
            results.append({"dataset": ds, "facet": facet, "variant": args.variant,
                            "n_noisy": int((yy == 1).sum()), "auc": auc,
                            "acc2afc": acc, "n_cycles": len(accs)})
            print(f"{ds}/{facet} AUROC {auc:.4f} 2AFC {acc:.4f}", flush=True)

    out = pd.DataFrame(results)
    tag = args.tag or args.variant
    p = OUT / f"c2s_{tag}.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    jdump({"variant": args.variant, "lam": args.lam, "steps": args.steps,
           "cycles_per_facet": args.cycles, "rows": results}, p.with_suffix(".json"))
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
