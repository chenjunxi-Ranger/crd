"""C2S-LI (M4): global + patch x token local interaction features.

Frozen CLIP. For each (I, T):
    S_ij = p_i . t_j / tau          (49 patches x L valid text tokens)
    a_j  = max_i S_ij               (best patch for token j)
    b_i  = max_j S_ij               (best token for patch i)
features:
    m_token  = mean_j a_j           (mean token alignment)
    q_token  = 20th pct of a_j      (weakest token support)
    c_patch  = mean_i [b_i > thr]   (patch coverage; thr from TRAIN only)
    d_align  = std_j a_j            (alignment dispersion)
A small MLP maps [s_global, m, q, c, d] -> cleanliness score.
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
TAU = 0.07


def build_features(split: str, enc, vgidx, resolve, limit_per_facet: int = 0) -> pd.DataFrame:
    from PIL import Image
    d = pd.read_parquet(BENCH / f"{split}.parquet")
    if limit_per_facet:
        d = (d.groupby(["dataset", "facet", "pair_label"], group_keys=False)
               .head(limit_per_facet).reset_index(drop=True))

    def ipath(ds, tid, tf):
        if ds == "VG":
            return str(VG_ROOT / vgidx[int(str(tid).split(":")[1])])
        return resolve(tf)

    d["k"] = [f"{ds}|{tid}" for ds, tid in zip(d.dataset, d.target_image_id)]
    img_path = {}
    for k, ds, tid, tf in zip(d.k, d.dataset, d.target_image_id, d.target_file):
        img_path.setdefault(k, ipath(ds, tid, tf))

    model, proc = enc.model, enc.processor
    eos = model.config.text_config.eos_token_id

    rows = []
    groups = d.groupby("k")
    keys = list(groups.groups)
    t0 = time.time()
    for bi in range(0, len(keys), 32):
        chunk = keys[bi:bi + 32]
        px = proc(images=[Image.open(img_path[k]).convert("RGB") for k in chunk],
                  return_tensors="pt")["pixel_values"].to(DEV)
        with torch.no_grad():
            vo = model.vision_model(pixel_values=px)
            patch = model.visual_projection(vo.last_hidden_state[:, 1:, :].float())
            patch = F.normalize(patch, dim=-1)                 # (B,49,D)
            pooled = F.normalize(model.visual_projection(vo.pooler_output.float()), dim=-1)

        texts, owners = [], []
        for k in chunk:
            for ridx, r in groups.get_group(k).iterrows():
                texts.append(str(r.caption)); owners.append((k, ridx))
        tk = proc(text=texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=77).to(DEV)
        with torch.no_grad():
            to = model.text_model(**tk)
            tok = model.text_projection(to.last_hidden_state.float())
            tok = F.normalize(tok, dim=-1)                     # (N,L,D)
            pooled_t = F.normalize(model.text_projection(to.pooler_output.float()), dim=-1)

        ids = tk["input_ids"]
        for n, (k, ridx) in enumerate(owners):
            j = chunk.index(k)
            pos = int((ids[n] == eos).nonzero()[0]) if (ids[n] == eos).any() else int(tk["attention_mask"][n].sum()) - 1
            tv = tok[n, 1:pos]                                 # skip SOT, drop EOT
            if tv.shape[0] == 0:
                tv = tok[n, 1:2]
            S = (patch[j] @ tv.T) / TAU                        # (49, Lt)
            aj = S.max(dim=0).values
            bi_ = S.max(dim=1).values
            rows.append({"row": ridx, "k": k,
                         "s_global": float((pooled[j] * pooled_t[n]).sum()),
                         "m_token": float(aj.mean()),
                         "q_token": float(torch.quantile(aj, 0.2)),
                         "d_align": float(aj.std()),
                         "b_mean": float(bi_.mean()),
                         "b_max": float(bi_.max())})
        if (bi // 32) % 20 == 0:
            print(f"  {split} {bi+len(chunk)}/{len(keys)} {time.time()-t0:.0f}s", flush=True)

    f = pd.DataFrame(rows).set_index("row").sort_index()
    f = f.reindex(d.index)
    f["pair_label"] = d.pair_label.values
    f["dataset"] = d.dataset.values
    f["facet"] = d.facet.values
    f["cycle_id"] = d.cycle_id.values
    f["target_image_id"] = d.target_image_id.values
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="m4", choices=["m4", "m5"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--train-per-facet", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    from _cache import resolve
    enc = JointEncoder(CLIP_DIR)
    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()

    cache_tr = OUT / "li_feat_train.parquet"
    cache_te = OUT / "li_feat_test.parquet"
    if cache_tr.is_file():
        ftr = pd.read_parquet(cache_tr)
        print("train features cached", ftr.shape, flush=True)
    else:
        ftr = build_features("train", enc, vgidx, resolve, args.train_per_facet)
        ftr.to_parquet(cache_tr)
        print("train features", ftr.shape, flush=True)
    if cache_te.is_file():
        fte = pd.read_parquet(cache_te)
        print("test features cached", fte.shape, flush=True)
    else:
        fte = build_features("test", enc, vgidx, resolve, 0)
        fte.to_parquet(cache_te)
        print("test features", fte.shape, flush=True)

    thr = float(np.quantile(ftr.b_mean.to_numpy(), 0.5))       # from TRAIN only
    for f in (ftr, fte):
        f["c_patch"] = (f.b_mean > thr).astype(float)

    cols = ["s_global", "m_token", "q_token", "c_patch", "d_align"]
    Xtr = torch.tensor(ftr[cols].to_numpy(np.float32), device=DEV)
    ytr = torch.tensor((1 - ftr.pair_label.to_numpy()).astype(np.float32), device=DEV)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(5, 32), nn.GELU(),
                                     nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(Xtr)
    for step in range(args.steps):
        idx = torch.tensor(rng.integers(0, n, 512), device=DEV)
        loss = F.binary_cross_entropy_with_logits(net(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    Xte = torch.tensor(fte[cols].to_numpy(np.float32), device=DEV)
    with torch.no_grad():
        s = net(Xte).cpu().numpy()
    fte["s"] = -s

    res = []
    for ds in ("COCO", "VG"):
        for facet in sorted(fte[fte.dataset == ds].facet.unique()):
            sub = fte[(fte.dataset == ds) & (fte.facet == facet)]
            y = sub.pair_label.to_numpy()
            if len(np.unique(y)) < 2:
                continue
            auc = float(roc_auc_score(y, sub.s))
            accs = []
            for cid, g in sub.groupby("cycle_id"):
                if len(g) != 4:
                    continue
                accs.append(g[g.pair_label == 0].s.sum() < g[g.pair_label == 1].s.sum())
            acc = float(np.mean(accs)) if accs else float("nan")
            res.append({"dataset": ds, "facet": facet, "variant": args.variant,
                        "n_noisy": int((y == 1).sum()), "auc": auc, "acc2afc": acc})
            print(f"{ds}/{facet} AUROC {auc:.4f} 2AFC {acc:.4f}", flush=True)

    out = pd.DataFrame(res)
    p = OUT / f"c2sli_{args.variant}.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    jdump({"variant": args.variant, "cols": cols, "thr": thr,
           "rows": int(len(fte)), "results": res}, p.with_suffix(".json"))
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
