"""CRD robustness checks: (1) random validation set, (2) backbone transfer (B/32).

Both keep the test split untouched for model/lambda selection.
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

OUT = ROOT / "results" / "fgbench"
VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B32 = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\clip-vit-base-patch32")


class Off(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.b = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u, v):
        return (self.a(u) + self.b(v) + self.b0).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="b16", choices=["b16", "b32"])
    ap.add_argument("--val-mode", default="official", choices=["official", "random"])
    ap.add_argument("--holdout", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tag = "b16" if args.backbone == "b16" else "clip"
    B16 = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\clip-vit-base-patch16")
    mdir = str(B16) if args.backbone == "b16" else str(B32)
    print(f"backbone={args.backbone} dir={mdir}", flush=True)
    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()

    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    from PIL import Image
    enc = JointEncoder(mdir or CLIP_DIR)
    model, proc = enc.model, enc.processor

    @torch.no_grad()
    def enc_i(paths, bs=64):
        out = []
        for i in range(0, len(paths), bs):
            px = proc(images=[Image.open(p).convert("RGB") for p in paths[i:i + bs]],
                      return_tensors="pt")["pixel_values"].to(DEV)
            vo = model.vision_model(pixel_values=px)
            out.append(F.normalize(model.visual_projection(vo.pooler_output).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def enc_t(caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            tk = proc(text=list(caps[i:i + bs]), return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(DEV)
            to = model.text_model(**tk)
            out.append(F.normalize(model.text_projection(to.pooler_output).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    def tensors(split):
        """(df, U, V) for VG Attribute; builds cache for this backbone if needed."""
        cache = OUT / f"crdx_{tag}_{split}.npz"
        meta = OUT / f"crdx_{tag}_{split}.meta.parquet"
        if cache.is_file():
            z = np.load(cache, allow_pickle=True)
            return pd.read_parquet(meta), z["U"], z["V"]
        d = pd.read_parquet(BENCH / f"{split}.parquet")
        d = d[(d.dataset == "VG") & (d.facet == "Attribute")].copy()
        d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
        d = d[d.iid.isin(vgidx)]
        d = d[d.groupby("cycle_id").target_image_id.transform("nunique") == 2].reset_index(drop=True)
        if split == "train":
            cids = d.cycle_id.unique(); rng.shuffle(cids)
            d = d[d.cycle_id.isin(cids[:8000])].reset_index(drop=True)
        imgs = d.iid.unique()
        U = enc_i([str(VG_ROOT / vgidx[i]) for i in imgs])
        pos = {i: k for k, i in enumerate(imgs)}
        U = U[[pos[i] for i in d.iid]]
        V = enc_t(d.caption.astype(str).tolist())
        np.savez_compressed(cache, U=U, V=V)
        d.to_parquet(meta, index=False)
        print(f"  built cache {split} {len(d)} rows", flush=True)
        return d, U, V

    # test cache: prefer the existing B/16 one (already built, identical rows)
    t0 = time.time()
    dte, Ute, Vte = tensors("test")
    print(f"test {len(dte)} rows ({time.time()-t0:.0f}s)", flush=True)

    if args.val_mode == "official":
        dtr, Utr, Vtr = tensors("train")
        dva, Uva, Vva = tensors("val")
    else:
        dtr, Utr, Vtr = tensors("train")
        cids = dtr.cycle_id.unique(); rng.shuffle(cids)
        hold = set(cids[: args.holdout])
        m = dtr.cycle_id.isin(hold).to_numpy()
        dva, Uva, Vva = dtr[m].reset_index(drop=True), Utr[m], Vtr[m]
        dtr, Utr, Vtr = dtr[~m].reset_index(drop=True), Utr[~m], Vtr[~m]
        print(f"random split: train {dtr.cycle_id.nunique()} cycles, "
              f"holdout {dva.cycle_id.nunique()} cycles", flush=True)

    def cyc_mean(d, U, V):
        s = (U * V).sum(1)
        return s, pd.Series(s).groupby(d.cycle_id.values).transform("mean").to_numpy()

    _, mtr = cyc_mean(dtr, Utr, Vtr)
    _, mva = cyc_mean(dva, Uva, Vva)
    cos_te, mte = cyc_mean(dte, Ute, Vte)

    head = Off().to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    Ut = torch.tensor(Utr, device=DEV); Vt = torch.tensor(Vtr, device=DEV)
    Mt = torch.tensor(mtr, dtype=torch.float32, device=DEV)
    n = len(Ut)
    for _ in range(args.steps):
        idx = torch.tensor(rng.integers(0, n, 256), device=DEV)
        loss = F.huber_loss(head(Ut[idx], Vt[idx]), Mt[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        mh_va = head(torch.tensor(Uva, device=DEV), torch.tensor(Vva, device=DEV)).cpu().numpy()
        mh_te = head(torch.tensor(Ute, device=DEV), torch.tensor(Vte, device=DEV)).cpu().numpy()

    def r2(d, m, mh):
        g = pd.DataFrame({"c": d.cycle_id.values, "m": m, "mh": mh}).groupby("c").agg(
            m=("m", "first"), mh=("mh", "mean"))
        return 1 - float(((g.m - g.mh) ** 2).sum()) / float(((g.m - g.m.mean()) ** 2).sum())

    yv = dva.pair_label.to_numpy(); yt = dte.pair_label.to_numpy()
    best = max(np.round(np.arange(0, 2.01, 0.1), 1),
               key=lambda L: roc_auc_score(yv, -( (Uva * Vva).sum(1) - L * mh_va)))
    res = {"backbone": args.backbone, "val_mode": args.val_mode,
           "lam": float(best), "r2_val": round(r2(dva, mva, mh_va), 4),
           "r2_test": round(r2(dte, mte, mh_te), 4),
           "cos_auc": round(float(roc_auc_score(yt, -cos_te)), 4),
           "crd_auc": round(float(roc_auc_score(yt, -(cos_te - best * mh_te))), 4)}
    accs = []
    df = pd.DataFrame({"c": dte.cycle_id.values, "y": yt,
                       "s": -(cos_te - best * mh_te)})
    for cid, g in df.groupby("c"):
        if len(g) != 4:
            continue
        accs.append(g[g.y == 0].s.sum() < g[g.y == 1].s.sum())
    res["acc2afc"] = round(float(np.mean(accs)), 6)
    print(res, flush=True)
    jdump(res, OUT / f"crd_validate_{args.backbone}_{args.val_mode}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
