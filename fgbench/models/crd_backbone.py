"""Backbone-agnostic CRD.

Each backbone only has to provide (u, v, s):
    u = f_I(I), v = f_T(T), s = S(I, T)
CRD then fits  m_hat(u,v) = a(u) + b(v) + b0  against the cycle mean of s.

Supported: clip (local ViT-B/16) and siglip2 (google/siglip2-base-patch16-224).
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
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH, jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
W = Path("/mnt/d/PyCharm/PythonProject/异常标签检测/LEMoN+CLIPsim/weights")
import hashlib

DATA_SEED = 0          # fixed: train subset + row order never depend on the run seed


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sha_arr(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def sha_text(values) -> str:
    """Hash a sequence of strings by content (object arrays' tobytes() is pointers)."""
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


B16 = W / "clip-vit-base-patch16"
B32 = W / "clip-vit-base-patch32"
L14 = W / "clip-vit-large-patch14"
SIG2 = W / "siglip2-base-patch16-224"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class ClipBB:
    name = "clip"

    def __init__(self, path=B16):
        from src.encoders import JointEncoder
        e = JointEncoder(str(path))
        self.m, self.proc, self.dev = e.model, e.processor, e.device

    @torch.no_grad()
    def images(self, paths, bs=64):
        from PIL import Image
        out = []
        for i in range(0, len(paths), bs):
            px = self.proc(images=[Image.open(p).convert("RGB") for p in paths[i:i + bs]],
                           return_tensors="pt")["pixel_values"].to(self.dev)
            o = self.m.vision_model(pixel_values=px)
            out.append(F.normalize(self.m.visual_projection(o.pooler_output.float()), dim=-1))
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def texts(self, caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            tk = self.proc(text=list(caps[i:i + bs]), return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(self.dev)
            o = self.m.text_model(**tk)
            out.append(F.normalize(self.m.text_projection(o.pooler_output.float()), dim=-1))
        return torch.cat(out).cpu().numpy()

    def score(self, U, V):
        return (U * V).sum(1)


class Siglip2BB:
    name = "siglip2"

    def __init__(self, path=SIG2):
        from transformers import AutoModel, AutoProcessor
        self.m = AutoModel.from_pretrained(str(path)).to(DEV).eval()
        self.proc = AutoProcessor.from_pretrained(str(path))
        self.dev = DEV
        sc = self.m.logit_scale.exp().item() if hasattr(self.m, "logit_scale") else 1.0
        bias = self.m.logit_bias.item() if getattr(self.m, "logit_bias", None) is not None else 0.0
        self.scale, self.bias = float(sc), float(bias)
        print(f"  siglip2 scale {self.scale:.3f} bias {self.bias:.3f}", flush=True)

    @torch.no_grad()
    def images(self, paths, bs=64):
        from PIL import Image
        out = []
        for i in range(0, len(paths), bs):
            px = self.proc(images=[Image.open(p).convert("RGB") for p in paths[i:i + bs]],
                           return_tensors="pt")["pixel_values"].to(self.dev)
            out.append(F.normalize(self.m.get_image_features(pixel_values=px).float(), dim=-1))
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def texts(self, caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            tk = self.proc(text=list(caps[i:i + bs]), return_tensors="pt",
                           padding="max_length", truncation=True,
                           max_length=64).to(self.dev)
            out.append(F.normalize(self.m.get_text_features(input_ids=tk["input_ids"]).float(),
                                   dim=-1))
        return torch.cat(out).cpu().numpy()

    def score(self, U, V):
        return self.scale * (U * V).sum(1) + self.bias


class FlavaBB:
    """FLAVA dual-encoder branch (image/text contrastive embeddings)."""
    name = "flava"

    def __init__(self, path=W / "flava-full"):
        from transformers import FlavaModel, FlavaProcessor
        self.m = FlavaModel.from_pretrained(str(path)).to(DEV).eval()
        self.proc = FlavaProcessor.from_pretrained(str(path))

    @torch.no_grad()
    def images(self, paths, bs=32):
        from PIL import Image
        out = []
        for i in range(0, len(paths), bs):
            px = self.proc(images=[Image.open(p).convert("RGB")
                                   for p in paths[i:i + bs]],
                           return_tensors="pt")["pixel_values"].to(DEV)
            h = self.m.get_image_features(pixel_values=px)
            if h.dim() == 3:                       # (B, N, D) -> CLS + projection
                h = self.m.image_projection(h[:, 0, :])
            out.append(F.normalize(h.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def texts(self, caps, bs=64):
        out = []
        for i in range(0, len(caps), bs):
            tk = self.proc(text=list(caps[i:i + bs]), return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to(DEV)
            h = self.m.get_text_features(input_ids=tk["input_ids"],
                                         attention_mask=tk["attention_mask"])
            if h.dim() == 3:                       # (B, L, D) -> CLS + projection
                h = self.m.text_projection(h[:, 0, :])
            out.append(F.normalize(h.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    def score(self, U, V):
        return (U * V).sum(1)


class Off(nn.Module):
    def __init__(self, d=512, parts="both", width=64):
        super().__init__()
        self.wa = parts in ("both", "image")
        self.wb = parts in ("both", "text")
        self.a = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1)) if self.wa else None
        self.b = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1)) if self.wb else None
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u, v):
        out = self.b0
        if self.wa:
            out = out + self.a(u)
        if self.wb:
            out = out + self.b(v)
        return out.squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="siglip2",
                    choices=["clip", "clipL", "siglip2", "flava"])
    ap.add_argument("--dataset", default="VG", choices=["VG", "COCO"])
    ap.add_argument("--facet", default="Attribute")
    ap.add_argument("--cycles", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--parts", default="both", choices=["both", "image", "text"])
    ap.add_argument("--lam-fixed", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="cycle", choices=["cycle", "row", "shuffled"],
                    help="supervision target: cycle mean / own score / placebo "
                         "(cycle means permuted across rows)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    factory = {"clip": lambda: ClipBB(), "clipL": lambda: ClipBB(L14),
               "siglip2": lambda: Siglip2BB(), "flava": lambda: FlavaBB()}
    bb = factory[args.backbone]()
    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()

    def load(split, limit_cycles=0):
        from _cache import resolve
        tag = f"{args.backbone}_{args.dataset}_{args.facet}"
        if limit_cycles:
            tag += f"_c{limit_cycles}"          # keep differently-sized subsets apart
        cache = OUT / f"crdbb_{tag}_{split}.npz"
        if cache.is_file():
            z = np.load(cache, allow_pickle=True)
            return pd.read_parquet(cache.with_suffix(".meta.parquet")), z["U"], z["V"], z["S"]
        d = pd.read_parquet(BENCH / f"{split}.parquet")
        d = d[(d.dataset == args.dataset) & (d.facet == args.facet)].copy()
        if args.dataset == "VG":
            d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
            d = d[d.iid.isin(vgidx)]
        else:
            d["iid"] = d.target_image_id.astype(str)
        d = d[d.groupby("cycle_id").target_image_id.transform("nunique") == 2]
        if limit_cycles and d.cycle_id.nunique() > limit_cycles:
            # DATA_SEED (not args.seed): the subset must be identical across seeds
            rng_data = np.random.default_rng(DATA_SEED)
            cids = d.cycle_id.unique(); rng_data.shuffle(cids)
            d = d[d.cycle_id.isin(cids[:limit_cycles])]
        d = d.reset_index(drop=True)
        imgs = d.iid.unique()
        if args.dataset == "VG":
            paths = [str(VG_ROOT / vgidx[i]) for i in imgs]
        else:
            pm = {}
            for i, f in zip(d.iid, d.target_file):
                pm.setdefault(i, resolve(f))
            paths = [pm[i] for i in imgs]
        t0 = time.time()
        U = bb.images(paths)
        pos = {i: k for k, i in enumerate(imgs)}
        U = U[[pos[i] for i in d.iid]]
        V = bb.texts(d.caption.astype(str).tolist())
        S = bb.score(U, V)
        print(f"  {split}: {len(d)} rows, {len(imgs)} imgs, {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(cache, U=U, V=V, S=S)
        d.to_parquet(cache.with_suffix(".meta.parquet"), index=False)
        return d, U, V, S

    tr, Utr, Vtr, _ = load("train", args.cycles)
    va, Uva, Vva, Sva = load("val")
    te, Ute, Vte, Ste = load("test")
    print(f"train {tr.cycle_id.nunique()} cycles | val {va.cycle_id.nunique()} | "
          f"test {te.cycle_id.nunique()}", flush=True)

    def cyc_mean(d, S):
        return pd.Series(S).groupby(d.cycle_id.values).transform("mean").to_numpy()

    Str = bb.score(Utr, Vtr)
    # standardise the native score with TRAIN statistics so CRD is scale-free
    mu, sd = float(Str.mean()), float(Str.std() + 1e-8)
    Str, Sva, Ste = (Str - mu) / sd, (Sva - mu) / sd, (Ste - mu) / sd
    print(f"  score standardised: train mean {mu:.3f} sd {sd:.3f}", flush=True)
    mtr, mva, mte = cyc_mean(tr, Str), cyc_mean(va, Sva), cyc_mean(te, Ste)
    if args.target == "row":
        mtr = Str                     # supervise on each pair's own standardised score
        print("  target = ROW", flush=True)
    elif args.target == "shuffled":
        rng_p = np.random.default_rng(DATA_SEED + 999)   # placebo, data-seed fixed
        mtr = mtr[rng_p.permutation(len(mtr))]
        print("  target = SHUFFLED cycle mean (placebo)", flush=True)

    head = Off(d=Utr.shape[1], parts=args.parts).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    Ut = torch.tensor(Utr, device=DEV, dtype=torch.float32)
    Vt = torch.tensor(Vtr, device=DEV, dtype=torch.float32)
    Mt = torch.tensor(mtr, device=DEV, dtype=torch.float32)
    n = len(Ut)
    for step in range(args.steps):
        idx = torch.tensor(rng.integers(0, n, args.batch), device=DEV)
        loss = F.huber_loss(head(Ut[idx], Vt[idx]), Mt[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        def pred(U, V):
            return head(torch.tensor(U, device=DEV, dtype=torch.float32),
                        torch.tensor(V, device=DEV, dtype=torch.float32)).cpu().numpy()
        mh_va, mh_te = pred(Uva, Vva), pred(Ute, Vte)

    def r2(d, m, mh):
        g = pd.DataFrame({"c": d.cycle_id.values, "m": m, "mh": mh}).groupby("c").agg(
            m=("m", "first"), mh=("mh", "mean"))
        return 1 - float(((g.m - g.mh) ** 2).sum()) / float(((g.m - g.m.mean()) ** 2).sum())

    yv, yt = va.pair_label.to_numpy(), te.pair_label.to_numpy()
    lam = args.lam_fixed if args.lam_fixed >= 0 else max(
        np.round(np.arange(0, 2.01, 0.1), 1),
        key=lambda L: roc_auc_score(yv, -(Sva - L * mh_va)))

    sc_crd = -(Ste - lam * mh_te)          # anomaly space
    sc_raw = -Ste
    auc_raw = float(roc_auc_score(yt, sc_raw))
    auc_crd = float(roc_auc_score(yt, sc_crd))

    def acc2afc(score, d):
        out = []
        for _, g in pd.DataFrame({"c": d.cycle_id.values, "y": yt,
                                  "s": score}).groupby("c"):
            if len(g) != 4:
                continue
            out.append(g[g.y == 0].s.sum() < g[g.y == 1].s.sum())
        return float(np.mean(out)) if out else float("nan")

    res = {"backbone": args.backbone, "dataset": args.dataset, "facet": args.facet,
           "target": args.target,
           "parts": args.parts, "seed": args.seed, "lam": float(lam),
           "r2_val": round(r2(va, mva, mh_va), 4), "r2_test": round(r2(te, mte, mh_te), 4),
           "auc_raw": round(auc_raw, 4), "auc_crd": round(auc_crd, 4),
           "auprc_raw": round(float(average_precision_score(yt, sc_raw)), 4),
           "auprc_crd": round(float(average_precision_score(yt, sc_crd)), 4),
           "acc2afc_raw": round(acc2afc(sc_raw, te), 6),
           "acc2afc_crd": round(acc2afc(sc_crd, te), 6),
           "delta_pp": round((auc_crd - auc_raw) * 100, 2)}
    # ---- provenance: every number must be traceable to data + code + seed ----
    cfg = f"{args.backbone}_{args.dataset}_{args.facet}_{args.parts}_{args.target}"
    prov = {
        "backbone": args.backbone, "dataset": args.dataset, "facet": args.facet,
        "parts": args.parts, "target": args.target, "seed": args.seed,
        "data_seed": DATA_SEED,
        "manifest_sha": {s: sha_file(BENCH / f"{s}.parquet")
                         for s in ("train", "val", "test")},
        "cache_sha": {"U": sha_arr(Utr), "V": sha_arr(Vtr),
                      "S_train": sha_arr(Str), "S_test": sha_arr(Ste)},
        "train_order_sha": sha_text(tr.cycle_id.astype(str).values),
        "n_train_rows": int(len(tr)), "n_val_rows": int(len(va)),
        "n_test_rows": int(len(te)),
        "mu": float(mu), "sd": float(sd), "lam": float(lam),
        "steps": args.steps, "batch": args.batch, "lr": args.lr,
    }
    jdump(prov, OUT / f"crdbb_prov_{cfg}_s{args.seed}.json")
    np.savez_compressed(OUT / f"crdbb_scores_{cfg}_s{args.seed}.npz",
                        cycle_id=te.cycle_id.values, y=yt, raw=Ste,
                        crd=Ste - lam * mh_te,
                        va_cycle_id=va.cycle_id.values, va_y=yv, va_raw=Sva,
                        va_crd=Sva - lam * mh_va, lam=np.array([lam]))
    torch.save({"state": head.state_dict(), "mu": mu, "sd": sd,
                "dim": int(Utr.shape[1]), "lam": float(lam)},
               OUT / f"crdbb_{cfg}_s{args.seed}_head.pt")
    jdump(res, OUT / f"crdbb_{cfg}_s{args.seed}.json")
    if args.parts == "both" and args.target == "row":
        # legacy alias so the main-table scripts keep resolving the official config
        jdump(res, OUT / f"crdbb_{args.backbone}_{args.dataset}_{args.facet}"
                         f"_s{args.seed}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
