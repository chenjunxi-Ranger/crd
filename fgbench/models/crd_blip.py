"""BLIP (blip-itm-base-coco) with two scoring paths, each wired to CRD.

  --path itc : s = cos(u_I, v_T)                      (dual encoder)
  --path itm : s = logit_match - logit_mismatch       (cross-modal fusion)

CRD always uses SEPARATE image/text towers for (u, v) — never the fused ITM
feature — so the additive decomposition stays valid and 2-ANFC invariance holds.
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
WDIR = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\blip-itm-base-coco")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Blip:
    def __init__(self, path=WDIR):
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        self.m = BlipForImageTextRetrieval.from_pretrained(str(path)).to(DEV).eval()
        self.proc = BlipProcessor.from_pretrained(str(path))
        self.dev = DEV

    @torch.no_grad()
    def image_cls(self, paths, bs=48):
        from PIL import Image
        out = []
        for i in range(0, len(paths), bs):
            px = self.proc(images=[Image.open(p).convert("RGB")
                                   for p in paths[i:i + bs]],
                           return_tensors="pt")["pixel_values"].to(self.dev)
            h = self.m.vision_model(pixel_values=px, return_dict=True).last_hidden_state[:, 0, :]
            if hasattr(self.m, "vision_proj"):
                h = self.m.vision_proj(h)
            out.append(F.normalize(h.float(), dim=-1))
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def text_cls(self, caps, bs=64):
        out = []
        for i in range(0, len(caps), bs):
            tk = self.proc(text=list(caps[i:i + bs]), return_tensors="pt",
                           padding=True, truncation=True, max_length=64).to(self.dev)
            h = self.m.text_encoder(input_ids=tk["input_ids"],
                                    attention_mask=tk["attention_mask"],
                                    return_dict=True).last_hidden_state[:, 0, :]
            if hasattr(self.m, "text_proj"):
                h = self.m.text_proj(h)
            out.append(F.normalize(h.float(), dim=-1))
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def itm_score(self, paths, caps, bs=32):
        from PIL import Image
        out = []
        for i in range(0, len(paths), bs):
            px = self.proc(images=[Image.open(p).convert("RGB")
                                   for p in paths[i:i + bs]],
                           return_tensors="pt")["pixel_values"].to(self.dev)
            tk = self.proc(text=list(caps[i:i + bs]), return_tensors="pt",
                           padding=True, truncation=True, max_length=64).to(self.dev)
            o = self.m(pixel_values=px, input_ids=tk["input_ids"],
                       attention_mask=tk["attention_mask"], use_itm_head=True)
            lg = o.itm_score
            out.append((lg[:, 1] - lg[:, 0]).float().cpu())
        return torch.cat(out).numpy()


class Off(nn.Module):
    def __init__(self, d, parts="both", width=64):
        super().__init__()
        self.wa = parts in ("both", "image")
        self.wb = parts in ("both", "text")
        self.a = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1)) if self.wa else None
        self.b = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1)) if self.wb else None
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u, v):
        o = self.b0
        if self.wa:
            o = o + self.a(u)
        if self.wb:
            o = o + self.b(v)
        return o.squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="itm", choices=["itc", "itm"])
    ap.add_argument("--dataset", default="VG", choices=["VG", "COCO"])
    ap.add_argument("--facet", default="Attribute")
    ap.add_argument("--cycles", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-fixed", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="cycle", choices=["cycle", "row"],
                    help="supervision target: cycle mean or the pair's own score")
    ap.add_argument("--smoke", type=int, default=0, help="only report raw-score sanity")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    bb = Blip()
    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()

    def rows(split):
        d = pd.read_parquet(BENCH / f"{split}.parquet")
        d = d[(d.dataset == args.dataset) & (d.facet == args.facet)].copy()
        if args.dataset == "VG":
            d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
            d = d[d.iid.isin(vgidx)]
        else:
            d["iid"] = d.target_image_id.astype(str)
        d = d[d.groupby("cycle_id").target_image_id.transform("nunique") == 2]
        return d.reset_index(drop=True)

    def paths_of(d):
        if args.dataset == "VG":
            return [str(VG_ROOT / vgidx[i]) for i in d.iid]
        from _cache import resolve
        pm = {}
        for i, f in zip(d.iid, d.target_file):
            pm.setdefault(i, resolve(f))
        return [pm[i] for i in d.iid]

    def load(split, limit_cycles=0):
        tag = f"blip{args.path}_{args.dataset}_{args.facet}" + \
              (f"_c{limit_cycles}" if limit_cycles else "")
        cache = OUT / f"crdbb_{tag}_{split}.npz"
        if cache.is_file():
            z = np.load(cache, allow_pickle=True)
            return pd.read_parquet(cache.with_suffix(".meta.parquet")), z["U"], z["V"], z["S"]
        d = rows(split)
        if limit_cycles and d.cycle_id.nunique() > limit_cycles:
            cids = d.cycle_id.unique(); rng.shuffle(cids)
            d = d[d.cycle_id.isin(cids[:limit_cycles])].reset_index(drop=True)
        ps = paths_of(d)
        caps = d.caption.astype(str).tolist()
        t0 = time.time()
        # the ITC cache holds the SAME U/V tensors -> reuse them for the ITM path
        itc_name = (f"crdbb_blipitc_{args.dataset}_{args.facet}"
                    + (f"_c{limit_cycles}" if limit_cycles else "")
                    + f"_{split}.npz")
        itc = OUT / itc_name
        if args.path == "itm" and itc.is_file():
            z = np.load(itc, allow_pickle=True)
            U, V = z["U"], z["V"]
            print(f"  {split}: reuse ITC features {U.shape}", flush=True)
            S = bb.itm_score(ps, caps)
        else:
            U = bb.image_cls(ps)
            V = bb.text_cls(caps)
            S = (U * V).sum(1) if args.path == "itc" else bb.itm_score(ps, caps)
        print(f"  {split}: {len(d)} rows, U{U.shape} V{V.shape} {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(cache, U=U, V=V, S=S)
        d.to_parquet(cache.with_suffix(".meta.parquet"), index=False)
        return d, U, V, S

    if args.smoke:
        d = rows("test")
        d = d[d.cycle_id.isin(d.cycle_id.unique()[:args.smoke])].reset_index(drop=True)
        ps, caps = paths_of(d), d.caption.astype(str).tolist()
        U = bb.image_cls(ps); V = bb.text_cls(caps)
        s_itc = (U * V).sum(1)
        s_itm = bb.itm_score(ps, caps)
        print("U", U.shape, "V", V.shape, "itm", s_itm.shape)
        print("nan itc/itm", int(np.isnan(s_itc).sum()), int(np.isnan(s_itm).sum()))
        df = pd.DataFrame({"c": d.cycle_id.values, "y": d.pair_label.values,
                           "itc": s_itc, "itm": s_itm})
        for col in ("itc", "itm"):
            acc = []
            for _, g in df.groupby("c"):
                if len(g) != 4:
                    continue
                acc.append(g[g.y == 0][col].sum() > g[g.y == 1][col].sum())
            print(f"{col}: 2AFC {np.mean(acc):.4f}  mean_clean {df[df.y==0][col].mean():.4f} "
                  f"mean_noisy {df[df.y==1][col].mean():.4f}")
        return 0

    tr, Utr, Vtr, Str = load("train", args.cycles)
    va, Uva, Vva, Sva = load("val")
    te, Ute, Vte, Ste = load("test")
    print(f"train {tr.cycle_id.nunique()} | val {va.cycle_id.nunique()} | "
          f"test {te.cycle_id.nunique()}", flush=True)

    mu, sd = float(Str.mean()), float(Str.std() + 1e-8)
    Str, Sva, Ste = (Str - mu) / sd, (Sva - mu) / sd, (Ste - mu) / sd
    print(f"  standardised: mean {mu:.3f} sd {sd:.3f}", flush=True)

    def cm(d, S):
        return pd.Series(S).groupby(d.cycle_id.values).transform("mean").to_numpy()

    mtr, mva, mte = cm(tr, Str), cm(va, Sva), cm(te, Ste)
    if args.target == "row":
        mtr = Str                      # supervise on each pair's own standardised score
        print("  target = ROW", flush=True)
    head = Off(Utr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    Ut = torch.tensor(Utr, device=DEV); Vt = torch.tensor(Vtr, device=DEV)
    Mt = torch.tensor(mtr, device=DEV, dtype=torch.float32)
    for _ in range(args.steps):
        idx = torch.tensor(rng.integers(0, len(Ut), args.batch), device=DEV)
        loss = F.huber_loss(head(Ut[idx], Vt[idx]), Mt[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        mh_va = head(torch.tensor(Uva, device=DEV), torch.tensor(Vva, device=DEV)).cpu().numpy()
        mh_te = head(torch.tensor(Ute, device=DEV), torch.tensor(Vte, device=DEV)).cpu().numpy()

    def r2(d, m, mh):
        g = pd.DataFrame({"c": d.cycle_id.values, "m": m, "mh": mh}).groupby("c").agg(
            m=("m", "first"), mh=("mh", "mean"))
        return 1 - float(((g.m - g.mh) ** 2).sum()) / float(((g.m - g.m.mean()) ** 2).sum())

    yv, yt = va.pair_label.to_numpy(), te.pair_label.to_numpy()
    lam = args.lam_fixed if args.lam_fixed >= 0 else max(
        np.round(np.arange(0, 2.01, 0.1), 1),
        key=lambda L: roc_auc_score(yv, -(Sva - L * mh_va)))

    raw, crd = -Ste, -(Ste - lam * mh_te)

    def a2(score):
        out = []
        for _, g in pd.DataFrame({"c": te.cycle_id.values, "y": yt, "s": score}).groupby("c"):
            if len(g) != 4:
                continue
            out.append(g[g.y == 0].s.sum() < g[g.y == 1].s.sum())
        return float(np.mean(out)) if out else float("nan")

    res = {"backbone": f"blip-{args.path}", "dataset": args.dataset, "facet": args.facet,
           "seed": args.seed, "lam": float(lam),
           "r2_val": round(r2(va, mva, mh_va), 4), "r2_test": round(r2(te, mte, mh_te), 4),
           "auc_raw": round(float(roc_auc_score(yt, raw)), 4),
           "auc_crd": round(float(roc_auc_score(yt, crd)), 4),
           "auprc_raw": round(float(average_precision_score(yt, raw)), 4),
           "auprc_crd": round(float(average_precision_score(yt, crd)), 4),
           "acc2afc_raw": round(a2(raw), 6), "acc2afc_crd": round(a2(crd), 6),
           "delta_pp": round((roc_auc_score(yt, crd) - roc_auc_score(yt, raw)) * 100, 2)}
    print(res, flush=True)
    sfx = f"blip{args.path}_{args.dataset}_{args.facet}_{args.target}"
    np.savez_compressed(OUT / f"crd_scores_{sfx}_s{args.seed}.npz",
                        cycle_id=te.cycle_id.values, y=yt, raw=Ste,
                        crd=Ste - lam * mh_te)
    torch.save({"state": head.state_dict(), "mu": mu, "sd": sd,
                "dim": int(Utr.shape[1]), "lam": float(lam)},
               OUT / f"crdbb_{sfx}_s{args.seed}_head.pt")
    jdump(res, OUT / f"crdbb_{sfx}_s{args.seed}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
