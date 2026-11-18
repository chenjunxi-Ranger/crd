"""CRD v1: single-sample cycle-offset prediction.

Train:  m_c = mean of the 4 similarities in a cycle (frozen CLIP B/16)
        fit  m_hat(I,T) = a(u) + b(v) + b0   with Huber(m_hat, m_c)
Infer:  s_CRD = s_CLIP - lambda * m_hat(I,T)      (lambda chosen on VAL)
No partner / cycle id / bank at inference.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--tag", default="b16")
    ap.add_argument("--cycles", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="VG", choices=["VG", "COCO"])
    ap.add_argument("--facet", default="Attribute")
    ap.add_argument("--lam-fixed", type=float, default=-1.0,
                    help="if >=0, use this lambda for every seed (no val selection)")
    ap.add_argument("--parts", default="both", choices=["both", "image", "text"],
                    help="structural ablation: which bias term is active")
    ap.add_argument("--target", default="cycle", choices=["cycle", "row", "shuffled"],
                    help="supervision target: cycle mean (official) / row score / placebo")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    import torchvision  # noqa: F401
    from PIL import Image
    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR

    enc = JointEncoder(args.model_dir or CLIP_DIR)
    model, proc = enc.model, enc.processor
    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()

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

    def load(split, limit_cycles=0):
        from _cache import resolve
        d = pd.read_parquet(BENCH / f"{split}.parquet")
        d = d[(d.dataset == args.dataset) & (d.facet == args.facet)].copy()
        if args.dataset == "VG":
            d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
            d = d[d.iid.isin(vgidx)]
        else:
            d["iid"] = d.target_image_id.astype(str)
        if limit_cycles:
            cids = d.cycle_id.unique(); rng.shuffle(cids)
            d = d[d.cycle_id.isin(cids[:limit_cycles])]
        d = d[d.groupby("cycle_id").target_image_id.transform("nunique") == 2]
        sfx = f"{args.dataset}_{args.facet}"
        cache = OUT / f"crd_{args.tag}_{sfx}_{split}.npz"
        if cache.is_file():
            z = np.load(cache, allow_pickle=True)
            return pd.read_parquet(cache.with_suffix(".meta.parquet")), z["U"], z["V"]
        imgs = d.iid.unique()
        t0 = time.time()
        if args.dataset == "VG":
            U = enc_i([str(VG_ROOT / vgidx[i]) for i in imgs])
        else:
            pm = {}
            for i, f in zip(d.iid, d.target_file):
                pm.setdefault(i, resolve(f))
            U = enc_i([pm[i] for i in imgs])
        pos = {i: k for k, i in enumerate(imgs)}
        U = U[[pos[i] for i in d.iid]]
        print(f"  {split} imgs {len(imgs)} {time.time()-t0:.0f}s", flush=True)
        V = enc_t(d.caption.astype(str).tolist())
        np.savez_compressed(cache, U=U, V=V)
        d.reset_index(drop=True).to_parquet(cache.with_suffix(".meta.parquet"))
        return d.reset_index(drop=True), U, V

    tr, Utr, Vtr = load("train", args.cycles)
    va, Uva, Vva = load("val")
    te, Ute, Vte = load("test")
    print(f"train {len(tr)} cycles {tr.cycle_id.nunique()} | "
          f"val {len(va)} | test {len(te)}", flush=True)

    def cycle_means(d, U, V):
        s = (U * V).sum(1)
        m = pd.Series(s).groupby(d.cycle_id.values).transform("mean").to_numpy()
        return s, m

    _, mtr = cycle_means(tr, Utr, Vtr)
    _, mva = cycle_means(va, Uva, Vva)
    _, mte = cycle_means(te, Ute, Vte)

    # ---- ablation A3 / A4: what supervision signal does the head actually learn ----
    if args.target == "row":
        mtr = (Utr * Vtr).sum(1)                      # own score, no cycle aggregation
        print("  target = ROW (individual pair score)", flush=True)
    elif args.target == "shuffled":
        rng_t = np.random.default_rng(args.seed + 999)
        mtr = mtr[rng_t.permutation(len(mtr))]         # placebo: break the cycle link
        print("  target = SHUFFLED cycle mean (placebo)", flush=True)
    else:
        print("  target = CYCLE MEAN (official)", flush=True)

    class Off(nn.Module):
        def __init__(self, d=512, parts="both"):
            super().__init__()
            self.wa = parts in ("both", "image")
            self.wb = parts in ("both", "text")
            self.a = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1)) \
                if self.wa else None
            self.b = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1)) \
                if self.wb else None
            self.b0 = nn.Parameter(torch.zeros(1))

        def forward(self, u, v):
            out = self.b0
            if self.wa:
                out = out + self.a(u)
            if self.wb:
                out = out + self.b(v)
            return out.squeeze(-1)

    head = Off(parts=args.parts).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    Ut = torch.tensor(Utr, device=DEV); Vt = torch.tensor(Vtr, device=DEV)
    Mt = torch.tensor(mtr, device=DEV, dtype=torch.float32)
    n = len(Ut)
    for step in range(args.steps):
        idx = torch.tensor(rng.integers(0, n, args.batch), device=DEV)
        loss = F.huber_loss(head(Ut[idx], Vt[idx]), Mt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 300 == 0:
            print(f"  step {step+1} huber {float(loss):.6f}", flush=True)

    with torch.no_grad():
        def r2mae(d, U, V, m):
            mh = head(torch.tensor(U, device=DEV), torch.tensor(V, device=DEV)).cpu().numpy()
            fr = pd.DataFrame({"cycle_id": d.cycle_id.values, "m": m, "mh": mh})
            g = fr.groupby("cycle_id").agg(m=("m", "first"), mh=("mh", "mean"))
            ss_res = float(((g.m - g.mh) ** 2).sum())
            ss_tot = float(((g.m - g.m.mean()) ** 2).sum())
            r2 = 1 - ss_res / ss_tot
            print(f"    cycles {len(g)} true_sd {g.m.std():.4f} pred_sd {g.mh.std():.4f} "
                  f"corr {np.corrcoef(g.m, g.mh)[0,1]:.4f}", flush=True)
            return r2, float(np.abs(g.m - g.mh).mean()), mh

        r2v, maev, mh_va = r2mae(va, Uva, Vva, mva)
        r2t, maet, mh_te = r2mae(te, Ute, Vte, mte)
        print(f"VAL  R2 {r2v:.4f} MAE {maev:.4f}", flush=True)
        print(f"TEST R2 {r2t:.4f} MAE {maet:.4f}", flush=True)

        cos_va = (Uva * Vva).sum(1)
        cos_te = (Ute * Vte).sum(1)
        yv, yt = va.pair_label.to_numpy(), te.pair_label.to_numpy()

        def acc2afc(d, sc):
            accs = []
            for cid, g in d.assign(s=sc).groupby("cycle_id"):
                if len(g) != 4:
                    continue
                accs.append(g[g.pair_label == 0].s.sum() > g[g.pair_label == 1].s.sum())
            return float(np.mean(accs)) if accs else float("nan")

        if args.lam_fixed >= 0:
            lam = float(args.lam_fixed)
        else:
            best = None
            for lam in np.round(np.arange(0.0, 2.01, 0.1), 1):
                a = roc_auc_score(yv, -(cos_va - lam * mh_va))
                if best is None or a > best[1]:
                    best = (float(lam), float(a))
            lam = best[0]
        sc = -(cos_te - lam * mh_te)
        res = {"dataset": args.dataset, "facet": args.facet, "parts": args.parts,
           "lam_from_val": lam,
               "r2_val": round(r2v, 4), "mae_val": round(maev, 4),
               "r2_test": round(r2t, 4), "mae_test": round(maet, 4),
               "cos_auc": float(roc_auc_score(yt, -cos_te)),
               "crd_auc": float(roc_auc_score(yt, sc)),
               "cos_2afc": acc2afc(te, cos_te),
               "crd_2afc": acc2afc(te, cos_te - lam * mh_te)}
        print(res, flush=True)

    # keep per-row scores so a separate script can bootstrap across seeds
    np.savez_compressed(
        OUT / f"crd_scores_{args.tag}_{args.dataset}_{args.facet}_{args.parts}"
              f"_{args.target}_s{args.seed}.npz",
        cycle_id=te.cycle_id.values, y=te.pair_label.values,
        cos=cos_te, crd=cos_te - lam * mh_te, lam=np.array([lam]),
        va_cycle_id=va.cycle_id.values, va_y=va.pair_label.values,
        va_cos=cos_va, va_crd=cos_va - lam * mh_va)
    from sklearn.metrics import average_precision_score
    res["auprc_cos"] = round(float(average_precision_score(yt, -cos_te)), 4)
    res["auprc_crd"] = round(float(average_precision_score(yt, -(cos_te - lam * mh_te))), 4)

    torch.save(head.state_dict(),
               OUT / f"crd_{args.tag}_{args.dataset}_{args.facet}_head.pt")
    p = OUT / f"crd_{args.tag}_{args.dataset}_{args.facet}.json"
    jdump(res, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
