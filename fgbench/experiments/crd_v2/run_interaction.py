"""Priority 3 gate: is there a *useful* interaction term left after additive correction?

The condition for building the low-rank interaction module was "only if unimodal
correction shows a clear limitation". Rather than argue it, measure it.

Model (additive + low-rank bilinear):

    q(u,v) = b0 + a(u) + b(v) + (pI(u) - mI)'(pT(v) - mT) / sqrt(r)

The projections are mean-centred (running means over training batches, frozen at
eval) so the bilinear term cannot absorb a constant or a rank-1 additive effect;
that is the identifiability issue raised when this module was proposed.

Two questions, two numbers:

  Q1  does the interaction add predictable variance?   R^2(additive) vs R^2(full)
  Q2  is it useful in the final score?                 AUROC(z - lam*(qI+qT)) kept
                                                       vs z - lam*(qI+qT+h) subtracted

If Q1 shows little extra R^2 and Q2 shows no gain, the module is not worth
building, regardless of how it is regularised.

Run:  python fgbench/experiments/crd_v2/run_interaction.py --facets Attribute
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "crd_v2"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = np.round(np.arange(0, 2.01, 0.1), 1)
FACETS = {"Entity": ("COCO", "Entity"), "Quantity": ("COCO", "Quantity"),
          "Attribute": ("VG", "Attribute"), "Relation": ("VG", "Relation")}


class BilinearAdditive(nn.Module):
    """a(u) + b(v) + low-rank centred bilinear interaction."""

    def __init__(self, d=512, r=32, width=64):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(d, width), nn.GELU(),
                                   nn.Linear(width, 1))
        self.a, self.b = mk(), mk()
        self.PI = nn.Linear(d, r, bias=False)
        self.PT = nn.Linear(d, r, bias=False)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.register_buffer("mI", torch.zeros(r))
        self.register_buffer("mT", torch.zeros(r))
        self.r = r

    def forward_q(self, u, v):
        qI = self.a(u).squeeze(-1)
        qT = self.b(v).squeeze(-1)
        pI = self.PI(u)
        pT = self.PT(v)
        h = ((pI - self.mI) * (pT - self.mT)).sum(1) / self.r ** 0.5
        return qI, qT, h

    def forward(self, u, v):
        qI, qT, h = self.forward_q(u, v)
        return self.b0 + qI + qT + h

    @torch.no_grad()
    def update_means(self, u, v, mom=0.9):
        self.mI.mul_(mom).add_((1 - mom) * self.PI(u).mean(0))
        self.mT.mul_(mom).add_((1 - mom) * self.PT(v).mean(0))


def auc(y, x):
    return float(roc_auc_score(y, -x))


def ap(y, x):
    return float(average_precision_score(y, -x))


def r2(z, p):
    return 1.0 - float(((z - p) ** 2).sum()) / float(((z - z.mean()) ** 2).sum())


def load(facet, name, mu, sd):
    ds, fc = FACETS[facet]
    fn = (sorted(CACHE.glob(f"crdbb_clip_{ds}_{fc}_c*_train.npz"))[0]
          if name == "train" else CACHE / f"crdbb_clip_{ds}_{fc}_{name}.npz")
    z = np.load(fn, allow_pickle=True)
    U, V = z["U"].astype(np.float32), z["V"].astype(np.float32)
    meta = pd.read_parquet(fn.with_suffix(".meta.parquet"))
    return U, V, ((U * V).sum(1) - mu) / sd, meta


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--facets", default="Attribute,Relation")
    ap_.add_argument("--seeds", default="0,1")
    ap_.add_argument("--rank", type=int, default=32)
    ap_.add_argument("--steps", type=int, default=1500)
    a = ap_.parse_args()

    rows = []
    for facet in a.facets.split(","):
        ds, fc = FACETS[facet]
        prov = json.loads((CACHE / f"crdbb_prov_clip_{ds}_{fc}_both_row_s0.json")
                          .read_text(encoding="utf-8"))
        mu, sd = float(prov["mu"]), float(prov["sd"])
        Utr, Vtr, ztr, _ = load(facet, "train", mu, sd)
        Uva, Vva, zva, mva = load(facet, "val", mu, sd)
        Ute, Vte, zte, mte = load(facet, "test", mu, sd)
        y_va, y_te = mva.pair_label.to_numpy(), mte.pair_label.to_numpy()
        print(f"\n{facet}: {len(ztr)} train / {len(zte)} test", flush=True)

        for seed in (int(s) for s in a.seeds.split(",")):
            t0 = time.time()
            torch.manual_seed(seed)
            m = BilinearAdditive(r=a.rank).to(DEV)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            Ut = torch.tensor(Utr, device=DEV); Vt = torch.tensor(Vtr, device=DEV)
            Mt = torch.tensor(ztr, device=DEV, dtype=torch.float32)
            rng = np.random.default_rng(seed)
            for _ in range(a.steps):
                idx = torch.tensor(rng.integers(0, len(Ut), 256), device=DEV)
                ub, vb = Ut[idx], Vt[idx]
                m.update_means(ub, vb)
                loss = F.huber_loss(m(ub, vb), Mt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()

            with torch.no_grad():
                def parts(U, V):
                    u = torch.tensor(U, device=DEV); v = torch.tensor(V, device=DEV)
                    qI, qT, h = m.forward_q(u, v)
                    return (qI.cpu().numpy().ravel(), qT.cpu().numpy().ravel(),
                            h.cpu().numpy().ravel(), float(m.b0))
                qIv, qTv, hv, b0 = parts(Uva, Vva)
                qIt, qTt, ht, _ = parts(Ute, Vte)

            add_va = b0 + qIv + qTv
            full_va = add_va + hv
            add_te = b0 + qIt + qTt
            full_te = add_te + ht

            lam = float(GRID[int(np.argmax(
                [auc(y_va, zva - L * (qIv + qTv)) for L in GRID]))])
            s_keep = zte - lam * (qIt + qTt)          # interaction kept
            s_sub = zte - lam * (qIt + qTt + ht)      # interaction also removed

            rows.append({
                "facet": facet, "seed": seed, "rank": a.rank, "lam": lam,
                "r2_additive": round(r2(zte, add_te), 4),
                "r2_full": round(r2(zte, full_te), 4),
                "d_r2_interaction": round(r2(zte, full_te) - r2(zte, add_te), 4),
                "std_h_over_std_z": round(float(ht.std() / zte.std()), 4),
                "auroc_keep": round(auc(y_te, s_keep), 4),
                "auroc_sub": round(auc(y_te, s_sub), 4),
                "auprc_keep": round(ap(y_te, s_keep), 4),
            })
            print(f"  s{seed} R2add {rows[-1]['r2_additive']:.4f} "
                  f"+h {rows[-1]['d_r2_interaction']:+.4f} | "
                  f"AUROC keep {rows[-1]['auroc_keep']:.4f} "
                  f"sub {rows[-1]['auroc_sub']:.4f} | {time.time()-t0:.0f}s",
                  flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "interaction.csv", index=False, encoding="utf-8-sig")
    print("\n== mean over seeds ==")
    print(r.groupby("facet")[["r2_additive", "r2_full", "d_r2_interaction",
                              "std_h_over_std_z", "auroc_keep", "auroc_sub",
                              "auprc_keep"]].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
