"""Two-stage test for a low-rank interaction term (identifiability-safe).

The joint fit (additive + bilinear trained together) degenerates: the bilinear
term absorbs the main effects, R^2(additive) goes negative and the decomposition
stops meaning anything. That is exactly the risk flagged when the module was
proposed, so this file uses the safe ordering instead:

    stage 1  frozen additive heads (existing checkpoints)  ->  q, residual r = z - q
    stage 2  fit a rank-r bilinear head h on r alone

Then two numbers decide whether the module is worth building:

    dR2_test   fraction of the residual variance that h explains
    dAUROC     change when h is subtracted from the residual score

Centring: both projections are mean-centred on the training split before the
product, so h cannot reproduce a constant; fitting on the residual keeps it from
competing with the additive part, which is already frozen.

Run:  python fgbench/experiments/crd_v2/residual_interaction.py --facets Attribute
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fgbench.models.crd_backbone import Off  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "crd_v2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = np.round(np.arange(0, 2.01, 0.1), 1)
FACETS = {"Entity": ("COCO", "Entity"), "Quantity": ("COCO", "Quantity"),
          "Attribute": ("VG", "Attribute"), "Relation": ("VG", "Relation")}


class LowRank(nn.Module):
    def __init__(self, d=512, r=32):
        super().__init__()
        self.PI = nn.Linear(d, r, bias=False)
        self.PT = nn.Linear(d, r, bias=False)
        self.r = r

    def forward(self, u, v):
        pI, pT = self.PI(u), self.PT(v)
        return (pI * pT).sum(1) / self.r ** 0.5


def auc(y, x):
    return float(roc_auc_score(y, -x))


def load(facet, name, mu, sd):
    ds, fc = FACETS[facet]
    fn = (sorted(CACHE.glob(f"crdbb_clip_{ds}_{fc}_c*_train.npz"))[0]
          if name == "train" else CACHE / f"crdbb_clip_{ds}_{fc}_{name}.npz")
    z = np.load(fn, allow_pickle=True)
    U, V = z["U"].astype(np.float32), z["V"].astype(np.float32)
    meta = pd.read_parquet(fn.with_suffix(".meta.parquet"))
    return U, V, ((U * V).sum(1) - mu) / sd, meta


def head_q(path, U, V):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Off(d=int(ck["dim"]), parts="both")
    m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad():
        return m(torch.tensor(U), torch.tensor(V)).numpy().ravel()


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--facets", default="Attribute,Relation")
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--rank", type=int, default=32)
    ap_.add_argument("--steps", type=int, default=800)
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

        for seed in (int(s) for s in a.seeds.split(",")):
            qtr = head_q(CACHE / f"crdbb_clip_{ds}_{fc}_both_row_s{seed}_head.pt",
                         Utr, Vtr)
            qva = head_q(CACHE / f"crdbb_clip_{ds}_{fc}_both_row_s{seed}_head.pt",
                         Uva, Vva)
            qte = head_q(CACHE / f"crdbb_clip_{ds}_{fc}_both_row_s{seed}_head.pt",
                         Ute, Vte)
            rtr, rva, rte = ztr - qtr, zva - qva, zte - qte

            torch.manual_seed(seed)
            m = LowRank(r=a.rank).to(DEV)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            Ut = torch.tensor(Utr, device=DEV); Vt = torch.tensor(Vtr, device=DEV)
            Rt = torch.tensor(rtr, device=DEV, dtype=torch.float32)
            rng = np.random.default_rng(seed)
            for _ in range(a.steps):
                idx = torch.tensor(rng.integers(0, len(Ut), 256), device=DEV)
                loss = F.mse_loss(m(Ut[idx], Vt[idx]), Rt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()

            with torch.no_grad():
                hva = m(torch.tensor(Uva, device=DEV),
                        torch.tensor(Vva, device=DEV)).cpu().numpy().ravel()
                hte = m(torch.tensor(Ute, device=DEV),
                        torch.tensor(Vte, device=DEV)).cpu().numpy().ravel()

            # how much of the residual does the interaction explain on test?
            d_r2 = 1.0 - float(((rte - hte) ** 2).sum()) / float((rte ** 2).sum())
            lam_h = float(GRID[int(np.argmax(
                [auc(y_va, rva - L * hva) for L in GRID]))])
            rows.append({
                "facet": facet, "seed": seed, "rank": a.rank,
                "resid_var_ratio": round(float(rte.var() / zte.var()), 4),
                "dR2_residual_explained": round(d_r2, 4),
                "std_h_over_std_resid": round(float(hte.std() / rte.std()), 4),
                "auroc_resid": round(auc(y_te, rte), 4),
                "auroc_resid_minus_h": round(auc(y_te, rte - lam_h * hte), 4),
                "lam_h": lam_h,
            })
            print(f"  {facet} s{seed} resid/var {rows[-1]['resid_var_ratio']:.3f} "
                  f"dR2 {d_r2:+.4f}  AUROC {rows[-1]['auroc_resid']:.4f} -> "
                  f"{rows[-1]['auroc_resid_minus_h']:.4f}", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "residual_interaction.csv", index=False, encoding="utf-8-sig")
    print("\n== mean over seeds ==")
    print(r.groupby("facet")[["resid_var_ratio", "dR2_residual_explained",
                              "auroc_resid", "auroc_resid_minus_h"]]
          .mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
