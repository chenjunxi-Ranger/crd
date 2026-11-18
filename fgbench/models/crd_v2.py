"""CRD-v2: additive offset + joint conditional residual.

    b_hat(u,v) = a(u) + b(v) + h(u,v) + b0
    L = Huber(b_hat, m_c) + beta * Var_cycle(b_hat) + gamma * ||h||^2

v1 is the special case h=0.  Unlike v1, 2AFC is NOT invariant once h is active,
so 2AFC must be audited alongside AUROC.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class V2(nn.Module):
    def __init__(self, d=512, joint=True, width=64):
        super().__init__()
        self.joint = joint
        self.a = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
        self.b = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
        if joint:
            self.h = nn.Sequential(nn.Linear(3 * d, width), nn.GELU(),
                                   nn.Linear(width, width), nn.GELU(),
                                   nn.Linear(width, 1))
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u, v):
        out = self.a(u) + self.b(v) + self.b0
        if self.joint:
            out = out + self.h(torch.cat([u * v, (u - v).abs(), u * v * (u - v)], -1))
        return out.squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default="on", choices=["on", "off"])
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64, help="cycles per step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam-fixed", type=float, default=-1.0)
    ap.add_argument("--tag", default="crdv2")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def load(split):
        for pref in (f"crdx_b16_{split}", f"crd_b16_{split}"):
            p = OUT / f"{pref}.npz"
            if p.is_file():
                z = np.load(p, allow_pickle=True)
                d = pd.read_parquet(p.with_suffix(".meta.parquet"))
                return d, z["U"], z["V"]
        raise FileNotFoundError(f"no tensors for {split}")

    dtr, Utr, Vtr = load("train")
    dva, Uva, Vva = load("val")
    dte, Ute, Vte = load("test")

    def cyc_mean(d, U, V):
        s = (U * V).sum(1)
        return s, pd.Series(s).groupby(d.cycle_id.values).transform("mean").to_numpy()

    _, mtr = cyc_mean(dtr, Utr, Vtr)
    _, mva = cyc_mean(dva, Uva, Vva)
    cos_te, mte = cyc_mean(dte, Ute, Vte)

    # cycle index on train rows
    gids_tr = pd.Series(np.arange(len(dtr))).groupby(dtr.cycle_id.values).apply(
        lambda x: x.to_numpy()).to_dict()
    cyc_list = [v for v in gids_tr.values() if len(v) == 4]

    net = V2(joint=args.joint == "on").to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.gamma)
    Ut = torch.tensor(Utr, device=DEV); Vt = torch.tensor(Vtr, device=DEV)
    Mt = torch.tensor(mtr, dtype=torch.float32, device=DEV)

    for step in range(args.steps):
        pick = rng.choice(len(cyc_list), min(args.batch, len(cyc_list)), replace=False)
        rows = np.concatenate([cyc_list[k] for k in pick])
        r = torch.tensor(rows, device=DEV)
        bh = net(Ut[r], Vt[r])
        loss = F.huber_loss(bh, Mt[r])
        if args.beta > 0:
            b4 = bh.view(len(pick), 4)
            loss = loss + args.beta * b4.var(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1} loss {float(loss):.6f}", flush=True)

    with torch.no_grad():
        def pred(U, V):
            return net(torch.tensor(U, device=DEV), torch.tensor(V, device=DEV)).cpu().numpy()
        bh_va, bh_te = pred(Uva, Vva), pred(Ute, Vte)

    def r2(d, m, bh):
        g = pd.DataFrame({"c": d.cycle_id.values, "m": m, "b": bh}).groupby("c").agg(
            m=("m", "first"), b=("b", "mean"))
        return 1 - float(((g.m - g.b) ** 2).sum()) / float(((g.m - g.m.mean()) ** 2).sum())

    yv, yt = dva.pair_label.to_numpy(), dte.pair_label.to_numpy()
    lam = args.lam_fixed if args.lam_fixed >= 0 else max(
        np.round(np.arange(0, 2.01, 0.1), 1),
        key=lambda L: roc_auc_score(yv, -((Uva * Vva).sum(1) - L * bh_va)))

    sc = -(cos_te - lam * bh_te)
    auc = float(roc_auc_score(yt, sc))
    df = pd.DataFrame({"c": dte.cycle_id.values, "y": yt, "s": sc})
    acc = []
    for _, g in df.groupby("c"):
        if len(g) != 4:
            continue
        acc.append(g[g.y == 0].s.sum() < g[g.y == 1].s.sum())

    res = {"joint": args.joint, "beta": args.beta, "gamma": args.gamma,
           "seed": args.seed, "lam": float(lam),
           "r2_val": round(r2(dva, mva, bh_va), 4),
           "r2_test": round(r2(dte, mte, bh_te), 4),
           "cos_auc": round(float(roc_auc_score(yt, -cos_te)), 4),
           "auc": round(auc, 4),
           "auprc": round(float(average_precision_score(yt, sc)), 4),
           "acc2afc": round(float(np.mean(acc)), 6)}
    print(res, flush=True)
    np.savez_compressed(OUT / f"score_{args.tag}_{args.joint}_b{args.beta}_s{args.seed}.npz",
                        cycle_id=dte.cycle_id.values, y=yt, score=sc)
    jdump(res, OUT / f"{args.tag}_{args.joint}_b{args.beta}_s{args.seed}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
