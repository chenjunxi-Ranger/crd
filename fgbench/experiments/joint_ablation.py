"""Same-budget ablation: additive CRD head vs joint MLP on [u;v]."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from fgbench.models.crd_backbone import DEV, OUT, Off

FACETS = [("COCO", "Entity"), ("COCO", "Quantity"), ("VG", "Attribute"),
          ("VG", "Relation")]


class Joint(nn.Module):
    def __init__(self, d=512, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, width), nn.GELU(),
                                 nn.Linear(width, 1))

    def forward(self, u, v):
        return self.net(torch.cat([u, v], dim=-1)).squeeze(-1)


def params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train(head, U, V, target, seed, steps, batch):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Ut = torch.tensor(U, device=DEV, dtype=torch.float32)
    Vt = torch.tensor(V, device=DEV, dtype=torch.float32)
    Mt = torch.tensor(target, device=DEV, dtype=torch.float32)
    n = len(Ut)
    head.train()
    for _ in range(steps):
        idx = torch.tensor(rng.integers(0, n, batch), device=DEV)
        loss = F.huber_loss(head(Ut[idx], Vt[idx]), Mt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict(head, U, V):
    return head(torch.tensor(U, device=DEV, dtype=torch.float32),
                torch.tensor(V, device=DEV, dtype=torch.float32)).cpu().numpy()


def acc2afc(score, d):
    out = []
    for _, g in pd.DataFrame({"c": d.cycle_id.values, "y": d.pair_label.values,
                              "s": score}).groupby("c"):
        if len(g) != 4:
            continue
        out.append(g[g.y == 0].s.sum() < g[g.y == 1].s.sum())
    return float(np.mean(out)) if out else float("nan")


def cache_paths(ds, facet):
    train = sorted(OUT.glob(f"crdbb_clip_{ds}_{facet}_c*_train.npz"))
    if not train:
        return None
    tr = train[0]
    va = OUT / f"crdbb_clip_{ds}_{facet}_val.npz"
    te = OUT / f"crdbb_clip_{ds}_{facet}_test.npz"
    if not va.is_file() or not te.is_file():
        return None
    return tr, va, te


def load(p):
    z = np.load(p, allow_pickle=True)
    d = pd.read_parquet(p.with_suffix(".meta.parquet"))
    return d, z["U"], z["V"], z["S"]


def run(ds, facet, seeds, steps, batch):
    paths = cache_paths(ds, facet)
    if paths is None:
        print("SKIP", facet, "no cache", flush=True)
        return []
    tr, Utr, Vtr, Str = load(paths[0])
    va, Uva, Vva, Sva = load(paths[1])
    te, Ute, Vte, Ste = load(paths[2])
    mu, sd = float(Str.mean()), float(Str.std() + 1e-8)
    Str, Sva, Ste = (Str - mu) / sd, (Sva - mu) / sd, (Ste - mu) / sd
    mtr = Str                                    # row target, same as main CRD
    yv, yt = va.pair_label.to_numpy(), te.pair_label.to_numpy()
    rows = []
    for seed in range(seeds):
        for name, head in (("additive", Off(d=Utr.shape[1], parts="both")),
                           ("joint", Joint(d=Utr.shape[1], width=64))):
            head = head.to(DEV)
            npar = params(head)
            train(head, Utr, Vtr, mtr, seed, steps, batch)
            gva, gte = predict(head, Uva, Vva), predict(head, Ute, Vte)
            lam = max(np.round(np.arange(0, 2.01, 0.1), 1),
                      key=lambda L: roc_auc_score(yv, -(Sva - L * gva)))
            sc = -(Ste - lam * gte)
            rows.append({"dataset": ds, "facet": facet, "head": name,
                          "seed": seed, "params": npar, "lambda": float(lam),
                          "auroc": round(float(roc_auc_score(yt, sc)), 4),
                          "auprc": round(float(average_precision_score(yt, sc)), 4),
                          "acc2afc": round(acc2afc(sc, te), 4)})
            print(rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--facets", default="Entity,Quantity,Attribute,Relation")
    args = ap.parse_args()
    wanted = set(args.facets.split(","))
    rows = []
    for ds, facet in FACETS:
        if facet not in wanted:
            continue
        rows += run(ds, facet, args.seeds, args.steps, args.batch)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "joint_vs_additive.csv", index=False, encoding="utf-8-sig")
    print(df.groupby(["facet", "head"])[["auroc", "auprc", "acc2afc"]]
          .mean().round(4).to_string())
    print(df.groupby(["facet", "head"]).params.first().to_string())


if __name__ == "__main__":
    main()
