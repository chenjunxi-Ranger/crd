"""MBPI-Net: margin-balanced pairwise interaction network.

Frozen backbone embeddings (cached) -> two projections -> bidirectional gating ->
interaction vector phi -> shared MLP -> pairing confidence q.  A = -q.

Negatives: within-batch derangement chosen to maximise
    S_{i,pi(i)} - lambda * |S_ii - S_{i,pi(i)}|
so positives and negatives are cosine-matched (no trivial shortcut).

Variants: full | concat | product  (ablation controls)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MBPI(nn.Module):
    def __init__(self, d_in=512, d=256, variant="full"):
        super().__init__()
        self.variant = variant
        self.pI = nn.Linear(d_in, d)
        self.pT = nn.Linear(d_in, d)
        self.lnI, self.lnT = nn.LayerNorm(d), nn.LayerNorm(d)
        self.U, self.V = nn.Linear(d, d), nn.Linear(d, d)
        head_in = {"full": 4 * d, "concat": 2 * d, "product": d}[variant]
        self.head = nn.Sequential(nn.Linear(head_in, 512), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(512, 1))

    def forward(self, u, v):
        a = self.lnI(self.pI(u))
        b = self.lnT(self.pT(v))
        if self.variant == "concat":
            phi = torch.cat([a, b], -1)
        elif self.variant == "product":
            phi = a * b
        else:
            git = a * torch.sigmoid(self.U(b))
            gti = b * torch.sigmoid(self.V(a))
            phi = torch.cat([git, gti, a * b, (a - b) ** 2], -1)
        return self.head(phi).squeeze(-1)


def load(bench, split):
    p = RESULTS_DIR / "gate" / f"emb_clip_{bench}_{split}.npz"
    z = np.load(p, allow_pickle=True)
    return z["img"], z["txt"], z["label"]


def derangement(S, lam):
    """Permutation with no fixed points, cosine-matched negatives."""
    n = S.shape[0]
    cost = S - lam * np.abs(np.diag(S)[:, None] - S)
    np.fill_diagonal(cost, -1e9)                 # forbid fixed points
    r, c = linear_sum_assignment(-cost)
    perm = c.copy()
    # repair any accidental fixed points (numerical edge cases)
    fixed = np.where(perm == np.arange(n))[0]
    for i in fixed:
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco")
    ap.add_argument("--variant", default="full", choices=["full", "concat", "product"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg", default="derange", choices=["derange", "real"])
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ui, vt, ytr = load(args.bench, "train")
    if args.neg == "derange":
        keep = np.where(ytr == 0)[0]
        ui, vt = ui[keep], vt[keep]
    print(f"train pairs {len(ui)} (neg={args.neg})", flush=True)

    uvi, vvt, yval = load(args.bench, "val")
    U = torch.tensor(ui, device=DEV); V = torch.tensor(vt, device=DEV)
    Uv = torch.tensor(uvi, device=DEV); Vv = torch.tensor(vvt, device=DEV)

    m = MBPI(ui.shape[1], 256, args.variant).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(U)
    clean_idx = torch.tensor(np.where(ytr == 0)[0], device=DEV)
    noisy_idx = torch.tensor(np.where(ytr == 1)[0], device=DEV)

    def cos_of(idx):
        with torch.no_grad():
            return (U[idx] * V[idx]).sum(1)

    for step in range(args.steps):
        if args.neg == "derange":
            idx = torch.randint(0, n, (args.batch,), device=DEV)
            u, v = U[idx], V[idx]
            with torch.no_grad():
                S = (u @ v.t()).cpu().numpy()
            perm = torch.tensor(derangement(S, args.lam), device=DEV)
            qpos = m(u, v)
            qneg = m(u, v[perm])
        else:
            # real noisy supervision: positives from clean rows, negatives from noisy rows,
            # paired by cosine quantile so the ranking loss cannot use similarity shortcuts
            ic = clean_idx[torch.randint(0, len(clean_idx), (args.batch,), device=DEV)]
            iN = noisy_idx[torch.randint(0, len(noisy_idx), (args.batch,), device=DEV)]
            cc, cn = cos_of(ic), cos_of(iN)
            oc = torch.argsort(cc); on = torch.argsort(cn)
            iN = iN[on][torch.argsort(oc)]        # order-statistics matching on cosine
            qpos, qneg = m(U[ic], V[ic]), m(U[iN], V[iN])
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([qpos, qneg]),
            torch.cat([torch.ones_like(qpos), torch.zeros_like(qneg)]))
        loss = loss + args.alpha * F.softplus(args.margin - qpos + qneg).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 300 == 0:
            print(f"  step {step+1} loss {loss.item():.4f}", flush=True)

    m.eval()
    with torch.no_grad():
        q = torch.cat([m(Uv[i:i + 512], Vv[i:i + 512]) for i in range(0, len(Uv), 512)])
        q = q.cpu().numpy()
    auc = float(roc_auc_score(yval, -q))
    cos = (uvi * vvt).sum(1)
    auc_cos = float(roc_auc_score(yval, -cos))
    auc_csls = None
    try:
        zb = np.load(RESULTS_DIR / "gate" / f"bank_{args.bench}_all.npz")
        k = 10
        sim_t = uvi @ zb["txt"].T
        sim_i = vvt @ zb["img"].T
        h = np.partition(sim_t, -k, axis=1)[:, -k:].mean(1) + \
            np.partition(sim_i, -k, axis=1)[:, -k:].mean(1)
        auc_csls = float(roc_auc_score(yval, -cos + 0.8 * h))
    except FileNotFoundError:
        pass

    out = {"bench": args.bench, "variant": args.variant, "neg": args.neg,
           "n_train": int(n), "auc_cos": auc_cos, "auc_csls": auc_csls,
           "auc_mbpi": auc, "delta_vs_cos": auc - auc_cos,
           "delta_vs_csls": None if auc_csls is None else auc - auc_csls}
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"mbpi_{args.variant}_{args.bench}.json", out)
    np.savez_compressed(RESULTS_DIR / "gate" / f"mbpi_scores_{args.variant}_{args.bench}.npz",
                        q=q, cos=cos, label=yval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
