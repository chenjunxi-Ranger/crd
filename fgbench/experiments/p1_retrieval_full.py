"""Full P1 retrieval: 113k pool, contrastive projection, bootstrap CI."""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_hard import (abtt_scores, csls_scores, crd_scores, hard_donor,
                     random_scores)

from fgbench.models.crd_backbone import ClipBB

COCO = Path("/mnt/d/PyCharm/PythonProject/异常标签检测/数据集/MS-COCO")
KARPATHY = Path("/mnt/d/PyCharm/PythonProject/异常标签检测/数据集/karpathy_split/dataset_coco.json")
OUT = Path("results/fgbench/p1_retrieval")
CACHE = OUT / "full_features.npz"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def load_pool():
    d = json.load(open(KARPATHY, encoding="utf-8"))["images"]
    pool, ev = [], []
    for x in d:
        cap = x["sentences"][0]["raw"].strip()
        if x["split"] in ("train", "restval"):
            pool.append((x["filename"], cap))
        elif x["split"] == "val":
            ev.append((x["filename"], cap))
    return pool, ev


def ipath(name):
    sub = "train2014" if name.startswith("COCO_train2014") else "val2014"
    return COCO / sub / name


def encode(paths, caps, bs_img=64, bs_txt=256):
    bb = ClipBB()
    U, V = [], []
    for i in range(0, len(paths), 5000):
        U.append(bb.images(paths[i:i + 5000], bs=bs_img))
        log("images", i, "/", len(paths))
    for i in range(0, len(caps), 5000):
        V.append(bb.texts(caps[i:i + 5000], bs=bs_txt))
        log("texts", i, "/", len(caps))
    return np.concatenate(U), np.concatenate(V)


def get_features(pool, ev, cache):
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        if len(z["U"]) == len(pool) and len(z["Ue"]) == len(ev):
            log("loaded", cache)
            return z["U"], z["V"], z["Ue"], z["Ve"]
    paths = [str(ipath(x[0])) for x in pool]
    caps = [x[1] for x in pool]
    U, V = encode(paths, caps)
    ep = [str(ipath(x[0])) for x in ev]
    ec = [x[1] for x in ev]
    Ue, Ve = encode(ep, ec)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, U=U, V=V, Ue=Ue, Ve=Ve)
    return U, V, Ue, Ve


class Proj(nn.Module):
    """Residual projection: starts as CLIP, then learns a contrastive update."""
    def __init__(self, d=512):
        super().__init__()
        self.i = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.t = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        for h in (self.i, self.t):
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)

    def forward(self, u, v):
        return (F.normalize(u + self.i(u), dim=-1),
                F.normalize(v + self.t(v), dim=-1))


def train_proj(U, V, keep, seed, epochs=5, bs=512):
    torch.manual_seed(seed)
    model = Proj().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    U = torch.tensor(U[keep], device="cuda")
    V = torch.tensor(V[keep], device="cuda")
    n = len(U)
    for ep in range(epochs):
        perm = torch.randperm(n, device="cuda")
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            ui, vt = model(U[j], V[j])
            logits = ui @ vt.T / 0.07
            loss = F.cross_entropy(logits, torch.arange(len(j), device="cuda"))
            opt.zero_grad(); loss.backward(); opt.step()
        log("proj epoch", ep, "loss", round(float(loss), 4))
    model.eval()
    return model


@torch.no_grad()
def r1_correct(model, Ue, Vbank, owner):
    """Retrieve from the full 113k caption bank, not only the 1k eval captions."""
    Ut = torch.tensor(Ue, device="cuda")
    ui = F.normalize(Ut + model.i(Ut), dim=-1)
    best = torch.full((len(ui),), -1e9, device="cuda")
    best_idx = torch.zeros(len(ui), dtype=torch.long, device="cuda")
    for i in range(0, len(Vbank), 8192):
        Vt = torch.tensor(Vbank[i:i + 8192], device="cuda")
        vt = F.normalize(Vt + model.t(Vt), dim=-1)
        s = ui @ vt.T
        v, k = s.max(1)
        m = v > best
        best[m] = v[m]
        best_idx[m] = k[m] + i
    top = best_idx.cpu().numpy()
    return (owner[top] == np.arange(len(Ue))).astype(float)


def boot_diff(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        d.append(a[idx].mean() - b[idx].mean())
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--eval_limit", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pool, ev = load_pool()
    if args.limit:
        pool = pool[:args.limit]
    ev = ev[:args.eval_limit]
    log("pool", len(pool), "eval", len(ev))
    cache = (CACHE if not args.limit else
             OUT / f"features_{args.limit}_{args.eval_limit}.npz")
    U, V, Ue, Ve = get_features(pool, ev, cache)
    Vbank = np.concatenate([V, Ve])
    owner = np.concatenate([-np.ones(len(V), dtype=np.int64),
                            np.arange(len(ev), dtype=np.int64)])

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        donor = hard_donor(V, args.topk, seed)
        noisy = rng.random(len(U)) < args.noise
        Vobs = V.copy(); Vobs[noisy] = V[donor[noisy]]
        cos = (U * Vobs).sum(1)
        scores = {
            "Cosine": cos,
            "CSLS": csls_scores(U, Vobs),
            "ABTT": abtt_scores(U, Vobs),
            "CRD": crd_scores(U, Vobs, noisy, seed)[0],
            "Oracle": (1 - noisy).astype(float),
            "None": random_scores(len(U), seed + 10007),
        }
        nkeep = int(round(args.budget * len(U)))
        for arm, sc in scores.items():
            keep = np.argsort(-sc)[:nkeep]
            model = train_proj(U, Vobs, keep, seed)
            corr = r1_correct(model, Ue, Vbank, owner)
            rows.append({"seed": seed, "arm": arm, "r1": float(corr.mean()),
                         "noise_keep": float(noisy[keep].mean())})
            log("R1", seed, arm, round(float(corr.mean()), 4))
            np.save(OUT / f"correct_s{seed}_{arm}.npy", corr)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "p1_full_retrieval.csv", index=False)
    df = pd.DataFrame(rows)
    summary = df.groupby("arm").r1.agg(["mean", "std"]).sort_values("mean",
                                                                   ascending=False)
    print(summary.round(4).to_string())
    # paired bootstrap CRD vs Cosine, same seed/reuse stored correctness
    for seed in range(args.seeds):
        crd = np.load(OUT / f"correct_s{seed}_CRD.npy")
        cos = np.load(OUT / f"correct_s{seed}_Cosine.npy")
        m, lo, hi = boot_diff(crd, cos, seed=seed)
        print(f"seed {seed} CRD-Cosine diff {m:+.4f} CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
