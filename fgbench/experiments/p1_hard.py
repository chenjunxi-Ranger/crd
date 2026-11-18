"""Hard P1: semantic-neighbour donor noise + low retention + stronger probe."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from p1_downstream import (abtt_scores, csls_scores, crd_scores,
                           load_karpathy, random_scores)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def hard_donor(V, topk, seed):
    Vt = torch.tensor(V, device="cuda")
    n = len(Vt)
    donor = np.empty(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for i in range(0, n, 2048):
        s = Vt[i:i + 2048] @ Vt.T
        s[torch.arange(len(s), device="cuda"),
          torch.arange(i, i + len(s), device="cuda")] = -1e9
        idx = s.topk(min(topk, n - 1), dim=1).indices.cpu().numpy()
        pick = rng.integers(0, idx.shape[1], len(idx))
        donor[i:i + len(idx)] = idx[np.arange(len(idx)), pick]
    return donor


def hard_neg(V, n_neg):
    Vt = torch.tensor(V, device="cuda")
    out = []
    for i in range(0, len(Vt), 2048):
        s = Vt[i:i + 2048] @ Vt.T
        s[torch.arange(len(s), device="cuda"),
          torch.arange(i, i + len(s), device="cuda")] = -1e9
        idx = s.topk(n_neg, dim=1).indices.cpu().numpy()
        out.append(idx)
    return np.concatenate(out)


def mlp_probe(X, y, Xv, yv, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(),
                          nn.Dropout(0.1), nn.Linear(256, 1)).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, device="cuda", dtype=torch.float32)
    yt = torch.tensor(y, device="cuda", dtype=torch.float32)
    for _ in range(10):
        perm = torch.randperm(len(Xt), device="cuda")
        for i in range(0, len(Xt), 4096):
            j = perm[i:i + 4096]
            opt.zero_grad()
            loss = lossf(model(Xt[j]).squeeze(1), yt[j])
            loss.backward()
            opt.step()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(Xv, device="cuda",
                                             dtype=torch.float32)).squeeze(1))
    p = p.cpu().numpy()
    return (accuracy_score(yv, p > 0.5), average_precision_score(yv, p),
            roc_auc_score(yv, p))


def make_train(U, Vobs, keep, nneg, seed):
    rng = np.random.default_rng(seed)
    Ukeep, Vkeep = U[keep], Vobs[keep]
    neg_idx = hard_neg(Vkeep, nneg)
    Xp, Xn = [], []
    for k in range(nneg):
        Xp.append(Ukeep * Vkeep)
        Xn.append(Ukeep * Vkeep[neg_idx[:, k]])
    X = np.concatenate(Xp + Xn)
    y = np.concatenate([np.ones(len(Ukeep))] * nneg +
                       [np.zeros(len(Ukeep))] * nneg)
    return X, y


def make_val(val_U, val_V, nneg):
    neg = hard_neg(val_V, nneg)
    Xp = [val_U * val_V] * nneg
    Xn = [val_U * val_V[neg[:, k]] for k in range(nneg)]
    X = np.concatenate(Xp + Xn)
    y = np.concatenate([np.ones(len(val_U))] * nneg +
                       [np.zeros(len(val_U))] * nneg)
    return X, y


def run_round(U, V, val_U, val_V, caps, round_cfg, noise=0.4, seeds=2):
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        donor = hard_donor(V, round_cfg["topk"], seed)
        noisy = rng.random(len(U)) < noise
        Vobs = V.copy()
        Vobs[noisy] = V[donor[noisy]]
        cos = (U * Vobs).sum(1)
        scores = {
            "Cosine": cos,
            "CSLS": csls_scores(U, Vobs),
            "ABTT": abtt_scores(U, Vobs),
            "CRD": crd_scores(U, Vobs, noisy, seed)[0],
            "Oracle": (1 - noisy).astype(float),
            "None": random_scores(len(U), seed + 10007),
        }
        budget = round_cfg["budget"]
        nkeep = int(round(budget * len(U)))
        Xv, yv = make_val(val_U, val_V, round_cfg["nneg"])
        for arm, sc in scores.items():
            keep = np.argsort(-sc)[:nkeep]
            X, y = make_train(U, Vobs, keep, round_cfg["nneg"], seed)
            if round_cfg["probe"] == "linear":
                clf = LogisticRegression(max_iter=300, solver="liblinear")
                clf.fit(X, y)
                p = clf.predict_proba(Xv)[:, 1]
                acc, ap, auc = (accuracy_score(yv, p > 0.5),
                                average_precision_score(yv, p),
                                roc_auc_score(yv, p))
            else:
                acc, ap, auc = mlp_probe(X, y, Xv, yv, seed)
            rows.append({
                "round": round_cfg["name"], "noise": noise, "seed": seed,
                "budget": budget, "arm": arm, "n_keep": nkeep,
                "noise_removed": round(float(1 - noisy[keep].mean()), 4),
                "clean_recall": round(float((~noisy[keep]).sum() /
                                            max((~noisy).sum(), 1)), 4),
                "val_acc": round(float(acc), 4),
                "val_ap": round(float(ap), 4),
                "val_auc": round(float(auc), 4),
            })
            log(round_cfg["name"], seed, arm, round(float(acc), 4))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--karpathy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_round", type=int, default=3)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    z = np.load(args.features)
    U, V, val_U, val_V = z["U"], z["V"], z["val_U"], z["val_V"]
    pool, clean = load_karpathy(Path(args.karpathy))
    pool = pool[:len(U)]
    clean = clean[:len(val_U)]
    caps = np.array([x[1] for x in pool], dtype=object)
    log("features", U.shape, val_U.shape)

    configs = [
        {"name": "r1_top100_keep10_linear", "topk": 100, "budget": 0.10,
         "probe": "linear", "nneg": 1},
        {"name": "r2_top10_keep5_mlp", "topk": 10, "budget": 0.05,
         "probe": "mlp", "nneg": 3},
        {"name": "r3_top1_keep5_mlp", "topk": 1, "budget": 0.05,
         "probe": "mlp", "nneg": 5},
    ]
    allrows = []
    for cfg in configs[:args.max_round]:
        log("ROUND", cfg["name"])
        rows = run_round(U, V, val_U, val_V, caps, cfg)
        allrows += rows
        df = pd.DataFrame(allrows)
        df.to_csv(Path(args.out) / "p1_hard_results.csv", index=False)
        d = df[df["round"] == cfg["name"]]
        piv = d.groupby("arm")[["val_acc", "val_ap"]].mean()
        print(piv.round(4).to_string(), flush=True)
        crd = piv.loc["CRD", "val_acc"]
        cos = piv.loc["Cosine", "val_acc"]
        rivals = piv.drop(index=["CRD", "Oracle"], errors="ignore")["val_acc"].max()
        if crd - cos >= 0.01 and crd >= rivals:
            log("CRD ahead of all non-oracle arms by >=1pp, stopping")
            break
    print("done")


if __name__ == "__main__":
    main()
