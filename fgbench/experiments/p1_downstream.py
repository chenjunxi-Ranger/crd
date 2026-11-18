"""P1 downstream cleaning on COCO Karpathy train+restval.

Self-contained: CLIP features -> pair-level donor noise -> six cleaning arms ->
linear match probe on a clean COCO val set.

Example (server):
  HF_ENDPOINT=https://hf-mirror.com python3 p1_downstream.py \
    --karpathy /root/fgb_data/karpathy/dataset_coco.json \
    --coco /root/fgb_data/COCO --out /root/fgb_out/p1 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from transformers import CLIPModel, CLIPProcessor

MODEL = "openai/clip-vit-base-patch16"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def img_path(coco: Path, name: str) -> Path:
    sub = "train2014" if name.startswith("COCO_train2014") else "val2014"
    return coco / sub / name


def load_karpathy(path: Path):
    d = json.load(open(path, encoding="utf-8"))["images"]
    pool, clean = [], []
    for x in d:
        cap = x["sentences"][0]["raw"].strip()
        item = (x["filename"], cap, x["split"])
        if x["split"] in ("train", "restval"):
            pool.append(item)
        elif x["split"] == "val":
            clean.append(item)
    return pool, clean


@torch.no_grad()
def encode_images(model, proc, paths, device, bs=96):
    out = []
    for i in range(0, len(paths), bs):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        x = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        f = model.get_image_features(pixel_values=x)
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        if i % (bs * 100) == 0:
            log("images", i, "/", len(paths))
    return torch.cat(out).numpy().astype("float32")


@torch.no_grad()
def encode_texts(model, proc, texts, device, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        x = proc(text=texts[i:i + bs], padding=True, truncation=True,
                 max_length=77, return_tensors="pt").to(device)
        f = model.get_text_features(**x)
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out).numpy().astype("float32")


def get_features(args, pool, clean):
    cache = Path(args.out) / "features.npz"
    if cache.is_file() and not args.no_cache:
        z = np.load(cache)
        if len(z["U"]) == len(pool) and len(z["val_U"]) == len(clean):
            log("loaded features", cache)
            return z["U"], z["V"], z["val_U"], z["val_V"]
    log("load CLIP", MODEL)
    model = CLIPModel.from_pretrained(MODEL).to(args.device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    U = encode_images(model, proc, [img_path(Path(args.coco), x[0]) for x in pool],
                      args.device)
    V = encode_texts(model, proc, [x[1] for x in pool], args.device)
    val_U = encode_images(model, proc,
                          [img_path(Path(args.coco), x[0]) for x in clean],
                          args.device)
    val_V = encode_texts(model, proc, [x[1] for x in clean], args.device)
    if not args.no_cache:
        np.savez_compressed(cache, U=U, V=V, val_U=val_U, val_V=val_V)
    return U, V, val_U, val_V


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def csls_scores(U, V, seed=0, bank=8000, k=10):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(U), min(bank, len(U)), replace=False)
    Ut = torch.tensor(U, device="cuda")
    Vt = torch.tensor(V, device="cuda")
    Ub, Vb = Ut[idx], Vt[idx]
    ri, rt = [], []
    for i in range(0, len(Ut), 4096):
        s = Ut[i:i + 4096] @ Vb.T
        ri.append(s.topk(k, dim=1).values.mean(1))
        s2 = Vt[i:i + 4096] @ Ub.T
        rt.append(s2.topk(k, dim=1).values.mean(1))
    ri = torch.cat(ri).cpu().numpy()
    rt = torch.cat(rt).cpu().numpy()
    cos = (U * V).sum(1)
    return 2 * cos - ri - rt


def abtt_scores(U, V, n_remove=10):
    def one(x):
        x = x - x.mean(0, keepdims=True)
        q = torch.pca_lowrank(torch.tensor(x, device="cuda"), q=n_remove + 4)[2]
        x = x - (x @ q[:, :n_remove].cpu().numpy()) @ q[:, :n_remove].cpu().numpy().T
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return (one(U) * one(V)).sum(1)


def crd_scores(U, V, noisy, seed=0):
    """Additive ridge g; lambda selected on a held-out 5k pool slice."""
    z = (U * V).sum(1)
    zs = standardize(z)
    rng = np.random.default_rng(seed)
    n = len(z)
    hold = rng.choice(n, min(5000, n // 5), replace=False)
    mask = np.ones(n, bool)
    mask[hold] = False
    X = np.concatenate([U, V, np.ones((n, 1), np.float32)], 1)
    Xt = torch.tensor(X, device="cuda")
    zt = torch.tensor(zs, device="cuda")
    Xtr, ztr = Xt[mask], zt[mask]
    A = Xtr.T @ Xtr + 1e-2 * torch.eye(X.shape[1], device="cuda")
    w = torch.linalg.solve(A, Xtr.T @ ztr)
    g = (Xt @ w).cpu().numpy()
    best = (0, -1.0)
    for lam in np.arange(0, 2.01, 0.25):
        auc = roc_auc_score(noisy[hold], -(zs[hold] - lam * g[hold]))
        if auc > best[1]:
            best = (lam, auc)
    lam = best[0]
    log("CRD lambda", lam, "holdout AUC", round(best[1], 4))
    return zs - lam * g, lam


def random_scores(n, seed):
    return np.random.default_rng(seed).random(n)


def run_probe(U, V, val_U, val_V, keep, labels, seed):
    rng = np.random.default_rng(seed)
    Uv = val_U
    Vp = val_V
    neg = rng.permutation(len(Vp))
    neg = np.where(neg == np.arange(len(neg)), (neg + 1) % len(neg), neg)
    Xp = Uv * Vp
    Xn = Uv * Vp[neg]
    Xv = np.concatenate([Xp, Xn])
    yv = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    clf = LogisticRegression(max_iter=300, solver="liblinear", C=1.0)
    clf.fit((U * V)[keep], labels[keep])
    p = clf.predict_proba(Xv)[:, 1]
    return (accuracy_score(yv, p > 0.5), average_precision_score(yv, p),
            roc_auc_score(yv, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--karpathy", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=2000)
    ap.add_argument("--noise_levels", default="0.4")
    ap.add_argument("--budgets", default="0.5")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    pool, clean = load_karpathy(Path(args.karpathy))
    if args.limit:
        pool = pool[:args.limit]
    if args.val_limit:
        clean = clean[:args.val_limit]
    log("pool", len(pool), "clean", len(clean))
    U, V, val_U, val_V = get_features(args, pool, clean)
    log("features", U.shape, V.shape, val_U.shape)
    rng0 = np.random.default_rng(0)
    caps = np.array([x[1] for x in pool], dtype=object)
    vals = np.array([x[1] for x in clean], dtype=object)

    rows = []
    for nz in [float(x) for x in args.noise_levels.split(",")]:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            rng = np.random.default_rng(seed)
            noisy = rng.random(len(pool)) < nz
            donor = rng.integers(0, len(pool), len(pool))
            donor = np.where(donor == np.arange(len(pool)),
                             (donor + 1) % len(pool), donor)
            obs = caps.copy()
            obs[noisy] = caps[donor[noisy]]
            Vobs = V.copy()
            Vobs[noisy] = V[donor[noisy]]
            cos = (U * Vobs).sum(1)
            scores = {
                "Cosine": cos,
                "CSLS": csls_scores(U, Vobs),
                "ABTT": abtt_scores(U, Vobs),
                "CRD": crd_scores(U, Vobs, noisy, seed)[0],
                "Oracle": (1 - noisy).astype(float),
                "None": random_scores(len(pool), seed + 10007),
            }
            for budget in [float(x) for x in args.budgets.split(",")]:
                nkeep = int(round(budget * len(pool)))
                for arm, sc in scores.items():
                    keep = np.argsort(-sc)[:nkeep]
                    Ukeep, Vkeep = U[keep], Vobs[keep]
                    # probe labels: positive = observed pair, negative = donor
                    neg = rng.permutation(len(keep))
                    neg = np.where(neg == np.arange(len(neg)),
                                   (neg + 1) % len(neg), neg)
                    Xp = Ukeep * Vkeep
                    Xn = Ukeep * Vkeep[neg]
                    X = np.concatenate([Xp, Xn])
                    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
                    clf = LogisticRegression(max_iter=300, solver="liblinear",
                                             C=1.0)
                    clf.fit(X, y)
                    # clean val
                    vneg = rng.permutation(len(val_V))
                    vneg = np.where(vneg == np.arange(len(vneg)),
                                    (vneg + 1) % len(vneg), vneg)
                    Xv = np.concatenate([val_U * val_V, val_U * val_V[vneg]])
                    yv = np.concatenate([np.ones(len(val_U)),
                                         np.zeros(len(val_U))])
                    p = clf.predict_proba(Xv)[:, 1]
                    rows.append({
                        "noise": nz, "budget": budget, "arm": arm,
                        "seed": seed, "n_keep": len(keep),
                        "noise_keep": int(noisy[keep].sum()),
                        "noise_removed": round(1 - noisy[keep].mean(), 4),
                        "clean_recall": round((~noisy[keep]).sum() /
                                              max((~noisy).sum(), 1), 4),
                        "val_acc": round(accuracy_score(yv, p > 0.5), 4),
                        "val_ap": round(average_precision_score(yv, p), 4),
                        "val_auc": round(roc_auc_score(yv, p), 4),
                    })
                    log("done", nz, seed, budget, arm)
                    pd.DataFrame(rows).to_csv(Path(args.out) / "p1_results.csv",
                                              index=False)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out) / "p1_results.csv", index=False)
    print(df.groupby(["noise", "budget", "arm"])[
        ["val_acc", "val_ap", "val_auc", "noise_removed"]].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
