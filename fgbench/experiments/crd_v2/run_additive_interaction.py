"""CRD v2 candidate: additive calibration PLUS a pair-interaction term.

The probe experiment showed that per-side features carry no class signal
(`LR[u;v]` = 0.5000 on every facet) while `u * v` does.  This script tests the
narrow question that follows:

    S = z - lam * q_add(u,v) + eta * g(u,v)

where q_add is the frozen CRD v1 head (subtracted, as before) and g is a NEW
head trained to judge the pair itself (added, with the opposite sign).  Unlike
`run_interaction.py` and `residual_interaction.py`, g is not regressed on z and
is not subtracted: it is trained to separate clean from noisy pairs and enters
with its own coefficient eta, selected on validation.

Interaction features (no per-side vector on purpose, to avoid duplicating q_add):

    dot       u * v                     (512)
    absdiff   |u - v|                   (512)
    dot_abs   [u*v ; |u-v|]             (1024)   <- proposed default
    uvdot     [u ; v ; u*v]             (1536)   <- probe reference
    placebo   [u*v_p ; |u-v_p|]         (1024)   <- v permuted, pairing destroyed

Training schemes for g (higher = cleaner):

    A_pair    label-free: original pair vs pair with a text from another cycle
    B_rank    supervised: softplus(-(g_clean - g_noisy) / tau) across cycles
    B_bce     supervised: BCE on the pair label (probe-equivalent reference)

eta is selected on validation AUROC from {0, 0.05, ..., 2.0}; eta = 0 is E1.
Test is scored once.  E3 - E1 is tested with a cycle-clustered paired bootstrap.

Run:
    python fgbench/experiments/crd_v2/run_additive_interaction.py --stage 1
    python fgbench/experiments/crd_v2/run_additive_interaction.py --stage 2
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
sys.path.insert(0, str(ROOT / "fgbench" / "models"))

from fgbench.models.crd_backbone import Off  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "crd_v2"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

FACETS = {"Entity": ("COCO", "Entity"), "Quantity": ("COCO", "Quantity"),
          "Attribute": ("VG", "Attribute"), "Action": ("VG", "Action"),
          "Relation": ("VG", "Relation")}
CYCLES = {"Entity": 8000, "Quantity": 2500, "Attribute": 8000,
          "Action": 0, "Relation": 2000}

STEPS = 1500
BATCH = 256
LR = 1e-3
WIDTH = 128
SEEDS = (0, 1, 2)
ETA_GRID = np.round(np.arange(0.0, 2.001, 0.05), 2)
TAU = 0.5


# ---------------------------------------------------------------- data
def load_split(facet: str, split: str, mu: float, sd: float):
    ds, fc = FACETS[facet]
    if split == "train":
        fn = CACHE / f"crdbb_clip_{ds}_{fc}_c{CYCLES[facet]}_train.npz"
        if not fn.is_file():                     # Action has no cached train
            fn = CACHE / f"crdbb_clip_{ds}_{fc}_train.npz"
    else:
        fn = CACHE / f"crdbb_clip_{ds}_{fc}_{split}.npz"
    z = np.load(fn, allow_pickle=True)
    meta = pd.read_parquet(fn.with_suffix(".meta.parquet"))
    U, V = z["U"].astype(np.float32), z["V"].astype(np.float32)
    S = (U * V).sum(1)
    return U, V, (S - mu) / sd, meta


def crd_v1(facet: str):
    """Frozen CRD v1: head, mu/sigma, lambda and a scorer for (U, V)."""
    ds, fc = FACETS[facet]
    prov = json.loads((CACHE / f"crdbb_prov_clip_{ds}_{fc}_both_row_s0.json")
                      .read_text(encoding="utf-8"))
    mu, sd = float(prov["mu"]), float(prov["sd"])
    ck = torch.load(CACHE / f"crdbb_clip_{ds}_{fc}_both_row_s0_head.pt",
                    map_location=DEV, weights_only=False)
    head = Off(d=int(ck["dim"]), parts="both").to(DEV)
    head.load_state_dict(ck["state"])
    head.eval()
    zsc = np.load(CACHE / f"crdbb_scores_clip_{ds}_{fc}_both_row_s0.npz",
                  allow_pickle=True)
    lam = float(zsc["lam"][0])

    @torch.no_grad()
    def q(U, V):
        return head(torch.tensor(U, device=DEV),
                    torch.tensor(V, device=DEV)).cpu().numpy().ravel()

    def score_sim(U, V):
        z = ((U * V).sum(1) - mu) / sd
        return z - lam * q(U, V)                  # similarity space
    return mu, sd, lam, score_sim


# ---------------------------------------------------------------- features
def feats(U: np.ndarray, V: np.ndarray, kind: str, perm: np.ndarray | None = None):
    if kind == "placebo":
        assert perm is not None
        V = V[perm]
        return np.concatenate([U * V, np.abs(U - V)], 1)
    if kind == "dot":
        return U * V
    if kind == "absdiff":
        return np.abs(U - V)
    if kind == "dot_abs":
        return np.concatenate([U * V, np.abs(U - V)], 1)
    if kind == "uvdot":
        return np.concatenate([U, V, U * V], 1)
    raise ValueError(kind)


class Head(nn.Module):
    def __init__(self, d: int, width: int = WIDTH, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, width), nn.GELU(), nn.Dropout(p),
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------- training
def negatives(batch_idx, cycle, image, caption, rng):
    """For every row, pick a row from another cycle with another image/text."""
    n = len(batch_idx)
    out = np.empty(n, dtype=np.int64)
    for k in range(n):
        i = batch_idx[k]
        for _ in range(20):
            j = int(rng.integers(0, len(cycle)))
            if (cycle[j] != cycle[i] and image[j] != image[i]
                    and caption[j] != caption[i]):
                out[k] = j
                break
        else:
            out[k] = batch_idx[(k + 1) % n]
    return out


def train_head(kind, facet, seed, scheme, Xtr, ytr, meta_tr, Xva, yva):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Head(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(ytr, device=DEV, dtype=torch.float32)
    Xv = torch.tensor(Xva, device=DEV)
    cycle = meta_tr.cycle_id.to_numpy()
    image = meta_tr.target_image_id.to_numpy()
    caption = meta_tr.caption.to_numpy()
    lossf = nn.BCEWithLogitsLoss()

    for _ in range(STEPS):
        idx = rng.integers(0, len(Xt), BATCH)
        xb, yb = Xt[idx], yt[idx]
        if scheme == "A_pair":
            neg = negatives(idx, cycle, image, caption, rng)
            pos = model(xb)
            neg_s = model(Xt[neg])
            loss = lossf(pos, torch.ones_like(pos)) + lossf(neg_s, torch.zeros_like(neg_s))
        elif scheme == "B_bce":
            loss = lossf(model(xb), 1.0 - yb)      # g high = clean
        elif scheme == "B_rank":
            pos_i, neg_i = idx[yb.cpu().numpy() == 0], idx[yb.cpu().numpy() == 1]
            if len(pos_i) == 0 or len(neg_i) == 0:
                continue
            j = rng.integers(0, len(neg_i), len(pos_i))
            g_pos = model(Xt[pos_i])
            g_neg = model(Xt[neg_i[j]])
            loss = F.softplus(-(g_pos - g_neg) / TAU).mean()
        else:
            raise ValueError(scheme)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        g_tr = model(Xt).cpu().numpy()
        g_va = model(Xv).cpu().numpy()
    # standardise on TRAIN so eta has a comparable scale across facets/seeds
    mu, sd = float(g_tr.mean()), float(g_tr.std() + 1e-8)
    return model, mu, sd, (g_tr - mu) / sd, (g_va - mu) / sd


# ---------------------------------------------------------------- metrics
def auc(y, s):
    return float(roc_auc_score(y, -s))            # s is in similarity space


def acc2afc(score, cycle, y):
    df = pd.DataFrame({"c": cycle, "y": y, "s": score})
    out = []
    for _, g in df.groupby("c"):
        if len(g) == 4:
            out.append(g[g.y == 0].s.sum() < g[g.y == 1].s.sum())
    return float(np.mean(out)) if out else float("nan")


def holm(p):
    order = np.argsort(p)
    m, out, run = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facet", default="Attribute")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    facet = args.facet
    ds, fc = FACETS[facet]

    mu, sd, lam, crd_sim = crd_v1(facet)
    Utr, Vtr, ztr, mtr = load_split(facet, "train", mu, sd)
    Uva, Vva, zva, mva = load_split(facet, "val", mu, sd)
    Ute, Vte, zte, mte = load_split(facet, "test", mu, sd)
    ytr = mtr.pair_label.to_numpy().astype(np.int64)          # 1 = noisy
    yva = mva.pair_label.to_numpy().astype(np.int64)
    yte = mte.pair_label.to_numpy().astype(np.int64)
    S1_va, S1_te = crd_sim(Uva, Vva), crd_sim(Ute, Vte)
    print(f"{facet}: train {len(ytr)} / val {len(yva)} / test {len(yte)} | "
          f"cos {auc(yte, zte):.4f} | CRD {auc(yte, S1_te):.4f}", flush=True)

    if args.stage == 1:
        configs = [("A_pair", "dot_abs"), ("B_rank", "dot_abs"), ("B_bce", "dot_abs")]
    elif args.stage == 2:
        configs = [("B_rank", k) for k in ("dot", "absdiff", "uvdot", "placebo")]
        configs += [("A_pair", "dot"), ("A_pair", "absdiff")]
    else:
        # confirmation stage: matched probes on the winning feature form
        configs = [("B_rank", "uvdot"), ("B_bce", "uvdot")]

    perm_tr = np.random.default_rng(7).permutation(len(Utr))
    perm_va = np.random.default_rng(7).permutation(len(Uva))
    perm_te = np.random.default_rng(7).permutation(len(Ute))
    rows, keep = [], {}
    for scheme, kind in configs:
        Xtr = feats(Utr, Vtr, kind, perm_tr)
        Xva = feats(Uva, Vva, kind, perm_va)
        Xte = feats(Ute, Vte, kind, perm_te)
        for seed in seeds:
            t0 = time.time()
            model, gm, gs, gtr, gva = train_head(
                kind, facet, seed, scheme, Xtr, ytr, mtr, Xva, yva)
            with torch.no_grad():
                Xtt = torch.tensor(Xte, device=DEV)
                gte = (model(Xtt).cpu().numpy() - gm) / gs
            # E2: interaction head alone
            e2 = auc(yte, gte)
            # E3: eta on validation, includes eta = 0 (E1)
            va_auc = [auc(yva, S1_va + e * gva) for e in ETA_GRID]
            eta = float(ETA_GRID[int(np.argmax(va_auc))])
            S3 = S1_te + eta * gte
            rows.append({
                "facet": facet, "scheme": scheme, "features": kind, "seed": seed,
                "cosine": round(auc(yte, zte), 4),
                "E1_crd": round(auc(yte, S1_te), 4),
                "E2_interaction": round(e2, 4),
                "eta": eta, "val_E3": round(float(max(va_auc)), 4),
                "E3_crd_plus": round(auc(yte, S3), 4),
                "E3_minus_E1_pp": round((auc(yte, S3) - auc(yte, S1_te)) * 100, 2),
                "E2_auprc": round(float(average_precision_score(yte, -gte)), 4),
                "E3_auprc": round(float(average_precision_score(yte, -S3)), 4),
                "E1_2afc": round(acc2afc(S1_te, mte.cycle_id.values, 1 - yte), 6),
                "E3_2afc": round(acc2afc(S3, mte.cycle_id.values, 1 - yte), 6),
                "seconds": round(time.time() - t0, 1),
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            keep[f"{scheme}_{kind}_{seed}"] = (
                mte.cycle_id.values, yte, S1_te, S3, gte)

    df = pd.DataFrame(rows)
    name = f"additive_interaction_stage{args.stage}_{facet}.csv"
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    if keep:
        payload = {}
        for k, (cyc, y, s1, s3, g) in keep.items():
            payload[f"{k}__cycle"] = cyc
            payload[f"{k}__y"] = y
            payload[f"{k}__E1"] = s1
            payload[f"{k}__E3"] = s3
            payload[f"{k}__g"] = g
        np.savez_compressed(
            OUT / f"interaction_scores_{facet}_stage{args.stage}.npz", **payload)
    print("\n", df.to_string(index=False), flush=True)
    print(f"\nwrote {OUT / name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
