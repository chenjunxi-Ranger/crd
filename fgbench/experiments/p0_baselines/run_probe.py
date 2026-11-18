"""P0-2: supervised probes on the frozen CLIP ViT-B/16 features.

Motivation.  CRD uses labels only to pick lambda: its heads are regressed on the
pair's own standardised score.  A reviewer will ask what a classifier trained on
the same features and the same training subset would achieve.  This script
answers that question with four probes:

    lr_uv      logistic regression on [u; v]            (per-side offsets only)
    lr_dot     logistic regression on u * v             (bilinear match term)
    lr_uvdot   logistic regression on [u; v; u * v]
    mlp_uvdot  two-layer MLP on [u; v; u * v]           (3 seeds)

The probes are trained on exactly the training subset CRD consumes, they see the
pair label (so they are fully supervised), and every hyper-parameter (the LR
regularisation strength) is selected on the validation split.  The test split is
evaluated once, on the frozen FGBench v1-strict manifest.

Reported per facet: AUROC, AUPRC and the cycle-level 2AFC used in the main table.
Per-sample test scores are written so that the paired bootstrap can compare a
probe with CRD on identical rows.

Run:  python fgbench/experiments/p0_baselines/run_probe.py
      python fgbench/experiments/p0_baselines/run_probe.py --skip-action
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from run_baselines import FACETS, action_splits, load_cache  # noqa: E402

OUT = ROOT / "results" / "fgbench" / "p0_baselines"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "results" / "fgbench"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA_SEED = 0

C_GRID = [0.01, 0.1, 1.0, 10.0]
MLP_STEPS = 1500
MLP_BATCH = 256
MLP_LR = 1e-3
MLP_WIDTH = 256
MLP_SEEDS = (0, 1, 2)


# ------------------------------------------------------------------ features
def build(U: np.ndarray, V: np.ndarray, kind: str) -> np.ndarray:
    if kind == "u":
        return U
    if kind == "v":
        return V
    if kind == "uv":
        return np.concatenate([U, V], 1)
    if kind == "dot":
        return U * V
    if kind == "uvdot":
        return np.concatenate([U, V, U * V], 1)
    raise ValueError(kind)


def labels(d: pd.DataFrame) -> np.ndarray:
    return d.pair_label.to_numpy().astype(np.int64)


# ------------------------------------------------------------------ metrics
def aauc(y, a) -> float:
    """AUROC with an anomaly score (higher = noisier)."""
    return float(roc_auc_score(y, a))


def acc2afc(score: np.ndarray, d: pd.DataFrame) -> float:
    """Cycle-level 2AFC: clean-sum < noisy-sum, with `score` in anomaly space."""
    out = []
    for _, g in pd.DataFrame({"c": d.cycle_id.values,
                              "y": d.pair_label.values,
                              "s": score}).groupby("c"):
        if len(g) != 4:
            continue
        out.append(g[g.y == 0].s.sum() < g[g.y == 1].s.sum())
    return float(np.mean(out)) if out else float("nan")


# ------------------------------------------------------------------ models
class MLP(nn.Module):
    def __init__(self, d: int, width: int = MLP_WIDTH, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, width), nn.GELU(), nn.Dropout(p), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_lr(Xtr, ytr, Xva, yva):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000, solver="lbfgs")
        clf.fit(Xtr_s, ytr)
        a = aauc(yva, clf.decision_function(Xva_s))
        if best is None or a > best[1]:
            best = (c, a, clf)
    c, val_auc, clf = best
    return (lambda X: clf.decision_function(sc.transform(X))), {"C": c,
                                                                "val_auroc": round(val_auc, 4)}


def fit_mlp(Xtr, ytr, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, device=DEV, dtype=torch.float32)
    yt = torch.tensor(ytr, device=DEV, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss()
    n = len(Xt)
    model.train()
    for _ in range(MLP_STEPS):
        idx = torch.tensor(rng.integers(0, n, MLP_BATCH), device=DEV)
        loss = lossf(model(Xt[idx]), yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    @torch.no_grad()
    def score(X):
        return model(torch.tensor(X, device=DEV, dtype=torch.float32)).cpu().numpy()
    return score


# ------------------------------------------------------------------ CRD side
def crd_scores(ds: str, facet: str, d: pd.DataFrame):
    """Integrated (3-seed mean) CRD score, anomaly space, aligned to `d`."""
    if facet == "Action":
        paths = [CACHE / f"crd_action_scores_s{s}.npz" for s in (0, 1, 2)]
        key = "crd"
    else:
        paths = [CACHE / f"crdbb_scores_clip_{ds}_{facet}_s{s}.npz" for s in (0, 1, 2)]
        key = "crd"
    if not all(p.is_file() for p in paths):
        return None
    sims = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if len(z["cycle_id"]) != len(d):
            return None
        if not np.array_equal(z["cycle_id"].astype(str), d.cycle_id.astype(str).values):
            return None
        sims.append(z[key].astype(np.float64))
    return -np.mean(sims, axis=0)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-action", action="store_true")
    ap.add_argument("--only", default="", help="single facet name, for debugging")
    args = ap.parse_args()

    rows, per_sample, meta = [], {}, {}
    for ds, facet, cycles in FACETS:
        if args.only and facet != args.only:
            continue
        if facet == "Action":
            if args.skip_action:
                print("[skip] Action (re-encoding disabled by --skip-action)", flush=True)
                continue
            (dtr, Utr, Vtr), (dva, Uva, Vva), (dte, Ute, Vte) = action_splits()
        else:
            dtr, Utr, Vtr = load_cache(ds, facet, "train", cycles)
            dva, Uva, Vva = load_cache(ds, facet, "val")
            dte, Ute, Vte = load_cache(ds, facet, "test")

        ytr, yva, yte = labels(dtr), labels(dva), labels(dte)
        t0 = time.time()
        print(f"{facet:9s} train {len(ytr):6d} | val {len(yva):5d} | test {len(yte):6d}",
              flush=True)

        for kind, name in (("uv", "lr_uv"), ("dot", "lr_dot"), ("uvdot", "lr_uvdot")):
            Xtr = build(Utr, Vtr, kind)
            Xva = build(Uva, Vva, kind)
            Xte = build(Ute, Vte, kind)
            score, info = fit_lr(Xtr, ytr, Xva, yva)
            s_te = score(Xte)
            rows.append({
                "dataset": ds, "facet": facet, "method": name,
                "dim": int(Xtr.shape[1]), "n_seeds": 1,
                "auroc": round(aauc(yte, s_te), 4),
                "auprc": round(float(average_precision_score(yte, s_te)), 4),
                "acc2afc": round(acc2afc(s_te, dte), 6),
                "val_auroc": info["val_auroc"], "hyper": f"C={info['C']}",
            })
            per_sample[f"{name}"] = s_te
            print(f"  {name:9s} AUROC {rows[-1]['auroc']:.4f} | val {info['val_auroc']:.4f}",
                  flush=True)

        Xtr = build(Utr, Vtr, "uvdot")
        Xva = build(Uva, Vva, "uvdot")
        Xte = build(Ute, Vte, "uvdot")
        te_scores, va_aucs = [], []
        for seed in MLP_SEEDS:
            f = fit_mlp(Xtr, ytr, seed)
            s_va, s_te = f(Xva), f(Xte)
            te_scores.append(s_te)
            va_aucs.append(aauc(yva, s_va))
            rows.append({
                "dataset": ds, "facet": facet, "method": f"mlp_uvdot_s{seed}",
                "dim": int(Xtr.shape[1]), "n_seeds": 1,
                "auroc": round(aauc(yte, s_te), 4),
                "auprc": round(float(average_precision_score(yte, s_te)), 4),
                "acc2afc": round(acc2afc(s_te, dte), 6),
                "val_auroc": round(va_aucs[-1], 4), "hyper": f"seed={seed}",
            })
        mean_te = np.mean(te_scores, axis=0)
        a_seeds = [aauc(yte, s) for s in te_scores]
        rows.append({
            "dataset": ds, "facet": facet, "method": "mlp_uvdot",
            "dim": int(Xtr.shape[1]), "n_seeds": len(MLP_SEEDS),
            "auroc": round(aauc(yte, mean_te), 4),
            "auprc": round(float(average_precision_score(yte, mean_te)), 4),
            "acc2afc": round(acc2afc(mean_te, dte), 6),
            "val_auroc": round(float(np.mean(va_aucs)), 4),
            "hyper": f"mean of {len(MLP_SEEDS)} seeds",
        })
        per_sample["mlp_uvdot"] = mean_te
        print(f"  mlp_uvdot seed AUROC {np.round(a_seeds, 4).tolist()} "
              f"mean {np.mean(a_seeds):.4f} | {time.time()-t0:.0f}s", flush=True)

        c = crd_scores(ds, facet, dte)
        if c is not None:
            print(f"  CRD integrated AUROC {aauc(yte, c):.4f}", flush=True)

        meta[f"{ds}_{facet}"] = {
            "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
            "n_train_cycles": int(dtr.cycle_id.nunique()),
            "n_test_cycles": int(dte.cycle_id.nunique()),
            "pos_rate_train": round(float(ytr.mean()), 4),
        }
        payload = {"cycle_id": dte.cycle_id.values, "y": yte,
                   "val_y": yva}
        payload.update({k: v for k, v in per_sample.items()})
        if c is not None:
            payload["crd"] = c
        np.savez_compressed(OUT / f"probe_scores_{ds}_{facet}.npz", **payload)
        per_sample = {}

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probe_raw.csv", index=False)
    (OUT / "probe_meta.json").write_text(
        json.dumps({"device": DEV, "data_seed": DATA_SEED, "c_grid": C_GRID,
                    "mlp": {"steps": MLP_STEPS, "batch": MLP_BATCH, "lr": MLP_LR,
                            "width": MLP_WIDTH, "seeds": list(MLP_SEEDS)},
                    "split": meta}, indent=2), encoding="utf-8")
    print(df.to_string(index=False), flush=True)
    print(f"\nwrote {OUT/'probe_raw.csv'} ({len(df)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
