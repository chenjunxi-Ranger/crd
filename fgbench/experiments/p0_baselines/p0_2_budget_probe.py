"""P0-2: label-budget-matched probe.

The supervised probe in Table II is trained on the full FGBench training split
(32,000 rows for Attribute, 10,000 for Quantity). CRD's lambda is selected on the
validation split (8,336 / 476 rows). This script trains the probe on the SAME
validation rows, so the comparison is budget-matched.

Protocol:
    fit      standardise + logistic regression on the validation split only
    C        5-fold CV inside the validation split
    test     scored once
    MLP      same budget, one seed, width 128

Output: results/fgbench/p0_baselines/p0_2_budget_probe.csv
Run:    python fgbench/experiments/p0_baselines/p0_2_budget_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "experiments" / "p0_baselines"))

from run_baselines import load_cache  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "p0_baselines"
FACETS = [("COCO", "Entity", 8000), ("COCO", "Quantity", 2500),
          ("VG", "Attribute", 8000), ("VG", "Relation", 2000)]
C_GRID = [0.01, 0.1, 1.0, 10.0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def feats(U, V):
    return np.concatenate([U, V, U * V], 1)


class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 128), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(X, y, seed=0, steps=600):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X, device=DEV, dtype=torch.float32)
    yt = torch.tensor(y, device=DEV, dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        idx = torch.tensor(rng.integers(0, len(Xt), 128), device=DEV)
        loss = lossf(model(Xt[idx]), yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def crd_anom(ds, facet):
    zs = [np.load(CACHE / f"crdbb_scores_clip_{ds}_{facet}_both_row_s{s}.npz",
                  allow_pickle=True) for s in (0, 1, 2)]
    return np.mean([-z["crd"] for z in zs], axis=0), zs[0]["cycle_id"], zs[0]["y"]


def main() -> int:
    rows = []
    for ds, facet, cycles in FACETS:
        dtr, Utr, Vtr = load_cache(ds, facet, "train", cycles)
        dva, Uva, Vva = load_cache(ds, facet, "val")
        dte, Ute, Vte = load_cache(ds, facet, "test")
        Xva, Xte = feats(Uva, Vva), feats(Ute, Vte)
        yva = (dva.pair_label.to_numpy() == 0).astype(int)   # 1 = clean
        yte = dte.pair_label.to_numpy()
        crd, cyc, yy = crd_anom(ds, facet)
        assert (cyc == dte.cycle_id.values).all() and (yy == yte).all(), facet

        sc = StandardScaler().fit(Xva)
        Zva, Zte = sc.transform(Xva), sc.transform(Xte)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        best = None
        for C in C_GRID:
            aucs = []
            for tr, va in cv.split(Zva, yva):
                clf = LogisticRegression(C=C, max_iter=3000).fit(Zva[tr], yva[tr])
                aucs.append(roc_auc_score(yva[va], clf.decision_function(Zva[va])))
            if best is None or np.mean(aucs) > best[1]:
                best = (C, float(np.mean(aucs)))
        clf = LogisticRegression(C=best[0], max_iter=3000).fit(Zva, yva)
        # anomaly = noisier
        a_lr = roc_auc_score(yte, -clf.decision_function(Zte))
        a_crd = roc_auc_score(yte, crd)

        # MLP on the same val rows, one seed
        m = fit_mlp(Zva, yva, seed=0, steps=400)
        with torch.no_grad():
            s = m(torch.tensor(Zte, device=DEV, dtype=torch.float32)).cpu().numpy()
        a_mlp = roc_auc_score(yte, -s)

        rows.append({
            "facet": facet, "n_train_full": int(len(yva) * 4),
            "n_budget": int(len(yva)), "C": best[0], "val_cv_auc": round(best[1], 4),
            "auroc_crd": round(float(a_crd), 4),
            "auroc_lr_budget": round(float(a_lr), 4),
            "auroc_mlp_budget": round(float(a_mlp), 4),
            "d_lr_minus_crd_pp": round(float(a_lr - a_crd) * 100, 2),
            "d_mlp_minus_crd_pp": round(float(a_mlp - a_crd) * 100, 2),
        })
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p0_2_budget_probe.csv", index=False, encoding="utf-8-sig")
    print("\n", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
