"""P0-D: label-budget-matched probe on the Action facet.

Action's official validation split has two cycles, so CRD selects lambda on 80
held-out training cycles (320 rows). The full-budget probe in Table II uses the
full training split (1,276 rows). This script trains the same LR/MLP on the
80-cycle budget and evaluates on the untouched 166-cycle test split.

Output: results/fgbench/p0_baselines/p0_d_action_budget.csv
Run:    python .../p0_d_action_budget.py
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
from run_baselines import action_splits  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "p0_baselines"
C_GRID = (0.01, 0.1, 1.0, 10.0)
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


def fit_mlp(X, y, seed=0, steps=400):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X, device=DEV, dtype=torch.float32)
    yt = torch.tensor(y, device=DEV, dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        idx = torch.tensor(rng.integers(0, len(Xt), min(128, len(Xt))), device=DEV)
        loss = lossf(model(Xt[idx]), yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def main() -> int:
    (dtr, Utr, Vtr), (dva, Uva, Vva), (dte, Ute, Vte) = action_splits()
    Xva, Xte = feats(Uva, Vva), feats(Ute, Vte)
    yva = dva.pair_label.to_numpy().astype(int)   # 1 = noisy
    yte = dte.pair_label.to_numpy()

    zs = [np.load(CACHE / f"crd_action_scores_s{s}.npz", allow_pickle=True)
          for s in (0, 1, 2)]
    crd = np.mean([-z["crd"] for z in zs], axis=0)
    a_crd = roc_auc_score(yte, crd)

    sc = StandardScaler().fit(Xva)
    Zva, Zte = sc.transform(Xva), sc.transform(Xte)
    # LR: predict clean (1 - noisy), C by 5-fold CV on the 320 rows
    yclean = 1 - yva
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best = None
    for C in C_GRID:
        aucs = []
        for tr, va in cv.split(Zva, yclean):
            clf = LogisticRegression(C=C, max_iter=3000).fit(Zva[tr], yclean[tr])
            aucs.append(roc_auc_score(yclean[va], clf.decision_function(Zva[va])))
        if best is None or np.mean(aucs) > best[1]:
            best = (C, float(np.mean(aucs)))
    clf = LogisticRegression(C=best[0], max_iter=3000).fit(Zva, yclean)
    a_lr = roc_auc_score(yte, -clf.decision_function(Zte))

    mlp_scores = []
    for s in (0, 1):
        m = fit_mlp(Zva, yclean, seed=s)
        with torch.no_grad():
            mlp_scores.append(
                m(torch.tensor(Zte, device=DEV, dtype=torch.float32)).cpu().numpy())
    mlp = np.mean(mlp_scores, axis=0)
    a_mlp = roc_auc_score(yte, -mlp)

    full = np.load(OUT / "probe_scores_VG_Action.npz", allow_pickle=True)
    if not np.array_equal(full["cycle_id"], dte.cycle_id.values):
        raise SystemExit("probe_scores_VG_Action.npz is not row-aligned")
    # stored probe scores are anomaly scores (higher = noisier)
    a_full_lr = roc_auc_score(yte, full["lr_uvdot"])
    a_full_mlp = roc_auc_score(yte, full["mlp_uvdot"])

    rec = {
        "facet": "Action", "n_train_full": int(len(dtr)), "n_budget": int(len(dva)),
        "n_test": int(len(dte)), "C": best[0], "val_cv_auc": round(best[1], 4),
        "auroc_crd": round(float(a_crd), 4),
        "auroc_lr_budget": round(float(a_lr), 4),
        "auroc_mlp_budget": round(float(a_mlp), 4),
        "auroc_lr_full": round(float(a_full_lr), 4),
        "auroc_mlp_full": round(float(a_full_mlp), 4),
        "d_lr_budget_minus_crd": round(float(a_lr - a_crd) * 100, 2),
        "d_mlp_budget_minus_crd": round(float(a_mlp - a_crd) * 100, 2),
    }
    pd.DataFrame([rec]).to_csv(OUT / "p0_d_action_budget.csv", index=False,
                               encoding="utf-8-sig")
    print(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
