"""Conditional polarity audit for the text-side bank density r_c.

Row benchmarks: within strata of the CLIP similarity s, measure
  * polarity(s) = AUC of -r_c  (>0.5 means noisy records have HIGHER r_c)
  * delta_auc(s) = CV-AUC([s, r_I, r_c]) - CV-AUC([s, r_I])
FOIL: within strata of delta_s, measure P(delta_r_c > 0) and the flip rate.

A shared, stable direction change across all four would justify a global
interaction term; dataset-specific thresholds would not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402

SEEDS = (1, 2, 3)
LAM = 0.8


def cv_auc(x, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(m, x, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main() -> int:
    out = {"row": {}, "foil": {}}
    for bench in ("coco", "flickr", "mmimdb"):
        per_seed = []
        for s in SEEDS:
            z = np.load(RESULTS_DIR / "seed_audit" / f"seed_{s}" / f"val_scores_{bench}.npz",
                        allow_pickle=True)
            y, cos, rx, rc = z["label"], z["cos"], z["rx"], z["rc"]
            q = np.quantile(cos, [0.25, 0.5, 0.75])
            bins = np.digitize(cos, q)
            rows = []
            for b in range(4):
                sel = bins == b
                if sel.sum() < 200 or y[sel].min() == y[sel].max():
                    continue
                pol = float(roc_auc_score(y[sel], -rc[sel]))
                a1 = cv_auc(np.column_stack([cos[sel], rx[sel]]), y[sel])
                a2 = cv_auc(np.column_stack([cos[sel], rx[sel], rc[sel]]), y[sel])
                rows.append({"bin": b, "n": int(sel.sum()),
                             "s_range": [float(cos[sel].min()), float(cos[sel].max())],
                             "polarity_rc": pol, "delta_auc": a2 - a1})
            per_seed.append(rows)
        out["row"][bench] = per_seed
        print(bench, json.dumps(per_seed[0], ensure_ascii=False), flush=True)

    for s in SEEDS:
        z = np.load(RESULTS_DIR / "gate" / f"foil_signed_seed{s}.npz")
        ds, drc = z["ds"], z["drc"]
        q = np.quantile(np.abs(ds), [0.25, 0.5, 0.75])
        bins = np.digitize(np.abs(ds), q)
        rows = []
        for b in range(4):
            sel = bins == b
            if sel.sum() < 100:
                continue
            rows.append({"bin": b, "n": int(sel.sum()),
                         "abs_ds_range": [float(np.abs(ds[sel]).min()),
                                          float(np.abs(ds[sel]).max())],
                         "p_drc_pos": float((drc[sel] > 0).mean()),
                         "flip_rate": float(np.mean((ds[sel] > 0) &
                                                    (ds[sel] - LAM * drc[sel] < 0)))})
        out["foil"][f"seed{s}"] = rows
        if s == 1:
            print("foil", json.dumps(rows, ensure_ascii=False), flush=True)

    write_json(RESULTS_DIR / "bench" / "conditional_polarity.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
