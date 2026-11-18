"""Per-dataset supervised fusion ceiling (C1 relaxed: per-dataset hyper-parameters).

Uses ONLY cached arrays, no new CLIP forward:
  cos, bank terms (rx/rc/max), GPT-2 prior  <- final_feats2_{bench}.npz
  D4 (cross-depth synchrony)                <- cdus_scores_{bench}.npz
  VES evidence (FOIL only)                  <- ves_pairs_VES-1.npz

Reports 5-fold CV AUROC of a linear and a GBM fusion, i.e. how much information
the union of all candidate signals actually carries for that column.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402


def _cv(x: np.ndarray, y: np.ndarray):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    lin, gbm = [], []
    for tr, te in skf.split(x, y):
        if len(np.unique(y[te])) < 2:
            continue
        p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        p.fit(x[tr], y[tr])
        lin.append(roc_auc_score(y[te], p.predict_proba(x[te])[:, 1]))
        g = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        g.fit(x[tr], y[tr])
        gbm.append(roc_auc_score(y[te], g.predict_proba(x[te])[:, 1]))
    return float(np.mean(lin)), float(np.mean(gbm))


def main() -> int:
    out = {}
    for bench in ("coco", "flickr", "mmimdb"):
        bank = np.load(RESULTS_DIR / "gate" / f"final_feats2_{bench}.npz")
        d = np.load(RESULTS_DIR / "gate" / f"cdus_scores_{bench}.npz")
        n = len(d["s"])
        y = bank["label"][:n]
        x = np.column_stack([
            -bank["cos"][:n], bank["rx"][:n] - bank["cos"][:n],
            bank["rc"][:n] - bank["cos"][:n],
            bank["mxt"][:n] - bank["cos"][:n], bank["mxi"][:n] - bank["cos"][:n],
            -bank["lp"][:n], -d["d4"], -d["d12"],
        ])
        lin, gb = _cv(x, y)
        out[bench] = {"n": int(n), "cv_linear": lin, "cv_gbm": gb,
                      "threshold": 0.96 if bench != "mmimdb" else 0.88,
                      "gap": float((0.96 if bench != "mmimdb" else 0.88) - max(lin, gb))}
        print(bench, json.dumps(out[bench], ensure_ascii=False), flush=True)

    # FOIL: pairs, 2-row format, grouped CV by pair
    bank = np.load(RESULTS_DIR / "gate" / "final_feats2_foil.npz")
    d = np.load(RESULTS_DIR / "gate" / "cdus_scores_foil.npz")
    ves = np.load(RESULTS_DIR / "gate" / "ves_pairs_VES-1.npz")
    n = min(len(d["s_pos"]), len(ves["s_pos"]))
    cols = lambda sfx: np.column_stack([
        -bank["cos" + sfx][:n], bank["rx" + sfx][:n] - bank["cos" + sfx][:n],
        bank["rc" + sfx][:n] - bank["cos" + sfx][:n],
        bank["mxt" + sfx][:n] - bank["cos" + sfx][:n],
        bank["mxi" + sfx][:n] - bank["cos" + sfx][:n],
        -bank["lp" + sfx][:n], -d["d_pos" if sfx == "_pos" else "d_neg"][:n],
        -ves["e_pos" if sfx == "_pos" else "e_neg"][:n],
    ])
    xp, xn = cols("_pos"), cols("_neg")
    x = np.vstack([xp, xn])
    y = np.r_[np.zeros(n), np.ones(n)]
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(n), 5)
    res = {"linear": [], "gbm": []}
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        mtr = np.zeros(2 * n, bool); mte = np.zeros(2 * n, bool)
        mtr[tr] = True; mtr[n + tr] = True
        mte[te] = True; mte[n + te] = True
        for name, model in (
            ("linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
            ("gbm", HistGradientBoostingClassifier(max_iter=200, random_state=0)),
        ):
            model.fit(x[mtr], y[mtr])
            if name == "linear":
                sp, sn = model.decision_function(xp[te]), model.decision_function(xn[te])
            else:
                sp = model.predict_proba(xp[te])[:, 1]
                sn = model.predict_proba(xn[te])[:, 1]
            res[name].append(float(np.mean(np.where(sn > sp, 1.0,
                                                    np.where(sn < sp, 0.0, 0.5)))))
    out["foil"] = {"n_pairs": int(n),
                   "cv_linear": float(np.mean(res["linear"])),
                   "cv_gbm": float(np.mean(res["gbm"])),
                   "threshold": 0.92,
                   "gap": float(0.92 - max(np.mean(res["linear"]), np.mean(res["gbm"])))}
    print("foil", json.dumps(out["foil"], ensure_ascii=False), flush=True)

    write_json(RESULTS_DIR / "gate" / "fusion_ceiling.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
