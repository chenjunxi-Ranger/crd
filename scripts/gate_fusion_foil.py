"""Supervised fusion upper bound on FOIL (2-row format, pair-aware CV).

Each pair contributes two rows (positive -> 0, negative -> 1).  A classifier is
fitted on 80% of *pairs* and evaluated by the sign of score(neg) - score(pos) on
the held-out pairs, so no pair leaks across the split.

Features: cosine, bigram witness, token-patch support/coverage (from the Gate CSV).
Analysis only.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.counterfactual import content_words          # noqa: E402
from src.protocol import PROJECT, RESULTS_DIR, load_foil_val, write_json  # noqa: E402


def main() -> int:
    train = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                        encoding="utf-8-sig")
    val = load_foil_val(2000)
    names = val["path"].str.replace("\\", "/").str.split("/").str[-1]
    tr_names = train["path"].str.replace("\\", "/").str.split("/").str[-1]
    corpus = train[~tr_names.isin(set(names))]
    bi: Counter = Counter()
    for cap in corpus["positive_caption"].astype(str):
        toks = [s.word.lower() for s in content_words(cap)]
        bi.update(zip(toks, toks[1:]))
    nb = math.log1p(sum(bi.values()))

    def wit(cap: str) -> float:
        toks = [s.word.lower() for s in content_words(cap)]
        if len(toks) < 2:
            return 0.0
        return float(np.mean([math.log1p(bi.get(p, 0)) for p in zip(toks, toks[1:])])) / nb

    wp = np.asarray([wit(c) for c in val["positive"]])
    wn = np.asarray([wit(c) for c in val["negative"]])

    feat = pd.read_csv(RESULTS_DIR / "gate" / "foil_val_features.csv", encoding="utf-8-sig")
    cols = ["cosine", "support_mean", "coverage_soft"]
    xp = np.column_stack([feat[f"{c}_pos"].to_numpy() for c in cols] + [wp])
    xn = np.column_stack([feat[f"{c}_neg"].to_numpy() for c in cols] + [wn])

    x = np.vstack([xp, xn])
    y = np.r_[np.zeros(len(xp)), np.ones(len(xn))]
    n = len(xp)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    folds = np.array_split(order, 5)
    accs = {"linear": [], "gbm": []}
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        mask_tr = np.r_[np.zeros(n, bool), np.zeros(n, bool)]
        mask_te = np.r_[np.zeros(n, bool), np.zeros(n, bool)]
        mask_tr[tr] = True
        mask_tr[n + tr] = True
        mask_te[te] = True
        mask_te[n + te] = True
        for name, model in (
            ("linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
            ("gbm", HistGradientBoostingClassifier(max_iter=200, random_state=0)),
        ):
            model.fit(x[mask_tr], y[mask_tr])
            if name == "linear":
                sp, sn = model.decision_function(xp[te]), model.decision_function(xn[te])
            else:
                sp = model.predict_proba(xp[te])[:, 1]
                sn = model.predict_proba(xn[te])[:, 1]
            accs[name].append(float(np.mean(np.where(sn > sp, 1.0,
                                                     np.where(sn < sp, 0.0, 0.5)))))
    out = {"bench": "foil", "n_pairs": int(n),
           "features": cols + ["bigram_witness"],
           "linear_mean": float(np.mean(accs["linear"])),
           "linear_sd": float(np.std(accs["linear"])),
           "gbm_mean": float(np.mean(accs["gbm"])),
           "gbm_sd": float(np.std(accs["gbm"]))}
    write_json(RESULTS_DIR / "gate" / "foil_fusion.json", out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
