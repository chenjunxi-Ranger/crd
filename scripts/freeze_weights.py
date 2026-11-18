"""Pick ONE global constant set for  s = -cos + l1*(1-coverage) + l2*(1-witness).

Selection is done once, on val, over a small fixed grid, and the same constants
are then used for every benchmark and both protocols (C1).  No per-benchmark
value is ever chosen.  Labels are used only to score the grid (analysis).

Run: python scripts/freeze_weights.py
"""
from __future__ import annotations

import itertools
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.counterfactual import content_words                  # noqa: E402
from src.protocol import (PROJECT, RESULTS_DIR, load_foil_val,  # noqa: E402
                          load_split, write_json)
from src.protocol import BENCH_CSV                             # noqa: E402


def _witness(corpus_caps, caps) -> np.ndarray:
    bi: Counter = Counter()
    for cap in corpus_caps:
        toks = [s.word.lower() for s in content_words(cap)]
        bi.update(zip(toks, toks[1:]))
    nb = math.log1p(sum(bi.values()))

    def one(cap: str) -> float:
        toks = [s.word.lower() for s in content_words(cap)]
        if len(toks) < 2:
            return 0.0
        return float(np.mean([math.log1p(bi.get(p, 0)) for p in zip(toks, toks[1:])])) / nb
    return np.asarray([one(c) for c in caps])


def main() -> int:
    # ---- FOIL val (2,000 pairs, cached token-patch features)
    feat = pd.read_csv(RESULTS_DIR / "gate" / "foil_val_features.csv", encoding="utf-8-sig")
    foil = load_foil_val(2000)
    train = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                        encoding="utf-8-sig")
    names = foil["path"].str.replace("\\", "/").str.split("/").str[-1]
    trn = train["path"].str.replace("\\", "/").str.split("/").str[-1]
    corpus = train[~trn.isin(set(names))]["positive_caption"].astype(str).tolist()
    wp = _witness(corpus, foil["positive"].tolist())
    wn = _witness(corpus, foil["negative"].tolist())

    cos_p, cos_n = feat["cosine_pos"].to_numpy(), feat["cosine_neg"].to_numpy()
    cov_p, cov_n = feat["coverage_soft_pos"].to_numpy(), feat["coverage_soft_neg"].to_numpy()

    # ---- COCO val (2,000 rows, cached token-patch features)
    coco = pd.read_csv(RESULTS_DIR / "gate" / "coco_val_features.csv", encoding="utf-8-sig")
    cdf = load_split("coco", "val", 2000)
    ctrain = load_split("coco", "train")
    wc = _witness(ctrain["caption"].tolist(), cdf["caption"].tolist())

    grid = list(itertools.product([0.0, 0.5, 1.0, 2.0], [0.0, 0.5, 1.0, 2.0]))
    rows = []
    for l1, l2 in grid:
        # orientation: suspicious = low cosine, low coverage, low witness
        s_p = -cos_p + l1 * (1 - cov_p) + l2 * (1 - wp)
        s_n = -cos_n + l1 * (1 - cov_n) + l2 * (1 - wn)
        d = s_n - s_p
        foil_acc = float(np.mean(np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))))
        s_c = -coco["cosine"].to_numpy() + l1 * (1 - coco["coverage_soft"].to_numpy()) \
            + l2 * (1 - wc)
        coco_auc = float(roc_auc_score(coco["label"].to_numpy(), s_c))
        rows.append({"l1_coverage": l1, "l2_witness": l2,
                     "foil_acc": foil_acc, "coco_auc": coco_auc,
                     "min_margin": min(foil_acc - 0.92, coco_auc - 0.96)})
    df = pd.DataFrame(rows).sort_values("min_margin", ascending=False)
    best = df.iloc[0].to_dict()
    write_json(RESULTS_DIR / "gate" / "frozen_constants.json",
               {"selected": best, "grid": rows,
                "selection_rule": "max min(foil_acc-0.92, coco_auc-0.96) on val, one set for all benches"})
    print(df.head(10).to_string(index=False))
    print("best:", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
