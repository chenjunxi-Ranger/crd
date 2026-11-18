"""Gate: dataset-internal attestation ("witness") on FOIL pairs.

Corpus = FOIL official **train** captions only (187,788 rows, images of the frozen
validation split removed), i.e. a dataset-internal corpus, not a pretraining
corpus (CONSTRAINTS C4 forbids the latter only).

Three statistics, all label-free:
* unigram witness   : mean log(1+count) over content words
* minimum witness   : the rarest content word decides
* bigram witness    : mean log(1+count) over adjacent content-word bigrams

Analysis only.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.counterfactual import content_words                 # noqa: E402
from src.protocol import THRESHOLDS, load_foil_val, write_json  # noqa: E402
from src.protocol import RESULTS_DIR, PROJECT                 # noqa: E402


def _tokens(text: str) -> list[str]:
    return [s.word.lower() for s in content_words(text)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    train = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                        encoding="utf-8-sig")
    val = load_foil_val()
    val_names = set(val["path"].str.replace("\\", "/").str.split("/").str[-1])
    names = train["path"].str.replace("\\", "/").str.split("/").str[-1]
    corpus = train[~names.isin(val_names)]

    uni: Counter = Counter()
    bi: Counter = Counter()
    for cap in corpus["positive_caption"].astype(str):
        toks = _tokens(cap)
        uni.update(toks)
        bi.update(zip(toks, toks[1:]))
    total_uni, total_bi = sum(uni.values()), sum(bi.values())

    def score(cap: str) -> tuple[float, float, float]:
        toks = _tokens(cap)
        if not toks:
            return 0.0, 0.0, 0.0
        u = [math.log1p(uni.get(t, 0)) for t in toks]
        b = [math.log1p(bi.get(p, 0)) for p in zip(toks, toks[1:])] or [0.0]
        # normalise by corpus size so values are comparable across folds
        nu = math.log1p(total_uni)
        nb = math.log1p(total_bi)
        return (float(np.mean(u)) / nu, float(np.min(u)) / nu, float(np.mean(b)) / nb)

    df = val.head(args.limit)
    pos = np.asarray([score(c) for c in df["positive"]])
    neg = np.asarray([score(c) for c in df["negative"]])
    names = ["unigram_mean", "unigram_min", "bigram_mean"]
    out = {"bench": "foil", "n_pairs": int(len(df)),
           "corpus_rows": int(len(corpus)), "vocab": int(len(uni)),
           "threshold": THRESHOLDS["foil"], "single": {}}
    for i, name in enumerate(names):
        d = neg[:, i] - pos[:, i]
        out["single"][name] = float(np.mean(np.where(d < 0, 1.0,
                                                     np.where(d > 0, 0.0, 0.5))))
    out["best_single"] = max(out["single"].values())
    out["gap"] = float(THRESHOLDS["foil"] - out["best_single"])
    write_json(RESULTS_DIR / "gate" / "foil_witness.json", out)
    print(pd.Series(out["single"]).sort_values(ascending=False).to_string())
    print({k: v for k, v in out.items() if k != "single"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
