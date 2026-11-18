"""Audit 11: can the caption ALONE reveal the label? (CLIP text embedding + TF-IDF)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cache import get          # noqa: E402
from fgbench.common import BENCH  # noqa: E402


def fit_eval(Xtr, ytr, Xte, yte):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p))


def main() -> int:
    dtr, _, ttr = get("train")
    dte, _, tte = get("test")
    out = {}
    for facet in ("Entity", "Quantity", "All"):
        def sel(d):
            if facet == "All":
                return np.ones(len(d), bool)
            return (d.facet == facet).to_numpy()
        a, b = sel(dtr), sel(dte)
        if len(np.unique(dtr.pair_label[a])) < 2 or len(np.unique(dte.pair_label[b])) < 2:
            continue
        out[f"clip_text_{facet}"] = fit_eval(
            ttr[a], dtr.pair_label[a].to_numpy(), tte[b], dte.pair_label[b].to_numpy())

        v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000)
        Xtr = v.fit_transform(dtr.caption.astype(str)[a])
        Xte = v.transform(dte.caption.astype(str)[b])
        m = LogisticRegression(max_iter=3000).fit(Xtr, dtr.pair_label[a].to_numpy())
        out[f"tfidf_{facet}"] = float(roc_auc_score(
            dte.pair_label[b].to_numpy(), m.predict_proba(Xte)[:, 1]))
    (BENCH / "audit_11_text_only.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: round(v, 4) for k, v in out.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
