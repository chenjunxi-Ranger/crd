"""Style-only vs Text-only control.

Both models are fit on the official train split (clean vs noisy) and evaluated on
the annotated audit rows (clean vs incompat / clean vs compat), i.e. never trained
on the 40 audit captions.

    Style-only : length, punctuation, casing, function-word ratio, lexical diversity,
                 corpus frequency statistics  (no semantic content)
    Text-only  : TF-IDF (1-2 gram) + logistic regression (pure lexical content)
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, load_split, write_json   # noqa: E402

FUNC = set("a an the of in on at to for with and or by from as is are was were be "
           "this that these those it its his her their our your my he she they we i".split())


def read_any(p: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"cannot decode {p}")


def style_feats(texts, freq):
    rows = []
    for t in texts:
        toks = re.findall(r"[A-Za-z']+", str(t))
        low = [w.lower() for w in toks]
        n = max(len(toks), 1)
        rows.append([
            len(str(t)), n, len(set(low)) / n, np.mean([len(w) for w in toks]) if toks else 0,
            sum(w in FUNC for w in low) / n,
            str(t).count(","), str(t).count("."), str(t).count("!"),
            sum(c.isupper() for c in str(t)) / max(len(str(t)), 1),
            sum(c.isdigit() for c in str(t)),
            np.mean([freq.get(w, 0) for w in low]) if toks else 0,
            sum(freq.get(w, 0) <= 1 for w in low) / n,
        ])
    return np.asarray(rows, dtype=float)


def main() -> int:
    d = RESULTS_DIR / "audit" / "formal"
    a = read_any(d / "batch01_filled.csv").merge(read_any(d / "batch01_key.csv"), on="anon_id")
    a["match"] = a["match"].fillna("").astype(str).str.strip().str.lower()
    a = a[a["match"].isin(["match", "mismatch"])]

    out = {}
    for bench in ("coco", "flickr"):
        tr = load_split(bench, "train", 20000)
        caps_tr = tr["caption"].astype(str).tolist()
        y_tr = tr["label"].to_numpy()
        freq = Counter()
        for c in caps_tr:
            freq.update(w.lower() for w in re.findall(r"[A-Za-z']+", c))

        val = load_split(bench, "val")
        yv = val["label"].to_numpy()
        sub = a[(a["bench"] == bench) & (a["label"] == 1)]
        inc = sorted(sub.loc[sub["match"] == "mismatch", "sample_id"].astype(int))
        com = sorted(sub.loc[sub["match"] == "match", "sample_id"].astype(int))
        clean = list(np.where(yv == 0)[0])

        X_style_tr = style_feats(caps_tr, freq)
        m_style = make_pipeline(LogisticRegression(max_iter=2000))
        m_style.fit(X_style_tr, y_tr)
        m_text = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000),
                               LogisticRegression(max_iter=2000))
        m_text.fit(caps_tr, y_tr)

        rec = {}
        for name, ids in (("incompat", inc), ("compat", com)):
            sel = clean + ids
            yy = np.r_[np.zeros(len(clean)), np.ones(len(ids))]
            caps = [val["caption"].iloc[i] for i in sel]
            ps = m_style.predict_proba(style_feats(caps, freq))[:, 1]
            pt = m_text.predict_proba(caps)[:, 1]
            rec[f"style_auc_{name}"] = float(roc_auc_score(yy, ps))
            rec[f"text_auc_{name}"] = float(roc_auc_score(yy, pt))
        rec["n_clean"] = len(clean)
        rec["n_incompat"] = len(inc)
        rec["n_compat"] = len(com)
        out[bench] = rec
        print(bench, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()},
              flush=True)
    write_json(RESULTS_DIR / "bench" / "style_control.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
