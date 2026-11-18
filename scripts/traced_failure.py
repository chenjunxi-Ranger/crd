"""TRACED failure map: per-sample trajectory features joined with A-group labels.

Answers: on the samples where A keeps failing (persistent_hard), does the
trajectory evidence help at all?  And does it overlap with what A already fixes?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from traced_repro import trajectory                       # noqa: E402
from src.protocol import RESULTS_DIR, load_split, write_json   # noqa: E402
from src.scorer import build_scorer                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    sc = build_scorer("clip")
    tr = load_split(args.bench, "train", args.train)
    va = load_split(args.bench, "val", args.val)

    def feats(df):
        return np.array([trajectory(sc, df["path"].iloc[i], str(df["caption"].iloc[i]), args.steps)
                         for i in range(len(df))])

    Xtr = feats(tr)
    Xva = feats(va)
    ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()

    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
    p_trace = m.predict_proba(Xva)[:, 1]
    s0 = -Xva[:, 0]                      # trajectory feature 0 is the final score

    groups = pd.read_csv(RESULTS_DIR / "audit" / f"{args.bench}_all_groups.csv",
                         encoding="utf-8-sig", usecols=["sample_id", "group", "label"])
    df = pd.DataFrame({"sample_id": np.arange(len(va)), "label": yva,
                       "s0": s0, "p_trace": p_trace})
    df = df.merge(groups, on="sample_id", how="left", suffixes=("", "_g"))

    out = {"bench": args.bench, "n_val": int(len(va)),
           "auc_cos": float(roc_auc_score(yva, s0)),
           "auc_traced": float(roc_auc_score(yva, p_trace)),
           "delta": float(roc_auc_score(yva, p_trace) - roc_auc_score(yva, s0))}
    by = {}
    for g, sub in df.groupby("group"):
        if len(sub) < 20 or sub["label"].nunique() < 2:
            by[g] = {"n": int(len(sub))}
            continue
        yg = sub["label"].to_numpy()
        by[g] = {"n": int(len(sub)),
                 "auc_cos": float(roc_auc_score(yg, sub["s0"])),
                 "auc_traced": float(roc_auc_score(yg, sub["p_trace"])),
                 "n_noisy": int((yg == 1).sum())}
    out["by_group"] = by

    # rescue / hurt vs cosine, within group
    df["cos_correct"] = (df["s0"] > 0) == (df["label"] == 1)
    df["trace_correct"] = (df["p_trace"] > 0.5) == (df["label"] == 1)
    rs = {}
    for g, sub in df.groupby("group"):
        rs[g] = {"rescue": int(((~sub.cos_correct) & sub.trace_correct).sum()),
                 "hurt": int((sub.cos_correct & (~sub.trace_correct)).sum())}
    out["rescue_hurt"] = rs
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"traced_failure_{args.bench}.json", out)
    df.to_csv(RESULTS_DIR / "gate" / f"traced_failure_{args.bench}.csv", index=False,
              encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
