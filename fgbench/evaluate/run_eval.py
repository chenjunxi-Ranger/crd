"""Evaluate any scorer on the benchmark: overall + per-facet AUROC/AUPRC/TPR@FPR."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import BENCH, PROJ       # noqa: E402
from src.protocol import RESULTS_DIR          # noqa: E402
from src.scorer import build_scorer           # noqa: E402


def tpr_at_fpr(y, s, fpr=0.01):
    neg = np.sort(s[y == 0])
    thr = np.quantile(neg, 1 - fpr) if len(neg) else np.inf
    return float((s[y == 1] >= thr).mean()) if (y == 1).any() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--scorer", default="clip", choices=["clip", "fg"])
    ap.add_argument("--img-root", default=os.environ.get(
        "COCO_IMG", str(PROJ / "数据集" / "MS-COCO" / "val2014")))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d = pd.read_parquet(BENCH / f"{args.split}.parquet")
    if args.limit:
        d = d.groupby(["facet", "pair_label"], group_keys=False).head(args.limit // 4)
    d["path"] = [str(Path(args.img_root) / f) for f in d.target_file]

    sc = build_scorer(args.scorer)
    paths, caps = d.path.tolist(), d.caption.astype(str).tolist()
    scores = np.empty(len(d))
    B = 128
    for i in range(0, len(d), B):
        scores[i:i + B] = sc.score_many(paths[i:i + B], caps[i:i + B])
    d["score"] = -scores                      # higher = more anomalous
    y = d.pair_label.to_numpy()

    rows = [{"facet": "Overall", "n_noisy": int((y == 1).sum()),
             "n_clean": int((y == 0).sum()),
             "auroc": float(roc_auc_score(y, d.score)),
             "auprc": float(average_precision_score(y, d.score)),
             "tpr@1%fpr": tpr_at_fpr(y, d.score.to_numpy(), 0.01)}]
    per = []
    for f in ("Entity", "Quantity"):
        sub = d[d.facet == f]
        if sub.pair_label.nunique() < 2:
            continue
        yy = sub.pair_label.to_numpy()
        a = float(roc_auc_score(yy, sub.score))
        per.append(a)
        rows.append({"facet": f, "n_noisy": int((yy == 1).sum()),
                     "n_clean": int((yy == 0).sum()), "auroc": a,
                     "auprc": float(average_precision_score(yy, sub.score)),
                     "tpr@1%fpr": tpr_at_fpr(yy, sub.score.to_numpy(), 0.01)})
    rows.append({"facet": "Macro(facets)", "n_noisy": None, "n_clean": None,
                 "auroc": float(np.mean(per)) if per else None,
                 "auprc": None, "tpr@1%fpr": None})

    out = pd.DataFrame(rows)
    out.insert(0, "scorer", args.scorer)
    out.insert(1, "split", args.split)
    p = RESULTS_DIR / "fgbench" / f"eval_{args.scorer}_{args.split}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
