"""Audit 15: cluster bootstrap (by target_image_id) for AUROC and paired accuracy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cache import cos, get        # noqa: E402
from fgbench.common import BENCH   # noqa: E402

B = 1000


def main(split="test", seed=0) -> int:
    d, img, txt = get(split)
    d["score"] = -cos(d, img, txt)
    clean = d[d.pair_label == 0].groupby("target_image_id").first()
    rng = np.random.default_rng(seed)
    ids = d.target_image_id.unique()
    out = {}
    for facet in ("Overall", "Entity", "Quantity"):
        sub = d if facet == "Overall" else d[d.facet == facet]
        aucs, accs = [], []
        groups = sub.groupby("target_image_id").indices
        keys = list(groups)
        for _ in range(B):
            pick = rng.choice(keys, len(keys), replace=True)
            idx = np.concatenate([groups[k] for k in pick])
            s = sub.iloc[idx]
            if s.pair_label.nunique() < 2:
                continue
            aucs.append(roc_auc_score(s.pair_label, s.score))
            nz = s[s.pair_label == 1].merge(
                clean[["score"]], left_on="target_image_id", right_index=True,
                suffixes=("", "_clean"))
            if len(nz):
                accs.append(float((nz.score_clean < nz.score).mean()))
        out[facet] = {"auroc": float(roc_auc_score(sub.pair_label, sub.score)),
                      "auroc_ci": [float(np.percentile(aucs, 2.5)),
                                   float(np.percentile(aucs, 97.5))] if aucs else None,
                      "paired_acc": float(accs and np.mean(accs)) if accs else None,
                      "paired_acc_ci": [float(np.percentile(accs, 2.5)),
                                        float(np.percentile(accs, 97.5))] if accs else None,
                      "n_boot": len(aucs)}
    (BENCH / "audit_15_bootstrap.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
