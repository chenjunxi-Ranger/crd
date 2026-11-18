"""Audit 12: can the IMAGE alone reveal the label? (CLIP image embedding + LR)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cache import get            # noqa: E402
from fgbench.common import BENCH  # noqa: E402


def main() -> int:
    dtr, itr, _ = get("train")
    dte, ite, _ = get("test")
    out = {}
    for facet in ("Entity", "Quantity", "All"):
        a = np.ones(len(dtr), bool) if facet == "All" else (dtr.facet == facet).to_numpy()
        b = np.ones(len(dte), bool) if facet == "All" else (dte.facet == facet).to_numpy()
        ytr, yte = dtr.pair_label[a].to_numpy(), dte.pair_label[b].to_numpy()
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(itr[a], ytr)
        out[f"image_only_{facet}"] = float(roc_auc_score(yte, m.predict_proba(ite[b])[:, 1]))
    (BENCH / "audit_12_image_only.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: round(v, 4) for k, v in out.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
