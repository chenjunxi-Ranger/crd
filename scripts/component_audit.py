"""Four-component x three-seed audit (cosine / image-only / text-only / full).

Row benchmarks: computed from the saved per-sample cos, rx, rc of the seed audit.
FOIL: taken from foil_decomp.json (same-image pairs, r_I cancels identically).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, THRESHOLDS, write_json   # noqa: E402

SEEDS = (1, 2, 3)
LAM = 0.8


def cluster_ci(score, y, groups, n_boot=1500, seed=0):
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        if y[sel].min() == y[sel].max():
            continue
        vals.append(roc_auc_score(y[sel], score[sel]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    out = {}
    for bench in ("coco", "flickr", "mmimdb"):
        variants = {"cosine": [], "image_only": [], "text_only": [], "full": []}
        cis = {k: [] for k in variants}
        for s in SEEDS:
            z = np.load(RESULTS_DIR / "seed_audit" / f"seed_{s}" / f"val_scores_{bench}.npz",
                        allow_pickle=True)
            y, cos, rx, rc = z["label"], z["cos"], z["rx"], z["rc"]
            g = z["image_id"].astype(str)
            cand = {"cosine": -cos,
                    "image_only": -cos + LAM * rx,
                    "text_only": -cos + LAM * rc,
                    "full": -cos + LAM * (rx + rc)}
            for k, v in cand.items():
                variants[k].append(float(roc_auc_score(y, v)))
                cis[k].append(cluster_ci(v, y, g, seed=s))
        out[bench] = {
            "threshold": THRESHOLDS[bench],
            "mean": {k: float(np.mean(v)) for k, v in variants.items()},
            "sd": {k: float(np.std(v, ddof=1)) for k, v in variants.items()},
            "per_seed": variants,
            "ci_low_min": {k: min(c[0] for c in cis[k]) for k in variants},
        }
        print(bench, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                 for kk, vv in out[bench].items() if kk != "per_seed"},
                                ensure_ascii=False), flush=True)

    foil_path = RESULTS_DIR / "gate" / "foil_decomp.json"
    if foil_path.is_file():
        f = json.loads(foil_path.read_text(encoding="utf-8"))
        out["foil"] = {"threshold": THRESHOLDS["foil"], "note": "seed 1 only",
                       "mean": {k: f[k]["within_pair"] for k in
                                ("cosine", "image_only", "text_only", "full")},
                       "ci95": {k: f[k]["within_ci"] for k in
                                ("cosine", "image_only", "text_only", "full")},
                       "pooled_auc": {k: f[k]["pooled_auc"] for k in
                                      ("cosine", "image_only", "text_only", "full")}}
        print("foil", json.dumps(out["foil"], ensure_ascii=False), flush=True)

    write_json(RESULTS_DIR / "bench" / "component_audit.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
