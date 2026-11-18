"""Summarise the three-seed audit: seed mean/SD + image-clustered bootstrap CI."""
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
N_BOOT = 2000


def cluster_ci(score, y, groups, n_boot=N_BOOT, seed=0):
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    out = {}
    for bench in ("coco", "flickr", "mmimdb"):
        rows = []
        for s in SEEDS:
            z = np.load(RESULTS_DIR / "seed_audit" / f"seed_{s}" / f"val_scores_{bench}.npz",
                        allow_pickle=True)
            rows.append((z["score"], z["label"], z["image_id"].astype(str)))
        aucs = [float(roc_auc_score(y, sc)) for sc, y, _ in rows]
        cis = [cluster_ci(sc, y, g, seed=s) for s, (sc, y, g) in zip(SEEDS, rows)]
        out[bench] = {
            "per_seed": aucs, "mean": float(np.mean(aucs)),
            "sd": float(np.std(aucs, ddof=1)),
            "ci95_per_seed": [list(c) for c in cis],
            "ci_low_min": min(c[0] for c in cis),
            "threshold": THRESHOLDS[bench],
            "n_seed_pass": int(sum(a >= THRESHOLDS[bench] for a in aucs)),
            "n_ci_pass": int(sum(c[0] >= THRESHOLDS[bench] for c in cis)),
        }
        print(bench, json.dumps(out[bench], ensure_ascii=False), flush=True)

    accs, gs = [], None
    for s in SEEDS:
        z = np.load(RESULTS_DIR / "seed_audit" / f"seed_{s}" / "val_scores_foil.npz",
                    allow_pickle=True)
        ok = (z["a_neg"] > z["a_pos"]).astype(float)
        accs.append(float(ok.mean()))
        gs = z["image_id"].astype(str)
        rng = np.random.default_rng(s)
        uniq = np.unique(gs)
        by_g = {g: np.where(gs == g)[0] for g in uniq}
        vals = []
        for _ in range(N_BOOT):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([by_g[g] for g in pick])
            vals.append(ok[idx].mean())
        if s == SEEDS[0]:
            cis_f = []
        cis_f.append([float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))])
    out["foil"] = {
        "per_seed": accs, "mean": float(np.mean(accs)),
        "sd": float(np.std(accs, ddof=1)),
        "ci95_per_seed": cis_f,
        "ci_low_min": min(c[0] for c in cis_f),
        "threshold": THRESHOLDS["foil"],
        "n_seed_pass": int(sum(a >= THRESHOLDS["foil"] for a in accs)),
        "n_ci_pass": int(sum(c[0] >= THRESHOLDS["foil"] for c in cis_f)),
    }
    print("foil", json.dumps(out["foil"], ensure_ascii=False), flush=True)
    write_json(RESULTS_DIR / "bench" / "seed_summary.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
