"""Paired cycle-clustered bootstrap: row-level target vs cycle-mean target."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
SEEDS = (0, 1, 2)


def load(target, ds="b16", dataset="VG", facet="Attribute"):
    ss = []
    for s in SEEDS:
        p = OUT / f"crd_scores_{ds}_{dataset}_{facet}_both_{target}_s{s}.npz"
        z = np.load(p, allow_pickle=True)
        ss.append(z["crd"])
        y, cid = z["y"], z["cycle_id"]
    return np.mean(ss, axis=0), y, cid


def main() -> int:
    boot = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    sc_c, y, cid = load("cycle")
    sc_r, _, _ = load("row")
    a_c, a_r = roc_auc_score(y, -sc_c), roc_auc_score(y, -sc_r)
    print(f"cycle {a_c:.5f}  row {a_r:.5f}  delta {(a_r-a_c)*100:+.2f} pp", flush=True)

    per_seed = []
    for s in SEEDS:
        p1 = np.load(OUT / f"crd_scores_b16_VG_Attribute_both_cycle_s{s}.npz", allow_pickle=True)
        p2 = np.load(OUT / f"crd_scores_b16_VG_Attribute_both_row_s{s}.npz", allow_pickle=True)
        d = (roc_auc_score(y, -p2["crd"]) - roc_auc_score(y, -p1["crd"])) * 100
        per_seed.append(round(float(d), 3))
    print("per-seed delta pp", per_seed, flush=True)

    idx = pd.Series(np.arange(len(y))).groupby(cid).apply(lambda x: x.to_numpy()).to_dict()
    keys = list(idx)
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(boot):
        pk = rng.choice(len(keys), len(keys), replace=True)
        ii = np.concatenate([idx[keys[t]] for t in pk])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        ds.append((roc_auc_score(yy, -sc_r[ii]) - roc_auc_score(yy, -sc_c[ii])) * 100)
    ds = np.array(ds)
    res = {"facet": "VG_Attribute", "auc_cycle": round(float(a_c), 4),
           "auc_row": round(float(a_r), 4),
           "delta_pp": round(float((a_r - a_c) * 100), 2),
           "ci": [round(float(np.percentile(ds, 2.5)), 2),
                  round(float(np.percentile(ds, 97.5)), 2)],
           "p_gt0": round(float((ds > 0).mean()), 4),
           "per_seed_delta_pp": per_seed,
           "boot": int(len(ds))}
    print(res, flush=True)
    jdump(res, OUT / "target_bootstrap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
