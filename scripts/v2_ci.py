"""Bootstrap CI for v2.0 on the four val columns (cached features, no GPU).

Reported per column: point estimate, 95% CI, and the CI computed with three
independent bootstrap seeds so the interval itself is shown to be stable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants_v2 import constants_dict, constants_hash   # noqa: E402
from src.protocol import RESULTS_DIR, THRESHOLDS, write_json  # noqa: E402


def _score(f, tag=None):
    sfx = "" if tag is None else f"_{tag}"
    cos, rx, rc = f["cos" + sfx], f["rx" + sfx], f["rc" + sfx]
    lp = f["lp" + sfx]
    return -cos + 6.0 * ((rx + rc) / 2 - cos) + 0.05 * (-lp)


def _auc_ci(y, s, n=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    point = float(roc_auc_score(y, s))
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(roc_auc_score(y[b], s[b]))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    rows = []
    for bench in ("coco", "flickr", "mmimdb"):
        with np.load(RESULTS_DIR / "gate" / f"final_feats2_{bench}.npz") as z:
            f = {k: z[k] for k in z.files}
        s = _score(f)
        y = f["label"]
        point, lo, hi = _auc_ci(y, s, seed=0)
        seeds = [_auc_ci(y, s, seed=k)[1:3] for k in (1, 2, 3)]
        rows.append({"bench": bench, "metric": "AUROC", "value": point,
                     "ci_low": lo, "ci_high": hi,
                     "ci_by_seed": seeds,
                     "threshold": THRESHOLDS[bench],
                     "gap": float(THRESHOLDS[bench] - point),
                     "pass_point": bool(point >= THRESHOLDS[bench]),
                     "pass_ci_low": bool(lo >= THRESHOLDS[bench])})

    with np.load(RESULTS_DIR / "gate" / "final_feats2_foil.npz") as z:
        f = {k: z[k] for k in z.files}
    sp, sn = _score(f, "pos"), _score(f, "neg")
    d = sn - sp
    correct = np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))
    rng = np.random.default_rng(0)
    point = float(correct.mean())
    vals = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(1000)]
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    seeds = []
    for k in (1, 2, 3):
        r2 = np.random.default_rng(k)
        v = [correct[r2.integers(0, len(correct), len(correct))].mean() for _ in range(1000)]
        seeds.append([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
    rows.append({"bench": "foil", "metric": "within-pair acc", "value": point,
                 "ci_low": lo, "ci_high": hi, "ci_by_seed": seeds,
                 "threshold": THRESHOLDS["foil"],
                 "gap": float(THRESHOLDS["foil"] - point),
                 "pass_point": bool(point >= THRESHOLDS["foil"]),
                 "pass_ci_low": bool(lo >= THRESHOLDS["foil"])})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "bench" / "v2_ci.csv", index=False, encoding="utf-8-sig")
    payload = {"constants": constants_dict(), "constants_hash": constants_hash(),
               "rows": rows}
    write_json(RESULTS_DIR / "bench" / "v2_ci.json", payload)
    print(df[["bench", "value", "ci_low", "ci_high", "threshold", "gap",
              "pass_point", "pass_ci_low"]].to_string(index=False))
    print("hash:", payload["constants_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
