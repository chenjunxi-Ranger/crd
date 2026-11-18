"""Paired bootstrap for the hard+/easy- cell + cross-dataset difference test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, load_split, write_json   # noqa: E402

B = 1000


def cell_auc(s, y, g, cell):
    gp, hn = cell
    mp = (y == 1) & (g == gp)
    mn = (y == 0) & (g == hn)
    if mp.sum() < 5 or mn.sum() < 5:
        return None
    return float(roc_auc_score(y[mp | mn], s[mp | mn]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="coco,flickr,mmimdb")
    args = ap.parse_args()

    out, samples = {}, {}
    for bench in args.benches.split(","):
        p = RESULTS_DIR / "gate" / f"traced_failure_{bench}.csv"
        d = pd.read_csv(p, encoding="utf-8-sig")
        if "group" not in d:
            d = d.rename(columns={[c for c in d.columns if c.startswith("group")][0]: "group"})
        d["group"] = d["group"].fillna("both_low_burden")
        val = load_split(bench, "val", len(d))
        d["image_id"] = val["image_id"].astype(str).to_numpy()

        y = d["label"].to_numpy()
        g = np.where(d["group"] == "persistent_hard", "hard", "easy")
        cos, tr = d["s0"].to_numpy(), d["p_trace"].to_numpy()

        uq = np.unique(d["image_id"].to_numpy())
        by = {k: np.where(d["image_id"].to_numpy() == k)[0] for k in uq}
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(B):
            pick = rng.choice(uq, len(uq), replace=True)
            idx = np.concatenate([by[k] for k in pick])
            a = cell_auc(cos[idx], y[idx], g[idx], ("hard", "easy"))
            b = cell_auc(tr[idx], y[idx], g[idx], ("hard", "easy"))
            if a is None or b is None:
                continue
            diffs.append(b - a)
        point_c = cell_auc(cos, y, g, ("hard", "easy"))
        point_t = cell_auc(tr, y, g, ("hard", "easy"))
        out[bench] = {"n_cell_noisy": int(((y == 1) & (g == "hard")).sum()),
                      "n_cell_clean": int(((y == 0) & (g == "easy")).sum()),
                      "auc_cos": point_c, "auc_traced": point_t,
                      "delta": None if (point_c is None or point_t is None) else point_t - point_c,
                      "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
                      if diffs else None,
                      "n_boot": len(diffs)}
        samples[bench] = np.array(diffs)
        print(bench, out[bench], flush=True)

    cross = {}
    keys = [k for k in samples if len(samples[k])]
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i >= j:
                continue
            a, b = samples[keys[i]], samples[keys[j]]
            n = min(len(a), len(b))
            dd = b[:n] - a[:n]
            cross[f"{keys[j]}-{keys[i]}"] = {
                "mean": float(dd.mean()),
                "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
            print("cross", keys[j], "-", keys[i], cross[f"{keys[j]}-{keys[i]}"], flush=True)
    out["cross_diff"] = cross
    write_json(RESULTS_DIR / "gate" / "cell_bootstrap.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
