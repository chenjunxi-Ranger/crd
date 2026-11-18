"""Cycle-clustered paired bootstrap: CRD (seed-avg) vs cosine / CSLS / LEMoN / TRACED.

All score files are row-aligned by construction (same test parquet, same filter);
we assert that before bootstrapping.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "fgbench"


def topk_mean(a, b, k=10, chunk=1024):
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        s = a[i:i + chunk] @ b.T
        kk = min(k, s.shape[1])
        out[i:i + chunk] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []

    for facet in ("Attribute", "Relation"):
        seeds = (0, 1, 2) if facet == "Attribute" else (0,)
        ref = np.load(OUT / f"crd_scores_b16_VG_{facet}_s{seeds[0]}.npz", allow_pickle=True)
        y = ref["y"]; cyc = ref["cycle_id"]; n = len(y)
        crd = []
        for s in seeds:
            z = np.load(OUT / f"crd_scores_b16_VG_{facet}_s{s}.npz", allow_pickle=True)
            assert len(z["y"]) == n and np.array_equal(z["y"], y), "row mismatch"
            crd.append(-z["crd"])                       # similarity direction

        scores = {"cosine": -ref["cos"]}
        for name in ("lemon", "traced"):
            z = np.load(OUT / f"score_{name}_VG_{facet}.npz", allow_pickle=True)
            assert len(z["y"]) == n and np.array_equal(z["y"], y), f"{name} row mismatch"
            scores[name] = z["score"]

        pref = "crdx_b16_test.npz" if facet == "Attribute" else \
            "crd_b16_VG_Relation_test.npz"
        z = np.load(OUT / pref, allow_pickle=True)
        assert len(z["U"]) == n, "tensor row mismatch"
        bank = np.load(OUT / "bank_b16_VG.npz", allow_pickle=True)
        ri = topk_mean(z["U"], bank["txt"]); rc = topk_mean(z["V"], bank["img"])
        cos = (z["U"] * z["V"]).sum(1)
        lam = {"Attribute": 0.2, "Relation": 0.3}[facet]
        scores["csls"] = -(cos - lam * (ri + rc))

        gids = pd.Series(np.arange(n)).groupby(cyc).apply(lambda x: x.to_numpy()).to_dict()
        keys = list(gids)
        base = {k: roc_auc_score(y, v) for k, v in scores.items()}
        crd_auc = float(np.mean([roc_auc_score(y, c) for c in crd]))

        boot = {k: [] for k in scores}
        for _ in range(args.boot):
            pick = rng.choice(len(keys), len(keys), replace=True)
            ii = np.concatenate([gids[keys[t]] for t in pick])
            yy = y[ii]
            if len(np.unique(yy)) < 2:
                continue
            a_crd = np.mean([roc_auc_score(yy, c[ii]) for c in crd])
            for k, v in scores.items():
                boot[k].append((a_crd - roc_auc_score(yy, v[ii])) * 100)

        for k in scores:
            d = np.array(boot[k])
            rows.append({"facet": facet, "baseline": k,
                         "crd_auc": round(crd_auc, 4),
                         "base_auc": round(float(base[k]), 4),
                         "delta_pp": round(float(d.mean()), 2),
                         "ci_lo": round(float(np.percentile(d, 2.5)), 2),
                         "ci_hi": round(float(np.percentile(d, 97.5)), 2),
                         "p_gt0": round(float((d > 0).mean()), 3)})
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pairwise_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
