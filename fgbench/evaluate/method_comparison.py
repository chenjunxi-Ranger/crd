"""Final same-task comparison on FGBench: cosine / CSLS / LEMoN-style /
TRACED-style / GMM / CRD(row).  All methods scored on identical rows.

Convention: every method is converted to an ANOMALY score (higher = noisier).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fgbench"
FACETS = [("VG", "Attribute"), ("VG", "Relation"), ("COCO", "Quantity")]
LAM = {"Attribute": 0.2, "Relation": 0.3, "Quantity": 0.6, "Entity": 0.5}
SEEDS = (0, 1, 2)


def topk_mean(a, b, k=10, chunk=1024):
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        s = a[i:i + chunk] @ b.T
        kk = min(k, s.shape[1])
        out[i:i + chunk] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def acc2afc(anom, y, cid):
    df = pd.DataFrame({"c": cid, "y": y, "s": anom})
    acc = [g[g.y == 0].s.sum() < g[g.y == 1].s.sum()
           for _, g in df.groupby("c") if len(g) == 4]
    return float(np.mean(acc)) if acc else float("nan")


def boot_delta(a_anom, b_anom, y, cid, n=1000, seed=0):
    """bootstrap AUC(b) - AUC(a), resampling cycles."""
    rng = np.random.default_rng(seed)
    idx = pd.Series(np.arange(len(y))).groupby(cid).apply(lambda x: x.to_numpy()).to_dict()
    keys = list(idx)
    ds = []
    for _ in range(n):
        pk = rng.choice(len(keys), len(keys), replace=True)
        ii = np.concatenate([idx[keys[t]] for t in pk])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        ds.append(roc_auc_score(yy, b_anom[ii]) - roc_auc_score(yy, a_anom[ii]))
    ds = np.array(ds) * 100
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def main() -> int:
    rows = []
    for ds, facet in FACETS:
        # reference ordering from the CRD score files (identical manifest order)
        crd = []
        for s in SEEDS:
            z = np.load(OUT / f"crd_scores_b16_{ds}_{facet}_both_row_s{s}.npz",
                        allow_pickle=True)
            crd.append(z["crd"])
            y, cid = z["y"], z["cycle_id"]
        crd = np.mean(crd, axis=0)
        n = len(y)

        z = np.load(OUT / f"crd_b16_{ds}_{facet}_test.npz", allow_pickle=True)
        meta = pd.read_parquet(OUT / f"crd_b16_{ds}_{facet}_test.meta.parquet")
        assert len(meta) == n, (len(meta), n)
        assert (meta.pair_label.to_numpy() == y).all()
        cos = (z["U"] * z["V"]).sum(1)
        bank = np.load(OUT / f"bank_b16_{ds}.npz", allow_pickle=True)
        ri = topk_mean(z["U"], bank["txt"]); rc = topk_mean(z["V"], bank["img"])
        csls = cos - LAM[facet] * (ri + rc)

        methods = {"cosine": -cos, "csls": -csls, "crd": -crd}
        for name in ("lemon", "traced"):
            p = OUT / f"score_{name}_{ds}_{facet}.npz"
            if p.is_file():
                methods[name] = np.load(p, allow_pickle=True)["score"]
        p = OUT / f"score_gmm_{ds}_{facet}.npz"
        if p.is_file():
            methods["gmm"] = np.load(p, allow_pickle=True)["score"]

        for name, anom in methods.items():
            rows.append({
                "facet": f"{ds}/{facet}", "method": name,
                "n": int(n), "cycles": int(pd.Series(cid).nunique()),
                "auroc": round(float(roc_auc_score(y, anom)), 4),
                "auprc": round(float(average_precision_score(y, anom)), 4),
                "acc2afc": round(acc2afc(anom, y, cid), 6)})
            print(rows[-1], flush=True)

        # paired bootstrap: CRD vs each non-trivial baseline
        for base in ("cosine", "csls", "lemon", "traced", "gmm"):
            if base not in methods:
                continue
            m, lo, hi = boot_delta(methods[base], methods["crd"], y, cid)
            rows.append({"facet": f"{ds}/{facet}", "method": f"crd - {base}",
                         "delta_pp": round(m, 2), "ci_lo": round(lo, 2),
                         "ci_hi": round(hi, 2)})
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "method_comparison.csv", index=False, encoding="utf-8-sig")
    print()
    main_tbl = df[df.delta_pp.isna()].pivot(index="method", columns="facet", values="auroc")
    print(main_tbl.to_string())
    print()
    print(df[df.delta_pp.notna()].pivot(index="method", columns="facet",
                                        values="delta_pp").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
