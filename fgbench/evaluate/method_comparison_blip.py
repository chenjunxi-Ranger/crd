"""Same-task comparison table in the BLIP-ITC feature space (strict manifests)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fgbench"
FACETS = [("VG", "Attribute"), ("VG", "Relation"), ("COCO", "Quantity")]
SEEDS = (0, 1, 2)


def acc2afc(anom, y, cid):
    df = pd.DataFrame({"c": cid, "y": y, "s": anom})
    a = [g[g.y == 0].s.sum() < g[g.y == 1].s.sum()
         for _, g in df.groupby("c") if len(g) == 4]
    return float(np.mean(a)) if a else float("nan")


def boot_delta(base, crd, y, cid, n=1000, seed=0):
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
        ds.append(roc_auc_score(yy, crd[ii]) - roc_auc_score(yy, base[ii]))
    ds = np.array(ds) * 100
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def main() -> int:
    rows = []
    for ds, facet in FACETS:
        crd = []
        for s in SEEDS:
            z = np.load(OUT / f"crd_scores_blipitc_{ds}_{facet}_row_s{s}.npz",
                        allow_pickle=True)
            crd.append(z["crd"])
            y, cid = z["y"], z["cycle_id"]
        crd = -np.mean(crd, axis=0)          # anomaly

        z = np.load(OUT / f"score_blipitc_csls_{ds}_{facet}.npz", allow_pickle=True)
        assert len(z["y"]) == len(y)
        methods = {"cosine": z["cos"], "csls": z["score"], "gmm": z["gmm"], "crd": crd}
        for name in ("lemon", "traced"):
            p = OUT / f"score_{name}_{ds}_{facet}.npz"
            if p.is_file():
                zz = np.load(p, allow_pickle=True)
                assert len(zz["score"]) == len(y), (name, len(zz["score"]), len(y))
                methods[name] = zz["score"]

        for name, anom in methods.items():
            rows.append({"facet": f"{ds}/{facet}", "method": name, "n": int(len(y)),
                         "cycles": int(pd.Series(cid).nunique()),
                         "auroc": round(float(roc_auc_score(y, anom)), 4),
                         "auprc": round(float(average_precision_score(y, anom)), 4),
                         "acc2afc": round(acc2afc(anom, y, cid), 6)})
            print(rows[-1], flush=True)

        for b in ("cosine", "csls", "lemon", "traced", "gmm"):
            if b not in methods:
                continue
            m, lo, hi = boot_delta(methods[b], methods["crd"], y, cid)
            rows.append({"facet": f"{ds}/{facet}", "method": f"crd - {b}",
                         "delta_pp": round(m, 2), "ci_lo": round(lo, 2),
                         "ci_hi": round(hi, 2)})
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "method_comparison_blipitc.csv", index=False, encoding="utf-8-sig")
    print()
    m = df[df.delta_pp.isna()]
    print(m.pivot(index="method", columns="facet", values="auroc").to_string())
    print()
    print(m.pivot(index="method", columns="facet", values="acc2afc").to_string())
    print()
    print(df[df.delta_pp.notna()].pivot(index="method", columns="facet",
                                        values="delta_pp").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
