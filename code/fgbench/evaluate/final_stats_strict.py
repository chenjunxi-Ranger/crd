"""F5: paired cycle-clustered bootstrap (CRD vs each baseline) + Holm correction.

Reads the same score files as strict_main_table so every comparison uses
identical rows and labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))

OUT = ROOT / "scores"
BENCH = ROOT / "data" / "manifests"
BOOT = 1000
FACETS = [("COCO", "Entity"), ("COCO", "Quantity"), ("VG", "Attribute"),
          ("VG", "Action"), ("VG", "Relation")]


def load_crd(ds, facet):
    files = (sorted(OUT.glob(f"crd_action_scores_s*.npz")) if facet == "Action"
             else sorted(OUT.glob(f"crdbb_scores_clip_{ds}_{facet}_both_row_s*.npz")))
    arrs = [np.load(f, allow_pickle=True) for f in files]
    if not arrs:
        return None
    return {"crd": np.mean([-z["crd"] for z in arrs], axis=0),
            "y": arrs[0]["y"], "cycle_id": arrs[0]["cycle_id"]}


def load_other(ds, facet, name):
    if name in ("cos", "csls"):
        p = OUT / f"strict_base_{ds}_{facet}.npz"
        if not p.is_file():
            return None
        z = np.load(p, allow_pickle=True)
        if name not in z.files or z[name] is None:
            return None
        return {"score": z[name], "cycle_id": z["cycle_id"], "y": z["y"]}
    p = OUT / f"score_gmm_{ds}_{facet}.npz" if name == "gmm" else \
        OUT / f"score_{name}_clip_{ds}_{facet}.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    return {"score": z["score"], "cycle_id": z["cycle_id"], "y": z["y"]}


def main() -> int:
    te = pd.read_parquet(BENCH / "test.parquet")
    rows = []
    for ds, facet in FACETS:
        base = load_crd(ds, facet)
        if base is None:
            continue
        strict = set(te[(te.dataset == ds) & (te.facet == facet)].cycle_id)
        keep = np.isin(base["cycle_id"], np.array(list(strict)))
        crd, y, cid = base["crd"][keep], base["y"][keep], base["cycle_id"][keep]

        for name in ("cos", "csls", "gmm", "lemon", "traced"):
            other = load_other(ds, facet, name)
            if other is None:
                continue
            m = np.isin(other["cycle_id"], np.array(list(strict)))
            sc = other["score"][m]
            if len(sc) != len(crd):
                continue
            rng = np.random.default_rng(0)
            idx = pd.Series(np.arange(len(y))).groupby(cid).apply(
                lambda x: x.to_numpy()).to_dict()
            keys = list(idx)
            diffs = []
            for _ in range(BOOT):
                pk = rng.choice(len(keys), len(keys), replace=True)
                ii = np.concatenate([idx[keys[t]] for t in pk])
                yy = y[ii]
                if yy.min() == yy.max():
                    continue
                # both sides are already ANOMALY scores (higher = noisier)
                diffs.append((roc_auc_score(yy, crd[ii])
                              - roc_auc_score(yy, sc[ii])) * 100)
            diffs = np.array(diffs)
            rows.append({"dataset": ds, "facet": facet, "baseline": name,
                         "delta_pp": round(float(diffs.mean()), 2),
                         "ci_lo": round(float(np.percentile(diffs, 2.5)), 2),
                         "ci_hi": round(float(np.percentile(diffs, 97.5)), 2),
                         "p_one_sided": round(float(max(1.0 / BOOT,
                                                         (diffs <= 0).mean())), 4)})
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no comparisons")
        return 1
    # Holm over all tests
    order = df.p_one_sided.sort_values().index
    m = len(df)
    holm = pd.Series(1.0, index=df.index)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * df.loc[i, "p_one_sided"])
        running = max(running, val)
        holm[i] = running
    df["p_holm"] = holm.round(4)
    df["significant"] = df.p_holm < 0.05
    df.to_csv(OUT / "final_stats_strict.csv", index=False, encoding="utf-8-sig")
    print()
    print(df.to_string(index=False))
    print(f"\ntests={m}  significant={int(df.significant.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
