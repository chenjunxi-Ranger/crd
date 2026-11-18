"""Audit: are the BLIP-ITC numbers consistent, and which manifest do they use?"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "fgbench"
BENCH = ROOT / "fgbench" / "data"

FACETS = [("VG", "Attribute"), ("VG", "Relation"), ("COCO", "Quantity")]


def manifest_counts():
    out = {}
    for split in ("test",):
        for name, env in (("fgbench_v1", BENCH / "fgbench_v1"),
                          ("fgbench_v1_strict", BENCH / "fgbench_v1_strict")):
            p = env / f"{split}.parquet"
            if not p.is_file():
                continue
            d = pd.read_parquet(p)
            for (ds, fac), g in d.groupby(["dataset", "facet"]):
                out[(name, ds, fac)] = (int(len(g)), int(g.cycle_id.nunique()))
    return out


def main() -> int:
    man = manifest_counts()
    print("=== manifest 行数 / cycle 数（test） ===")
    for k, v in sorted(man.items()):
        print(f"  {k[0]:18s} {k[1]}/{k[2]:10s} rows={v[0]:6d} cycles={v[1]:6d}")

    print("\n=== BLIP-ITC CRD：逐种子 vs 集成 ===")
    for ds, fac in FACETS:
        per_seed, files = [], []
        for s in (0, 1, 2):
            p = OUT / f"crd_scores_blipitc_{ds}_{fac}_row_s{s}.npz"
            if not p.is_file():
                print(f"  MISSING {p.name}")
                continue
            z = np.load(p, allow_pickle=True)
            files.append(p.name)
            per_seed.append(roc_auc_score(z["y"], -z["crd"]))
        zs = [np.load(OUT / f, allow_pickle=True) for f in files]
        ens = np.mean([-z["crd"] for z in zs], axis=0)
        y = zs[0]["y"]
        print(f"\n  {ds}/{fac}: n={len(y)} cycles={len(np.unique(zs[0]['cycle_id']))}")
        print(f"    per-seed AUC : {[round(a,4) for a in per_seed]}")
        print(f"    seed mean    : {np.mean(per_seed):.4f} ± {np.std(per_seed, ddof=1):.4f}")
        print(f"    ensemble AUC : {roc_auc_score(y, ens):.4f}")
        print(f"    gap (ens-mean): {roc_auc_score(y, ens)-np.mean(per_seed):+.4f}")

    print("\n=== 各方法在同一 facet 上的 n / y 一致性 ===")
    for ds, fac in FACETS:
        ref = None
        for name, path in (
                ("crd", f"crd_scores_blipitc_{ds}_{fac}_row_s0.npz"),
                ("csls/gmm", f"score_blipitc_csls_{ds}_{fac}.npz"),
                ("lemon", f"score_lemon_blipitc_{ds}_{fac}.npz"),
                ("traced", f"score_traced_blipitc_{ds}_{fac}.npz")):
            p = OUT / path
            if not p.is_file():
                print(f"  {ds}/{fac:10s} {name:9s} MISSING ({path})")
                continue
            z = np.load(p, allow_pickle=True)
            y = z["y"]
            if ref is None:
                ref = y
                tag = "ref"
            else:
                tag = "same" if (len(y) == len(ref) and np.array_equal(y, ref)) else "DIFF"
            print(f"  {ds}/{fac:10s} {name:9s} n={len(y):6d} y_mean={y.mean():.3f} {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
