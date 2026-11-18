"""F1-F8 verification: conventions, manifest consistency, isolation, checkpoints."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fgbench"
BENCH = ROOT / "fgbench" / "data" / "fgbench_v1_strict"


def strict_ids():
    te = pd.read_parquet(BENCH / "test.parquet")
    return {(d, f): set(g.cycle_id) for (d, f), g in te.groupby(["dataset", "facet"])}


def main() -> int:
    S = strict_ids()

    print("=== F4 · 各方法的测试集 cycle 集合是否与 strict manifest 一致 ===")
    files = []
    files += [(f, "crd", ("crdbb", )) for f in glob.glob(str(OUT / "crdbb_scores_clip_*_both_row_s0.npz"))]
    files += [(f, "lemon", None) for f in glob.glob(str(OUT / "score_lemon_clip_*.npz"))]
    files += [(f, "traced", None) for f in glob.glob(str(OUT / "score_traced_clip_*.npz"))]
    files += [(f, "gmm", None) for f in glob.glob(str(OUT / "score_gmm_*.npz"))]
    for f, kind, _ in sorted(files):
        z = np.load(f, allow_pickle=True)
        name = Path(f).name
        ds, fac = None, None
        for d in ("VG", "COCO"):
            for fc in ("Attribute", "Relation", "Quantity", "Action", "Entity"):
                if f"_{d}_{fc}" in name:
                    ds, fac = d, fc
        if ds is None:
            print(f"  {name:60s} (facet 未识别)")
            continue
        ids = set(z["cycle_id"])
        ref = S.get((ds, fac), set())
        extra, missing = ids - ref, ref - ids
        print(f"  {name:60s} n={len(ids):6d} extra={len(extra):4d} missing={len(missing):4d} "
              f"{'OK' if not extra and not missing else 'MISMATCH'}")

    print("\n=== F8 · checkpoint 是否保存 backbone/facet/seed/mu/sd/lam ===")
    import torch
    for f in sorted(glob.glob(str(OUT / "crdbb_*_head.pt")))[:6]:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        keys = sorted(ck.keys())
        print(f"  {Path(f).name:60s} keys={keys}")

    print("\n=== F2/F1 · 两种聚合口径是否同时可得 ===")
    for ds, fac in (("VG", "Attribute"), ("VG", "Relation"), ("COCO", "Quantity")):
        seeds = []
        for s in (0, 1, 2):
            p = OUT / f"crdbb_scores_clip_{ds}_{fac}_both_row_s{s}.npz"
            if p.is_file():
                seeds.append(np.load(p, allow_pickle=True))
        if len(seeds) < 2:
            print(f"  {ds}/{fac}: seeds={len(seeds)}")
            continue
        from sklearn.metrics import roc_auc_score
        per = [roc_auc_score(z["y"], -z["crd"]) for z in seeds]
        ens = roc_auc_score(seeds[0]["y"], -np.mean([z["crd"] for z in seeds], axis=0))
        print(f"  {ds}/{fac:10s} seedmean={np.mean(per):.4f}±{np.std(per, ddof=1):.4f} "
              f"ensemble={ens:.4f} gap={ens-np.mean(per):+.4f}")

    print("\n=== F3 · GMM 是否有 CLIP 空间结果 ===")
    for f in sorted(glob.glob(str(OUT / "score_gmm_*.npz"))):
        z = np.load(f, allow_pickle=True)
        print(f"  {Path(f).name:40s} n={len(z['y'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
