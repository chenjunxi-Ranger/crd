"""Stage-3 verification + unified ablation table for the standardised pipeline.

Checks
  1. manifest / cache / train-order hashes identical across the three seeds
  2. mu, sd computed from TRAIN only (recomputed from cache, compared)
  3. lambda is the argmax of the VAL AUROC curve (recomputed from saved scores)
  4. AUROC / AUPRC / 2AFC recomputed from the saved per-row scores
  5. cycle-clustered paired bootstrap for the key contrasts
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
CFG = "clip_VG_Attribute"
SEEDS = (0, 1, 2)
ARMS = {"image-only": "image_row", "text-only": "text_row",
        "additive(row)": "both_row", "cycle-level": "both_cycle",
        "shuffled placebo": "both_shuffled"}


def prov(arm_key: str, seed: int) -> dict:
    return json.load(open(OUT / f"crdbb_prov_{CFG}_{arm_key}_s{seed}.json",
                          encoding="utf-8"))


def scores(arm_key: str, seed: int) -> dict:
    return dict(np.load(OUT / f"crdbb_scores_{CFG}_{arm_key}_s{seed}.npz",
                        allow_pickle=True))


def acc2afc(anom, y, cid):
    df = pd.DataFrame({"c": cid, "y": y, "s": anom})
    a = [g[g.y == 0].s.sum() < g[g.y == 1].s.sum()
         for _, g in df.groupby("c") if len(g) == 4]
    return float(np.mean(a)) if a else float("nan")


def main() -> int:
    print("=== 1. 三种子数据一致性 ===")
    for name, key in ARMS.items():
        ps = [prov(key, s) for s in SEEDS]
        man = {json.dumps(p["manifest_sha"], sort_keys=True) for p in ps}
        cac = {json.dumps(p["cache_sha"], sort_keys=True) for p in ps}
        ordr = {p["train_order_sha"] for p in ps}
        musd = {(round(p["mu"], 10), round(p["sd"], 10)) for p in ps}
        rows = {p["n_train_rows"] for p in ps}
        print(f"  {name:16s} manifest={len(man)==1} cache={len(cac)==1} "
              f"order={len(ordr)==1} mu/sd={len(musd)==1} n_train={rows}")

    print("\n=== 2. mu/sd 只来自 train（用缓存复算比对） ===")
    for name, key in ARMS.items():
        p = prov(key, 0)
        z = np.load(OUT / f"crdbb_clip_VG_Attribute_c8000_train.npz", allow_pickle=True)
        S = z["S"]
        mu, sd = float(S.mean()), float(S.std() + 1e-8)
        print(f"  {name:16s} prov mu={p['mu']:.6f} recomputed={mu:.6f} "
              f"diff={abs(p['mu']-mu):.2e} | sd diff={abs(p['sd']-sd):.2e}")

    print("\n=== 3. lambda 是否为 val AUROC 的 argmax ===")
    for name, key in ARMS.items():
        for s in SEEDS:
            d = scores(key, s)
            lam = float(d["lam"][0])
            if lam <= 0:
                print(f"  {name:16s} s{s}: lam=0 (val 选到不校正，无法反解 mh)")
                continue
            mh = (d["va_raw"] - d["va_crd"]) / lam
            grid = np.round(np.arange(0, 2.01, 0.1), 1)
            aucs = [roc_auc_score(d["va_y"], -(d["va_raw"] - L * mh)) for L in grid]
            best = float(grid[int(np.argmax(aucs))])
            print(f"  {name:16s} s{s}: stored lam={lam:.1f} argmax={best:.1f} "
                  f"{'OK' if abs(best-lam) < 1e-9 else 'MISMATCH'}")

    print("\n=== 4. 指标（三种子，从逐样本分数重算） ===")
    rows = []
    for name, key in ARMS.items():
        aucs, aps, a2 = [], [], []
        for s in SEEDS:
            d = scores(key, s)
            y, cid, anom = d["y"], d["cycle_id"], -d["crd"]
            aucs.append(roc_auc_score(y, anom))
            aps.append(average_precision_score(y, anom))
            a2.append(acc2afc(anom, y, cid))
        rows.append({"arm": name, "auroc": round(float(np.mean(aucs)), 4),
                     "sd": round(float(np.std(aucs, ddof=1)), 4),
                     "auprc": round(float(np.mean(aps)), 4),
                     "acc2afc": round(float(np.mean(a2)), 6),
                     "delta_pp": round((float(np.mean(aucs)) - 0.6806) * 100, 2)})
        print(f"  {rows[-1]}")

    print("\n=== 5. cycle 聚类配对 bootstrap（CRD 臂 vs 对照臂） ===")
    pairs = [("additive(row)", "image-only"), ("additive(row)", "text-only"),
             ("additive(row)", "cycle-level"), ("additive(row)", "shuffled placebo")]
    for a, b in pairs:
        deltas = []
        rng = np.random.default_rng(0)
        for s in SEEDS:
            da, db = scores(ARMS[a], s), scores(ARMS[b], s)
            y, cid = da["y"], da["cycle_id"]
            idx = pd.Series(np.arange(len(y))).groupby(cid).apply(
                lambda x: x.to_numpy()).to_dict()
            keys = list(idx)
            ds = []
            for _ in range(500):
                pk = rng.choice(len(keys), len(keys), replace=True)
                ii = np.concatenate([idx[keys[t]] for t in pk])
                yy = y[ii]
                if yy.min() == yy.max():
                    continue
                ds.append((roc_auc_score(yy, -da["crd"][ii])
                           - roc_auc_score(yy, -db["crd"][ii])) * 100)
            deltas.append(np.mean(ds))
        print(f"  {a:16s} - {b:16s} delta={np.mean(deltas):+.2f} pp "
              f"(per-seed {[round(x,2) for x in deltas]})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ablation_standardised.csv", index=False, encoding="utf-8-sig")
    print("\nwrote", OUT / "ablation_standardised.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
