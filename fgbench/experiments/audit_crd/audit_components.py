"""Audit of what CRD v1 actually learns: components, variance, and usefulness.

Question this answers, and nothing else: the trained additive head q = a(u) +
b(v) + b0 was fitted to the pair's own standardised score z. What *is* q?

Part A - per-head statistics on the TEST split (never seen in training):
    std(q_I), std(q_T), std(q), corr(q_I, z), corr(q_T, z),
    R^2(q -> z), std(z - q) / std(z), AUROC(q_I), AUROC(q_T), AUROC(q)

Part B - is q useful or harmful? compare four scores on the same test rows:
    S0 = z            raw standardised cosine
    S1 = q            the prediction itself
    S2 = z - q        residual after removing the whole prediction
    S3 = z - lam*q    the calibrated score CRD actually deploys
  AUROC / AUPRC, plus cycle-clustered paired bootstrap of S1, S2, S3 vs S0.

Run:  python fgbench/experiments/audit_crd/audit_components.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fgbench.models.crd_backbone import Off  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "audit_crd"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FACETS = [("COCO", "Entity"), ("COCO", "Quantity"), ("VG", "Attribute"),
          ("VG", "Relation")]
N_BOOT = 1000


def auc(y, x):
    return float(roc_auc_score(y, -x))


def ap(y, x):
    return float(average_precision_score(y, -x))


def load_facet(ds, facet, seed):
    npz = CACHE / f"crdbb_clip_{ds}_{facet}_test.npz"
    meta = npz.with_suffix(".meta.parquet")
    z = np.load(npz, allow_pickle=True)
    prov = json.loads((CACHE / f"crdbb_prov_clip_{ds}_{facet}_both_row_s{seed}.json")
                      .read_text(encoding="utf-8"))
    head = CACHE / f"crdbb_clip_{ds}_{facet}_both_row_s{seed}_head.pt"
    sc = np.load(CACHE / f"crdbb_scores_clip_{ds}_{facet}_both_row_s{seed}.npz",
                 allow_pickle=True)
    ck = torch.load(head, map_location="cpu", weights_only=False)
    m = Off(d=int(ck["dim"]), parts="both")
    m.load_state_dict(ck["state"]); m.eval()
    U, V = z["U"].astype(np.float32), z["V"].astype(np.float32)
    mu, sd = float(prov["mu"]), float(prov["sd"])
    S = (U * V).sum(1)
    d = pd.read_parquet(meta)
    return {"U": U, "V": V, "z": (S - mu) / sd, "y": d.pair_label.to_numpy(),
            "cyc": d.cycle_id.values, "head": m, "lam": float(sc["lam"][0])}


def heads_out(m, U, V):
    with torch.no_grad():
        qI = m.a(torch.tensor(U)).numpy().ravel()
        qT = m.b(torch.tensor(V)).numpy().ravel()
        b0 = float(m.b0)
    return qI, qT, qI + qT + b0


def main() -> int:
    rng = np.random.default_rng(0)
    partA, partB, boot = [], [], []

    for ds, facet in FACETS:
        for seed in (0, 1, 2):
            f = load_facet(ds, facet, seed)
            z, y, cyc = f["z"], f["y"], f["cyc"]
            qI, qT, q = heads_out(f["head"], f["U"], f["V"])
            res = z - q
            sst = float(((z - z.mean()) ** 2).sum())

            partA.append({
                "facet": facet, "seed": seed,
                "std_z": round(float(z.std()), 4),
                "std_qI": round(float(qI.std()), 4),
                "std_qT": round(float(qT.std()), 4),
                "std_q": round(float(q.std()), 4),
                "corr_qI_z": round(float(np.corrcoef(qI, z)[0, 1]), 4),
                "corr_qT_z": round(float(np.corrcoef(qT, z)[0, 1]), 4),
                "corr_q_z": round(float(np.corrcoef(q, z)[0, 1]), 4),
                "r2_test": round(1 - float((res ** 2).sum()) / sst, 4),
                "resid_ratio": round(float(res.std() / z.std()), 4),
                "auroc_qI": round(auc(y, qI), 4),
                "auroc_qT": round(auc(y, qT), 4),
                "auroc_q": round(auc(y, q), 4),
            })

            lam = f["lam"]
            variants = {"S0_z": z, "S1_q": q, "S2_z_minus_q": z - q,
                        "S3_z_minus_lamq": z - lam * q}
            for name, x in variants.items():
                partB.append({"facet": facet, "seed": seed, "score": name,
                              "auroc": round(auc(y, x), 4),
                              "auprc": round(ap(y, x), 4)})

            # paired bootstrap vs S0, only for the deployed lambda and the residual
            idx_by = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
                lambda s: s.to_numpy()).to_dict()
            ids = list(idx_by)
            for name in ("S3_z_minus_lamq", "S2_z_minus_q"):
                d = []
                for _ in range(N_BOOT):
                    pick = rng.choice(ids, len(ids), replace=True)
                    ix = np.concatenate([idx_by[c] for c in pick])
                    yy = y[ix]
                    if yy.min() == yy.max():
                        continue
                    d.append(auc(yy, variants[name][ix]) - auc(yy, z[ix]))
                d = np.asarray(d)
                boot.append({"facet": facet, "seed": seed, "vs": name,
                             "delta_pp": round((auc(y, variants[name]) - auc(y, z)) * 100, 2),
                             "ci_lo": round(float(np.percentile(d, 2.5)) * 100, 2),
                             "ci_hi": round(float(np.percentile(d, 97.5)) * 100, 2),
                             "p_better": round(float((d <= 0).mean()), 3)})
        print(f"{facet} done", flush=True)

    a = pd.DataFrame(partA)
    b = pd.DataFrame(partB)
    c = pd.DataFrame(boot)
    a.to_csv(OUT / "components.csv", index=False, encoding="utf-8-sig")
    b.to_csv(OUT / "scores.csv", index=False, encoding="utf-8-sig")
    c.to_csv(OUT / "bootstrap.csv", index=False, encoding="utf-8-sig")

    print("\n== Part A: what is q? (mean over 3 seeds) ==")
    print(a.groupby("facet")[["std_z", "std_qI", "std_qT", "std_q", "corr_q_z",
                              "r2_test", "resid_ratio", "auroc_q"]]
          .mean().round(4).to_string())

    print("\n== Part B: AUROC of the four scores ==")
    print(b.groupby(["facet", "score"]).auroc.mean().unstack().round(4).to_string())

    print("\n== bootstrap vs S0 ==")
    print(c.groupby(["facet", "vs"])[["delta_pp", "ci_lo", "ci_hi"]]
          .mean().round(2).to_string())
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
