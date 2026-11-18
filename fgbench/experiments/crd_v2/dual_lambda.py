"""Priority 1: can the two heads be subtracted with different strengths?

No training. Existing checkpoints are reused; the only new freedom is a pair of
scalars (lambda_I, lambda_T) chosen on the validation split.

Both head outputs are standardised with TRAIN mean/std first, so that the two
lambdas are on a common scale and a difference between them is not an artefact of
one head having a larger output variance.

    A  S = z - lambda (qI~ + qT~)      shared strength (current CRD)
    B  S = z - lambdaI qI~             image side only
    C  S = z - lambdaT qT~             text side only
    D  S = z - lambdaI qI~ - lambdaT qT~   independent strengths

Selection is a grid search on validation AUROC (step 0.1, range 0..2); the test
split is scored once per configuration, after the lambdas are fixed.

Run:  python fgbench/experiments/crd_v2/dual_lambda.py
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
OUT = CACHE / "crd_v2"
OUT.mkdir(parents=True, exist_ok=True)
import os  # noqa: E402

_STEP = float(os.environ.get("DUAL_STEP", "0.1"))
GRID = np.round(np.arange(0, 2.01, _STEP), 1)
N_BOOT = 1000
FACETS = [("COCO", "Entity"), ("COCO", "Quantity"), ("VG", "Attribute"),
          ("VG", "Relation")]


def auc(y, x):
    return float(roc_auc_score(y, -x))


def ap(y, x):
    return float(average_precision_score(y, -x))


def load(ds, facet, name, mu, sd):
    fn = (sorted(CACHE.glob(f"crdbb_clip_{ds}_{facet}_c*_train.npz"))[0]
          if name == "train" else CACHE / f"crdbb_clip_{ds}_{facet}_{name}.npz")
    z = np.load(fn, allow_pickle=True)
    U, V = z["U"].astype(np.float32), z["V"].astype(np.float32)
    meta = pd.read_parquet(fn.with_suffix(".meta.parquet"))
    return U, V, ((U * V).sum(1) - mu) / sd, meta


def heads(m, U, V):
    with torch.no_grad():
        qI = m.a(torch.tensor(U)).numpy().ravel() if m.wa else np.zeros(len(U))
        qT = m.b(torch.tensor(V)).numpy().ravel() if m.wb else np.zeros(len(V))
    return qI, qT


def main() -> int:
    rng = np.random.default_rng(0)
    rows, boot, grids = [], [], []

    for ds, facet in FACETS:
        for seed in (0, 1, 2):
            prov = json.loads((CACHE / f"crdbb_prov_clip_{ds}_{facet}_both_row_s{seed}"
                               ".json").read_text(encoding="utf-8"))
            mu, sd = float(prov["mu"]), float(prov["sd"])
            Utr, Vtr, ztr, _ = load(ds, facet, "train", mu, sd)
            Uva, Vva, zva, mva = load(ds, facet, "val", mu, sd)
            Ute, Vte, zte, mte = load(ds, facet, "test", mu, sd)

            ck = torch.load(CACHE / f"crdbb_clip_{ds}_{facet}_both_row_s{seed}_head.pt",
                            map_location="cpu", weights_only=False)
            m = Off(d=int(ck["dim"]), parts="both")
            m.load_state_dict(ck["state"]); m.eval()

            qI_tr, qT_tr = heads(m, Utr, Vtr)
            qI_va, qT_va = heads(m, Uva, Vva)
            qI_te, qT_te = heads(m, Ute, Vte)

            # standardise both heads with TRAIN statistics
            mI, sI = float(qI_tr.mean()), float(qI_tr.std() + 1e-8)
            mT, sT = float(qT_tr.mean()), float(qT_tr.std() + 1e-8)
            qI_va, qI_te = (qI_va - mI) / sI, (qI_te - mI) / sI
            qT_va, qT_te = (qT_va - mT) / sT, (qT_te - mT) / sT

            y_va, y_te = mva.pair_label.to_numpy(), mte.pair_label.to_numpy()

            # ---- A: shared lambda over the summed heads
            s_va = qI_va + qT_va
            a_va = [auc(y_va, zva - L * s_va) for L in GRID]
            lamA = float(GRID[int(np.argmax(a_va))])
            teA = zte - lamA * (qI_te + qT_te)

            # ---- B / C: one side only
            b_va = [auc(y_va, zva - L * qI_va) for L in GRID]
            lamB = float(GRID[int(np.argmax(b_va))])
            teB = zte - lamB * qI_te

            c_va = [auc(y_va, zva - L * qT_va) for L in GRID]
            lamC = float(GRID[int(np.argmax(c_va))])
            teC = zte - lamC * qT_te

            # ---- D: independent 2D grid
            G = np.zeros((len(GRID), len(GRID)))
            for i, LI in enumerate(GRID):
                col = zva - LI * qI_va
                for j, LT in enumerate(GRID):
                    G[i, j] = auc(y_va, col - LT * qT_va)
            i, j = np.unravel_index(int(np.argmax(G)), G.shape)
            lamI, lamT = float(GRID[i]), float(GRID[j])
            teD = zte - lamI * qI_te - lamT * qT_te
            grids.append({"facet": facet, "seed": seed, "lamI": lamI, "lamT": lamT,
                          "val_auc": round(float(G[i, j]), 4),
                          "val_auc_A": round(float(max(a_va)), 4),
                          "gain_val_pp": round((float(G[i, j]) - max(a_va)) * 100, 2)})

            rows.append({
                "facet": facet, "seed": seed,
                "lam_A": lamA, "lam_B": lamB, "lam_C": lamC,
                "lamI_D": lamI, "lamT_D": lamT,
                "auroc_A": round(auc(y_te, teA), 4),
                "auroc_B": round(auc(y_te, teB), 4),
                "auroc_C": round(auc(y_te, teC), 4),
                "auroc_D": round(auc(y_te, teD), 4),
                "auprc_A": round(ap(y_te, teA), 4),
                "auprc_D": round(ap(y_te, teD), 4),
                "auroc_cos": round(auc(y_te, zte), 4),
            })

            # ---- bootstrap D vs A (cycle clustered)
            cyc = mte.cycle_id.values
            by = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
                lambda s: s.to_numpy()).to_dict()
            ids = list(by)
            d = []
            for _ in range(N_BOOT):
                pick = rng.choice(ids, len(ids), replace=True)
                ix = np.concatenate([by[c] for c in pick])
                yy = y_te[ix]
                if yy.min() == yy.max():
                    continue
                d.append(auc(yy, teD[ix]) - auc(yy, teA[ix]))
            d = np.asarray(d)
            boot.append({"facet": facet, "seed": seed,
                         "delta_pp": round((auc(y_te, teD) - auc(y_te, teA)) * 100, 2),
                         "ci_lo": round(float(np.percentile(d, 2.5)) * 100, 2),
                         "ci_hi": round(float(np.percentile(d, 97.5)) * 100, 2),
                         "p": round(float((d <= 0).mean()), 3)})
        print(f"{facet} done", flush=True)

    r = pd.DataFrame(rows)
    b = pd.DataFrame(boot)
    g = pd.DataFrame(grids)
    r.to_csv(OUT / "dual_lambda.csv", index=False, encoding="utf-8-sig")
    b.to_csv(OUT / "dual_lambda_boot.csv", index=False, encoding="utf-8-sig")
    g.to_csv(OUT / "dual_lambda_grid.csv", index=False, encoding="utf-8-sig")

    print("\n== test AUROC (mean over 3 seeds) ==")
    print(r.groupby("facet")[["auroc_cos", "auroc_A", "auroc_B", "auroc_C",
                              "auroc_D"]].mean().round(4).to_string())
    print("\n== selected lambdas ==")
    print(r.groupby("facet")[["lam_A", "lam_B", "lam_C", "lamI_D", "lamT_D"]]
          .mean().round(2).to_string())
    print("\n== D vs A (cycle bootstrap) ==")
    print(b.groupby("facet")[["delta_pp", "ci_lo", "ci_hi"]].mean().round(2).to_string())
    print("\n== validation gain of 2D over shared lambda ==")
    print(g.groupby("facet")["gain_val_pp"].mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
