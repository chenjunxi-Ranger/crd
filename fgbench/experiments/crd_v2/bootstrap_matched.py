"""Matched-probe bootstrap for CRD v2 across all five facets.

Compares, on identical test rows and with a cycle-clustered bootstrap:

    E1          CRD v1
    E2_rank     pair classifier (pairwise ranking), uvdot, 3 seeds
    E3_rank     E1 + eta * E2_rank
    E2_bce      pair classifier (BCE), uvdot, 3 seeds        <- matched probe
    E3_bce      E1 + eta * E2_bce

Two aggregation conventions are reported side by side, because the earlier note
quoted 0.7235-0.7240 (per-seed) next to 0.7256 (seed-averaged score):

    delta_ensemble_pp   AUROC(mean score over seeds) - AUROC(reference)
    delta_seedmean_pp   mean over seeds of AUROC(seed) - AUROC(reference)

Run:  python fgbench/experiments/crd_v2/bootstrap_matched.py --n-boot 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "fgbench" / "crd_v2"
FACETS = ["Attribute", "Quantity", "Relation", "Entity", "Action"]


def load(facet: str):
    p = OUT / f"interaction_scores_{facet}_stage3.npz"
    z = np.load(p, allow_pickle=True)
    out = {}
    for scheme in ("B_rank", "B_bce"):
        keys = sorted(k for k in z.files
                      if k.startswith(f"{scheme}_uvdot_") and k.endswith("__E3"))
        if not keys:
            continue
        out[scheme] = {
            "E3": [z[k] for k in keys],
            "E2": [z[k.replace("__E3", "__g")] for k in keys],
            "E1": z[keys[0].replace("__E3", "__E1")],
            "cycle": z[keys[0].replace("__E3", "__cycle")],
            "y": z[keys[0].replace("__E3", "__y")],
            "n_seeds": len(keys),
        }
    return out


def auc_anom(y, s):
    return float(roc_auc_score(y, -s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--facet", default="all")
    ap.add_argument("--tests",
                    default="E3_bce-E1,E3_bce-E2_bce,E3_rank-E1")
    args = ap.parse_args()
    wanted = [tuple(t.split("-", 1)) for t in args.tests.split(",")]
    facets = FACETS if args.facet == "all" else [args.facet]
    rng = np.random.default_rng(0)
    recs = []
    for facet in facets:
        try:
            z = load(facet)
        except FileNotFoundError:
            print(f"[skip] {facet}: no stage-3 scores")
            continue
        if "B_bce" not in z:
            print(f"[skip] {facet}: no B_bce")
            continue
        base = z["B_bce"]
        y, cyc = base["y"], base["cycle"]
        cyc_ids = pd.unique(cyc)
        by = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
            lambda s: s.to_numpy()).to_dict()
        e1 = base["E1"]
        refs = {
            "E1": ([e1] * base["n_seeds"]),
            "E2_rank": z["B_rank"]["E2"],
            "E2_bce": z["B_bce"]["E2"],
        }
        alts = {
            "E3_rank": z["B_rank"]["E3"],
            "E3_bce": z["B_bce"]["E3"],
        }
        tests = [t for t in wanted if t[0] in alts and t[1] in refs]
        for alt, ref in tests:
            A = np.mean(alts[alt], axis=0)
            R = np.mean(refs[ref], axis=0)
            a_alt, a_ref = auc_anom(y, A), auc_anom(y, R)
            seeds_alt = [auc_anom(y, s) for s in alts[alt]]
            seeds_ref = [auc_anom(y, s) for s in refs[ref]]
            delta_seed = float(np.mean(np.array(seeds_alt) - np.array(seeds_ref)))
            d = []
            print(f"[boot] {facet} {alt}-{ref}: {a_ref:.4f}->{a_alt:.4f} "
                  f"({args.n_boot} resamples)", flush=True)
            for b in range(args.n_boot):
                pick = rng.choice(cyc_ids, len(cyc_ids), replace=True)
                idx = np.concatenate([by[c] for c in pick])
                yy = y[idx]
                if yy.min() == yy.max():
                    continue
                d.append(auc_anom(yy, A[idx]) - auc_anom(yy, R[idx]))
                if b and b % 500 == 0:
                    print(f"    {b}/{args.n_boot}", flush=True)
            d = np.asarray(d)
            p_two = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
            rec = {
                "facet": facet, "comparison": f"{alt} - {ref}",
                "agg_ref": round(a_ref, 4), "agg_alt": round(a_alt, 4),
                "delta_ensemble_pp": round((a_alt - a_ref) * 100, 2),
                "ci_lo": round(float(np.percentile(d, 2.5)) * 100, 2),
                "ci_hi": round(float(np.percentile(d, 97.5)) * 100, 2),
                "p_two_sided": round(p_two, 4),
                "delta_seedmean_pp": round(delta_seed * 100, 2),
                "n_boot": int(len(d)),
            }
            recs.append(rec)
            print(json_dumps(rec), flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "bootstrap_matched.csv", index=False, encoding="utf-8-sig")
    print("\n", df.to_string(index=False), flush=True)
    return 0


def json_dumps(rec):
    import json
    return json.dumps(rec, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
