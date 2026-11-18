"""Paired cycle-clustered bootstrap for CRD v2 (E3 = CRD v1 + eta * g).

Reads `results/fgbench/crd_v2/interaction_scores_ Attribute .npz` written by
`run_additive_interaction.py --stage 2`, averages the three seeds per
configuration, and reports E3 - E1 with a 95% CI and a two-sided p.

E1 is the frozen CRD v1 score and is identical across seeds; only E3 varies.

Run:  python fgbench/experiments/crd_v2/bootstrap_interaction.py
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
N_BOOT = 500


def acc2afc(score, cycle, y):
    df = pd.DataFrame({"c": cycle, "y": y, "s": score})
    out = []
    for _, g in df.groupby("c"):
        if len(g) == 4:
            out.append(g[g.y == 0].s.sum() < g[g.y == 1].s.sum())
    return float(np.mean(out)) if out else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facet", default="Attribute")
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--configs", default="B_rank_uvdot,B_rank_dot_abs,B_rank_placebo")
    args = ap.parse_args()
    z = np.load(OUT / f"interaction_scores_{args.facet}_stage{args.stage}.npz",
                allow_pickle=True)
    rng = np.random.default_rng(0)
    recs = []
    for cfg in args.configs.split(","):
        # keys are "<scheme>_<kind>_<seed>__<field>"
        fields: dict[str, list] = {}
        for k in z.files:
            if not k.startswith(cfg + "_") or "__" not in k:
                continue
            _, field = k.split("__")
            fields.setdefault(field, []).append(z[k])
        if "E3" not in fields:
            print(f"[skip] {cfg}: no stored scores")
            continue
        e3 = fields["E3"]
        e1 = fields["E1"][0]
        cyc = fields["cycle"][0]
        y = fields["y"][0]
        n = len(e3)
        S3 = np.mean(e3, axis=0)
        a1 = roc_auc_score(y, -e1)
        a3 = roc_auc_score(y, -S3)
        print(f"[boot] {cfg}: E1 {a1:.4f} -> E3 {a3:.4f} ({N_BOOT} resamples)",
              flush=True)
        cyc_ids = pd.unique(cyc)
        by = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
            lambda s: s.to_numpy()).to_dict()
        d = []
        for b in range(N_BOOT):
            pick = rng.choice(cyc_ids, len(cyc_ids), replace=True)
            idx = np.concatenate([by[c] for c in pick])
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            d.append(roc_auc_score(yy, -S3[idx]) - roc_auc_score(yy, -e1[idx]))
        d = np.asarray(d)
        p_two = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
        recs.append({
            "facet": args.facet, "config": cfg, "seeds": n,
            "E1": round(float(a1), 4), "E3": round(float(a3), 4),
            "delta_pp": round(float(a3 - a1) * 100, 2),
            "ci_lo": round(float(np.percentile(d, 2.5)) * 100, 2),
            "ci_hi": round(float(np.percentile(d, 97.5)) * 100, 2),
            "p_two_sided": round(p_two, 4),
            "2afc_E1": round(acc2afc(-e1, cyc, y), 6),
            "2afc_E3": round(acc2afc(-S3, cyc, y), 6),
        })
        print(recs[-1], flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(OUT / f"bootstrap_interaction_{args.facet}.csv", index=False,
              encoding="utf-8-sig")
    print()
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
