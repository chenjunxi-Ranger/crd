"""Aggregate CRD multi-seed results: per-seed metrics + clustered bootstrap of Delta."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b16")
    ap.add_argument("--dataset", default="VG")
    ap.add_argument("--facet", default="Attribute")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    seeds = [int(s) for s in args.seeds.split(",")]
    data = {}
    for s in seeds:
        p = OUT / f"crd_scores_{args.tag}_{args.dataset}_{args.facet}_s{s}.npz"
        z = np.load(p, allow_pickle=True)
        data[s] = z
        print(f"seed {s}: rows {len(z['y'])} lam {float(z['lam'][0])}", flush=True)

    cyc = data[seeds[0]]["cycle_id"]
    y = data[seeds[0]]["y"]
    gids = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(lambda x: x.to_numpy()).to_dict()
    keys = list(gids)

    rows = []
    boot_auc = {s: [] for s in seeds}
    boot_d = {s: [] for s in seeds}
    for s in seeds:
        cos, crd = data[s]["cos"], data[s]["crd"]
        accs = []
        df = pd.DataFrame({"c": cyc, "y": y, "s": -crd})
        for cid, g in df.groupby("c"):
            if len(g) != 4:
                continue
            accs.append(g[g.y == 0].s.sum() < g[g.y == 1].s.sum())
        rows.append({"seed": s,
                     "auc_cos": round(float(roc_auc_score(y, -cos)), 4),
                     "auc_crd": round(float(roc_auc_score(y, -crd)), 4),
                     "auprc_cos": round(float(average_precision_score(y, -cos)), 4),
                     "auprc_crd": round(float(average_precision_score(y, -crd)), 4),
                     "acc2afc": round(float(np.mean(accs)), 6),
                     "delta_pp": round(float(roc_auc_score(y, -crd)
                                             - roc_auc_score(y, -cos)) * 100, 2)})

    for _ in range(args.boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        ii = np.concatenate([gids[keys[k]] for k in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        for s in seeds:
            cos, crd = data[s]["cos"][ii], data[s]["crd"][ii]
            a0 = roc_auc_score(y[ii], -cos)
            a1 = roc_auc_score(y[ii], -crd)
            boot_auc[s].append(a1)
            boot_d[s].append((a1 - a0) * 100)

    per = []
    for r in rows:
        s = r["seed"]
        per.append({**r,
                    "delta_ci": [round(float(np.percentile(boot_d[s], 2.5)), 2),
                                 round(float(np.percentile(boot_d[s], 97.5)), 2)],
                    "p_delta_gt0": round(float(np.mean(np.array(boot_d[s]) > 0)), 3)})

    d = pd.DataFrame(per)
    summary = {
        "seeds": seeds,
        "auc_crd_mean": round(float(d.auc_crd.mean()), 4),
        "auc_crd_sd": round(float(d.auc_crd.std(ddof=1)), 4),
        "auc_cos_mean": round(float(d.auc_cos.mean()), 4),
        "delta_pp_mean": round(float(d.delta_pp.mean()), 2),
        "delta_pp_sd": round(float(d.delta_pp.std(ddof=1)), 3),
        "acc2afc_all_seeds": sorted(set(d.acc2afc.tolist())),
    }
    print(d.to_string(index=False), flush=True)
    print(summary, flush=True)
    d.to_csv(OUT / f"crd_seeds_{args.dataset}_{args.facet}.csv", index=False,
             encoding="utf-8-sig")
    jdump({"per_seed": per, "summary": summary},
          OUT / f"crd_seeds_{args.dataset}_{args.facet}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
