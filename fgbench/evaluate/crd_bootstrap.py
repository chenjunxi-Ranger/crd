"""Cycle-clustered paired bootstrap for CRD vs cosine vs CSLS, plus a 2AFC invariance check."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
DEV = "cpu"

CONFIGS = [
    ("VG", "Attribute", "crd_b16", "crd_b16_head.pt", "crd_b16.json", 0.2),
    ("VG", "Relation", "crd_b16_VG_Relation", "crd_b16_VG_Relation_head.pt",
     "crd_b16_VG_Relation.json", 0.3),
    ("COCO", "Quantity", "crd_b16_COCO_Quantity", "crd_b16_COCO_Quantity_head.pt",
     "crd_b16_COCO_Quantity.json", 0.6),
]


class Off(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.b = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u, v):
        return (self.a(u) + self.b(v) + self.b0).squeeze(-1)


def topk_mean(a, b, k=10, chunk=1024):
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        s = a[i:i + chunk] @ b.T
        kk = min(k, s.shape[1])
        out[i:i + chunk] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []

    for ds, facet, prefix, head_f, json_f, lam_csls in CONFIGS:
        z = np.load(OUT / f"{prefix}_test.npz", allow_pickle=True)
        d = pd.read_parquet(OUT / f"{prefix}_test.meta.parquet")
        U, V = z["U"], z["V"]
        head = Off(); head.load_state_dict(torch.load(OUT / head_f, map_location="cpu"))
        head.eval()
        with torch.no_grad():
            mh = head(torch.tensor(U), torch.tensor(V)).numpy()
        lam = json.load(open(OUT / json_f, encoding="utf-8"))["lam_from_val"]

        cos = (U * V).sum(1)
        bank = np.load(OUT / f"bank_b16_{ds}.npz", allow_pickle=True)
        ri = topk_mean(U, bank["txt"])
        rc = topk_mean(V, bank["img"])

        d = d.reset_index(drop=True).assign(
            cos=cos, mh=mh, y=d.pair_label.to_numpy(),
            s_cos=-cos, s_crd=-(cos - lam * mh),
            s_csls=-(cos - lam_csls * (ri + rc)))

        # ---- 2AFC invariance (exact, cycle by cycle, SIMILARITY space) ----
        d = d.assign(sim_cos=cos, sim_crd=cos - lam * mh)
        max_err = 0.0
        ncyc = 0
        acc_cos = acc_crd = 0
        for cid, g in d.groupby("cycle_id"):
            if len(g) != 4:
                continue
            cl = g[g.y == 0]; no = g[g.y == 1]
            d0 = cl.sim_cos.sum() - no.sim_cos.sum()
            d1 = cl.sim_crd.sum() - no.sim_crd.sum()
            max_err = max(max_err, abs(d0 - d1))
            acc_cos += int(d0 > 0); acc_crd += int(d1 > 0); ncyc += 1
        print(f"   2AFC cosine {acc_cos/ncyc:.6f} vs CRD {acc_crd/ncyc:.6f} "
              f"| max|Δ'-Δ| = {max_err:.3e}", flush=True)

        # ---- clustered bootstrap ----
        gids = d.groupby("cycle_id").indices
        cyc_ids = list(gids)
        y = d.y.to_numpy()
        stats = {"crd-cos": [], "crd-csls": []}
        base = {"cos": roc_auc_score(y, d.s_cos), "crd": roc_auc_score(y, d.s_crd),
                "csls": roc_auc_score(y, d.s_csls)}
        for _ in range(args.boot):
            pick = rng.choice(len(cyc_ids), len(cyc_ids), replace=True)
            ii = np.concatenate([gids[cyc_ids[k]] for k in pick])
            yy = y[ii]
            if len(np.unique(yy)) < 2:
                continue
            a_cos = roc_auc_score(yy, d.s_cos.to_numpy()[ii])
            a_crd = roc_auc_score(yy, d.s_crd.to_numpy()[ii])
            a_csls = roc_auc_score(yy, d.s_csls.to_numpy()[ii])
            stats["crd-cos"].append(a_crd - a_cos)
            stats["crd-csls"].append(a_crd - a_csls)

        rec = {"dataset": ds, "facet": facet, "cycles": int(ncyc),
               "cosine": round(base["cos"], 4), "csls": round(base["csls"], 4),
               "crd": round(base["crd"], 4),
               "d_crd_cos": round(base["crd"] - base["cos"], 4),
               "ci_crd_cos": [round(float(np.percentile(stats["crd-cos"], 2.5)), 4),
                              round(float(np.percentile(stats["crd-cos"], 97.5)), 4)],
               "p_crd_gt_cos": round(float(np.mean(np.array(stats["crd-cos"]) > 0)), 3),
               "d_crd_csls": round(base["crd"] - base["csls"], 4),
               "ci_crd_csls": [round(float(np.percentile(stats["crd-csls"], 2.5)), 4),
                               round(float(np.percentile(stats["crd-csls"], 97.5)), 4)],
               "p_crd_gt_csls": round(float(np.mean(np.array(stats["crd-csls"]) > 0)), 3),
               "lam": lam, "lam_csls": lam_csls,
               "max_abs_2afc_change": float(f"{max_err:.2e}")}
        rows.append(rec)
        print(rec, flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "crd_bootstrap.csv", index=False, encoding="utf-8-sig")
    jdump({"rows": rows}, OUT / "crd_bootstrap.json")
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
