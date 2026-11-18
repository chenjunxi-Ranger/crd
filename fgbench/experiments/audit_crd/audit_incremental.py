"""Experiment 1: do the two heads contribute independent predictive variance?

Non-zero std of q_I and q_T only shows each head *varies*; it does not show that
each adds information the other cannot supply. This script reports held-out R^2
for image-only, text-only and the joint additive prediction, plus the incremental
R^2 of each branch.

Intercepts: the joint head carries b0, so a fair comparison gives each partial
predictor its own intercept estimated on the TRAIN split (never on test).

Two sources of "image-only / text-only" are reported:
  (a) the sub-modules a(.) and b(.) of the jointly trained head
  (b) heads trained in isolation (parts="image" / "text" checkpoints)
Agreement between (a) and (b) indicates the decomposition is not an artefact of
joint training.

Run:  python fgbench/experiments/audit_crd/audit_incremental.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fgbench.models.crd_backbone import Off  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "audit_crd"
OUT.mkdir(parents=True, exist_ok=True)
FACETS = [("COCO", "Entity"), ("COCO", "Quantity"), ("VG", "Attribute"),
          ("VG", "Relation")]


def r2(z, pred):
    sst = float(((z - z.mean()) ** 2).sum())
    return 1.0 - float(((z - pred) ** 2).sum()) / sst


def load_head(path: Path, parts: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Off(d=int(ck["dim"]), parts=parts)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def sub(m, U, V):
    with torch.no_grad():
        qI = m.a(torch.tensor(U)).numpy().ravel() if m.wa else None
        qT = m.b(torch.tensor(V)).numpy().ravel() if m.wb else None
        b0 = float(m.b0)
    return qI, qT, b0


def main() -> int:
    rows = []
    for ds, facet in FACETS:
        for seed in (0, 1, 2):
            prov = json.loads(
                (CACHE / f"crdbb_prov_clip_{ds}_{facet}_both_row_s{seed}.json")
                .read_text(encoding="utf-8"))
            mu, sd = float(prov["mu"]), float(prov["sd"])

            def split(name):
                if name == "train":
                    cand = sorted(CACHE.glob(f"crdbb_clip_{ds}_{facet}_c*_train.npz"))
                    if not cand:
                        raise FileNotFoundError(
                            f"no train cache for {ds}/{facet}")
                    fn = cand[0]
                else:
                    fn = CACHE / f"crdbb_clip_{ds}_{facet}_{name}.npz"
                z = np.load(fn, allow_pickle=True)
                U, V = z["U"].astype(np.float32), z["V"].astype(np.float32)
                return U, V, ((U * V).sum(1) - mu) / sd

            Utr, Vtr, ztr = split("train")
            Ute, Vte, zte = split("test")

            joint = load_head(
                CACHE / f"crdbb_clip_{ds}_{facet}_both_row_s{seed}_head.pt", "both")
            qI_tr, qT_tr, b0 = sub(joint, Utr, Vtr)
            qI_te, qT_te, _ = sub(joint, Ute, Vte)

            # intercepts from TRAIN only
            cI = float((ztr - qI_tr).mean())
            cT = float((ztr - qT_tr).mean())

            r_i = r2(zte, qI_te + cI)
            r_t = r2(zte, qT_te + cT)
            r_j = r2(zte, qI_te + qT_te + b0)

            # isolated heads, if the ablation checkpoint exists
            iso = {}
            for part in ("image", "text"):
                p = CACHE / f"crdbb_clip_{ds}_{facet}_{part}_row_s{seed}_head.pt"
                if p.is_file():
                    m = load_head(p, part)
                    pr_tr, pt_tr, b_ = sub(m, Utr, Vtr)
                    pr_te, pt_te, _ = sub(m, Ute, Vte)
                    pred_tr = (pr_tr if pr_tr is not None else 0) + \
                              (pt_tr if pt_tr is not None else 0) + b_
                    pred_te = (pr_te if pr_te is not None else 0) + \
                              (pt_te if pt_te is not None else 0) + b_
                    iso[f"r2_iso_{part}"] = round(r2(zte, pred_te), 4)

            rows.append({
                "facet": facet, "seed": seed,
                "r2_image_only": round(r_i, 4),
                "r2_text_only": round(r_t, 4),
                "r2_joint": round(r_j, 4),
                "dR2_add_image": round(r_j - r_t, 4),
                "dR2_add_text": round(r_j - r_i, 4),
                "r2_sum_minus_joint": round((r_i + r_t) - r_j, 4),
                **iso,
            })
        print(f"{facet} done", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "incremental_r2.csv", index=False, encoding="utf-8-sig")
    agg = d.groupby("facet")[["r2_image_only", "r2_text_only", "r2_joint",
                              "dR2_add_image", "dR2_add_text",
                              "r2_sum_minus_joint"]].mean().round(4)
    print("\n== held-out R^2 and incremental R^2 (mean over 3 seeds) ==")
    print(agg.to_string())
    print("\nwrote", OUT / "incremental_r2.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
