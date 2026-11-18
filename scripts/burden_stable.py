"""Seed-stable ranking-burden grouping + blind annotation packets.

Per seed r, per method m in {cos, A}, per label y:
    q_{r,m,y} = 0.9-quantile of b_i^{(r,m)} among samples with that label
    H_i^{(r,m)} = 1[ b_i^{(r,m)} >= q_{r,m,y_i} ]

persistent_hard : sum_r H^(r,cos) * H^(r,A) >= 2     (high burden under BOTH, twice)
A_corrected     : sum_r H^(r,cos) * (1-H^(r,A)) >= 2
A_damaged       : sum_r (1-H^(r,cos)) * H^(r,A) >= 2
both_low_burden    : everything else that is stable (>=2 seeds low on both)

Exports, per benchmark:
  results/audit/{bench}_{group}.csv     internal sheet (has group + label)
  results/audit/blind/{bench}_blind.csv annotator sheet (anonymous id, image, caption)
  results/audit/blind/{bench}_key.csv   private mapping (anonym -> sample/group)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, load_split, write_json   # noqa: E402

SEEDS = (1, 2, 3)
LAM = 0.8
Q = 0.9


def burden(score, y):
    noisy, clean = np.where(y == 1)[0], np.where(y == 0)[0]
    b = np.zeros(len(y))
    sc, sn = score[clean], score[noisy]
    b[noisy] = (sn[:, None] < sc[None, :]).mean(1)
    b[clean] = (sn[None, :] < sc[:, None]).mean(1)
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--benches", default="coco,flickr")
    args = ap.parse_args()

    blind_dir = RESULTS_DIR / "audit" / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for bench in args.benches.split(","):
        df = None
        rows = None
        for s in SEEDS:
            z = np.load(RESULTS_DIR / "seed_audit" / f"seed_{s}" / f"val_scores_{bench}.npz",
                        allow_pickle=True)
            y, cos, rx, rc = z["label"], z["cos"], z["rx"], z["rc"]
            if df is None:
                df = load_split(bench, "val")
                rows = pd.DataFrame({"sample_id": np.arange(len(y)),
                                     "image_id": z["image_id"].astype(str),
                                     "label": y})
            for name, score in (("cos", -cos), ("A", -cos + LAM * (rx + rc))):
                b = burden(score, y)
                thr = {lab: float(np.quantile(b[y == lab], Q)) for lab in (0, 1)}
                rows[f"b_{name}_s{s}"] = b
                rows[f"H_{name}_s{s}"] = np.where(
                    y == 1, b >= thr[1], b >= thr[0]).astype(int)

        hard = sum(rows[f"H_cos_s{s}"] * rows[f"H_A_s{s}"] for s in SEEDS)
        corrected = sum(rows[f"H_cos_s{s}"] * (1 - rows[f"H_A_s{s}"]) for s in SEEDS)
        damaged = sum((1 - rows[f"H_cos_s{s}"]) * rows[f"H_A_s{s}"] for s in SEEDS)
        rows["n_hard"] = hard
        rows["n_corrected"] = corrected
        rows["n_damaged"] = damaged
        rows["mean_b_cos"] = rows[[f"b_cos_s{s}" for s in SEEDS]].mean(1)
        rows["mean_b_A"] = rows[[f"b_A_s{s}" for s in SEEDS]].mean(1)
        rows["delta"] = rows["mean_b_cos"] - rows["mean_b_A"]

        rows["group"] = np.select(
            [hard >= 2, corrected >= 2, damaged >= 2],
            ["persistent_hard", "A_corrected", "A_damaged"],
            default="both_low_burden")

        counts = rows["group"].value_counts().to_dict()
        rows.to_csv(RESULTS_DIR / "audit" / f"{bench}_all_groups.csv", index=False,
                    encoding="utf-8-sig")
        blind_rows, key_rows = [], []
        rng = np.random.default_rng(0)
        for g in ("persistent_hard", "A_corrected", "A_damaged", "both_low_burden"):
            pool = rows[rows["group"] == g]
            # stratify by label, then dedupe images, then random sample
            picked = []
            for lab in (0, 1):
                sub = pool[pool["label"] == lab].drop_duplicates("image_id")
                share = len(sub) / max(1, len(pool.drop_duplicates("image_id")))
                want = int(round(args.per_group * share))
                if want:
                    picked.append(sub.sample(min(want, len(sub)), random_state=0))
            take = pd.concat(picked) if picked else pool.head(0)
            take = take.sample(min(args.per_group, len(take)), random_state=1)
            take = take.assign(bench=bench, group=g)
            take.to_csv(RESULTS_DIR / "audit" / f"{bench}_{g}.csv", index=False,
                        encoding="utf-8-sig")
            for _, r in take.iterrows():
                rec = df.iloc[int(r["sample_id"])]
                anon = f"{bench.upper()}-{len(blind_rows):04d}"
                blind_rows.append({"anon_id": anon, "image": rec["path"],
                                   "caption": rec["caption"]})
                key_rows.append({"anon_id": anon, "sample_id": int(r["sample_id"]),
                                 "image_id": r["image_id"], "label": int(r["label"]),
                                 "bench": bench, "group": g,
                                 "n_hard": int(r["n_hard"]),
                                 "mean_b_cos": float(r["mean_b_cos"]),
                                 "mean_b_A": float(r["mean_b_A"])})
        pd.DataFrame(blind_rows).to_csv(blind_dir / f"{bench}_blind.csv", index=False,
                                        encoding="utf-8-sig")
        pd.DataFrame(key_rows).to_csv(blind_dir / f"{bench}_key.csv", index=False,
                                      encoding="utf-8-sig")
        summary[bench] = {"n": int(len(rows)), "groups": counts,
                          "sampled": len(blind_rows),
                          "per_group": {g: int((rows["group"] == g).sum())
                                        for g in ("persistent_hard", "A_corrected",
                                                  "A_damaged", "both_low_burden")}}
        print(bench, summary[bench], flush=True)

    write_json(RESULTS_DIR / "bench" / "burden_stable.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
