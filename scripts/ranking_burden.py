"""Sample-level pairwise ranking burden (no thresholds).

For noisy i:  b_i = mean_{j clean} 1(S_i < S_j)          (ranked below a clean pair)
For clean i:  b_i = mean_{j noisy} 1(S_j < S_i)          (a noisy pair ranked above it)
d_i = b_i(cosine) - b_i(A)      >0 : A reduced this sample's ranking errors

Three seeds share the same val rows, so we deduplicate by sample_id and report
per-seed values, the mean delta, and whether the group is stable (>=2/3 seeds).
Exports candidate sheets for the blind semantic audit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402

SEEDS = (1, 2, 3)
LAM = 0.8
HIGH = 0.5          # burden above a coin flip = "wrong more often than right"


def burden(score, y):
    noisy = np.where(y == 1)[0]
    clean = np.where(y == 0)[0]
    b = np.zeros(len(y))
    sc, sn = score[clean], score[noisy]
    # noisy rows: fraction of clean rows ranked above them
    b[noisy] = (sn[:, None] < sc[None, :]).mean(1)
    # clean rows: fraction of noisy rows ranked below them
    b[clean] = (sn[None, :] < sc[:, None]).mean(1)
    return b


def main() -> int:
    out = {}
    audit_dir = RESULTS_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for bench in ("coco", "flickr", "mmimdb"):
        rows = None
        for s in SEEDS:
            z = np.load(RESULTS_DIR / "seed_audit" / f"seed_{s}" / f"val_scores_{bench}.npz",
                        allow_pickle=True)
            y, cos, rx, rc = z["label"], z["cos"], z["rx"], z["rc"]
            if rows is None:
                rows = pd.DataFrame({"sample_id": np.arange(len(y)),
                                     "image_id": z["image_id"].astype(str),
                                     "label": y})
            rows[f"b_cos_s{s}"] = burden(-cos, y)
            rows[f"b_A_s{s}"] = burden(-cos + LAM * (rx + rc), y)
        bcols_c = [f"b_cos_s{s}" for s in SEEDS]
        bcols_a = [f"b_A_s{s}" for s in SEEDS]
        rows["b_cos"] = rows[bcols_c].mean(1)
        rows["b_A"] = rows[bcols_a].mean(1)
        rows["delta"] = rows["b_cos"] - rows["b_A"]
        # "high burden" = top decile WITHIN the sample's own label group,
        # because the two label classes have different baseline burdens
        yv = rows["label"].to_numpy()
        thr = {}
        for name, col in (("cos", "b_cos"), ("A", "b_A")):
            vals = rows[col].to_numpy()
            for lab in (0, 1):
                m = yv == lab
                thr[(name, lab)] = float(np.quantile(vals[m], 0.9))
        bcos_v, bA_v = rows["b_cos"].to_numpy(), rows["b_A"].to_numpy()
        rows["hard_cos"] = np.where(yv == 1, bcos_v > thr[("cos", 1)],
                                    bcos_v > thr[("cos", 0)])
        rows["hard_A"] = np.where(yv == 1, bA_v > thr[("A", 1)],
                                  bA_v > thr[("A", 0)])
        rows["stable_hard"] = rows["hard_A"].astype(int)

        def group(r):
            if r["hard_cos"] and not r["hard_A"]:
                return "A_corrected"
            if r["hard_cos"] and r["hard_A"]:
                return "persistent_hard"
            if (not r["hard_cos"]) and r["hard_A"]:
                return "A_damaged"
            return "both_correct"

        rows["group"] = rows.apply(group, axis=1)
        out_thr = {f"{k[0]}_{k[1]}": v for k, v in thr.items()}
        rows.attrs = {}
        counts = rows["group"].value_counts().to_dict()
        stable = rows[rows["stable_hard"].astype(bool)]["group"].value_counts().to_dict()
        out[bench] = {"n": int(len(rows)), "groups": counts, "stable_hard": stable,
                      "mean_delta": float(rows["delta"].mean()),
                      "share_delta_pos": float((rows["delta"] > 0).mean()),
                      "thresholds": out_thr}
        print(bench, json.dumps(out[bench], ensure_ascii=False), flush=True)

        hard = rows[rows["group"] == "persistent_hard"].sort_values("b_A", ascending=False)
        hard.head(60).to_csv(audit_dir / f"{bench}_persistent_hard.csv", index=False,
                             encoding="utf-8-sig")
        rows[rows["group"] == "A_damaged"].sort_values("b_A", ascending=False).head(40)\
            .to_csv(audit_dir / f"{bench}_A_damaged.csv", index=False, encoding="utf-8-sig")
        rows[rows["group"] == "A_corrected"].sort_values("delta", ascending=False).head(40)\
            .to_csv(audit_dir / f"{bench}_A_corrected.csv", index=False, encoding="utf-8-sig")

    write_json(RESULTS_DIR / "bench" / "ranking_burden.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
