"""Operating-point audit: compare cosine and A at the SAME FPR.

Protocol
  1. split val by image into two halves;
  2. on one half pick the threshold that gives the target FPR (clean quantile);
  3. on the other half measure TPR and the confusion matrix;
  4. swap halves and average; repeat for three seeds.
This removes the per-model threshold calibration difference that made the
earlier FP/FN table incomparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402

SEEDS = (1, 2, 3)
LAM = 0.8
FPRS = (0.01, 0.05, 0.10)


def split_by_image(g, seed=0):
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    half = len(uniq) // 2
    a = set(uniq[perm[:half]].tolist())
    m = np.array([x in a for x in g])
    return m


def tpr_at_fpr(score_tr, y_tr, score_te, y_te, fpr):
    clean = score_tr[y_tr == 0]
    if len(clean) < 20:
        return None
    thr = np.quantile(clean, 1 - fpr)
    pred = score_te > thr
    tp = int(((pred == 1) & (y_te == 1)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())
    tn = int(((pred == 0) & (y_te == 0)).sum())
    return {"tpr": tp / max(1, tp + fn), "fp": fp, "fn": fn, "tp": tp, "tn": tn,
            "realised_fpr": fp / max(1, fp + tn)}


def main() -> int:
    out = {}
    for bench in ("coco", "flickr", "mmimdb"):
        per = {f: {"cos": [], "A": []} for f in FPRS}
        for s in SEEDS:
            z = np.load(RESULTS_DIR / "seed_audit" / f"seed_{s}" / f"val_scores_{bench}.npz",
                        allow_pickle=True)
            y, cos, rx, rc = z["label"], z["cos"], z["rx"], z["rc"]
            g = z["image_id"].astype(str)
            sc = {"cos": -cos, "A": -cos + LAM * (rx + rc)}
            m = split_by_image(g, seed=s)
            for f in FPRS:
                for name, v in sc.items():
                    vals = []
                    for tr, te in ((m, ~m), (~m, m)):
                        r = tpr_at_fpr(v[tr], y[tr], v[te], y[te], f)
                        if r:
                            vals.append(r)
                    if vals:
                        per[f][name].append({"tpr": float(np.mean([x["tpr"] for x in vals])),
                                             "fp": int(np.sum([x["fp"] for x in vals])),
                                             "fn": int(np.sum([x["fn"] for x in vals]))})
        block = {}
        for f in FPRS:
            c = [x["tpr"] for x in per[f]["cos"]]
            a = [x["tpr"] for x in per[f]["A"]]
            block[str(f)] = {
                "tpr_cos_mean": float(np.mean(c)), "tpr_cos_sd": float(np.std(c, ddof=1)),
                "tpr_A_mean": float(np.mean(a)), "tpr_A_sd": float(np.std(a, ddof=1)),
                "delta": float(np.mean(a) - np.mean(c)),
                "fp_cos": int(np.sum([x["fp"] for x in per[f]["cos"]])),
                "fp_A": int(np.sum([x["fp"] for x in per[f]["A"]])),
                "fn_cos": int(np.sum([x["fn"] for x in per[f]["cos"]])),
                "fn_A": int(np.sum([x["fn"] for x in per[f]["A"]])),
            }
        out[bench] = block
        print(bench, json.dumps(block, ensure_ascii=False), flush=True)
    write_json(RESULTS_DIR / "bench" / "operating_point.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
