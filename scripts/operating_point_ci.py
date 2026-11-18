"""Paired image-clustered bootstrap for the delta-TPR at a fixed FPR."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402

LAM = 0.8
FPRS = (0.01, 0.05, 0.10)


def main() -> int:
    out = {}
    for bench in ("coco", "flickr", "mmimdb"):
        block = {}
        for fpr in FPRS:
            deltas = []
            for s in (1, 2, 3):
                z = np.load(RESULTS_DIR / "seed_audit" / f"seed_{s}" /
                            f"val_scores_{bench}.npz", allow_pickle=True)
                y, cos, rx, rc = z["label"], z["cos"], z["rx"], z["rc"]
                g = z["image_id"].astype(str)
                sc = {"cos": -cos, "A": -cos + LAM * (rx + rc)}
                uniq = np.unique(g)
                rng = np.random.default_rng(s)
                perm = rng.permutation(len(uniq))
                half = set(uniq[perm[:len(uniq) // 2]].tolist())
                m = np.array([x in half for x in g])
                for tr, te in ((m, ~m), (~m, m)):
                    thr = {k: np.quantile(v[tr][y[tr] == 0], 1 - fpr) for k, v in sc.items()}
                    gt = g[te]
                    uq = np.unique(gt)
                    idx = {u: np.where(gt == u)[0] for u in uq}
                    yy = y[te]
                    pc = (sc["cos"][te] > thr["cos"]).astype(int)
                    pa = (sc["A"][te] > thr["A"]).astype(int)
                    brng = np.random.default_rng(0)
                    vals = []
                    for _ in range(400):
                        pick = brng.choice(uq, len(uq), replace=True)
                        ii = np.concatenate([idx[u] for u in pick])
                        if yy[ii].sum() == 0:
                            continue
                        tc = pc[ii][yy[ii] == 1].mean()
                        ta = pa[ii][yy[ii] == 1].mean()
                        vals.append(ta - tc)
                    if vals:
                        deltas.append(float(np.mean(vals)))
            block[str(fpr)] = {"mean_delta": float(np.mean(deltas)),
                               "ci95": [float(np.percentile(deltas, 2.5)),
                                        float(np.percentile(deltas, 97.5))],
                               "n_dirs": len(deltas)}
        out[bench] = block
        print(bench, json.dumps(block, ensure_ascii=False), flush=True)
    write_json(RESULTS_DIR / "bench" / "operating_point_ci.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
