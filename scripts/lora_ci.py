"""Bootstrap CI for the v3.0 four-column results at the frozen lambda=0.8."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, THRESHOLDS, write_json   # noqa: E402

LAM = 0.8
N_BOOT = 2000


def main() -> int:
    out = {}
    for bench in ("coco", "flickr", "mmimdb"):
        z = np.load(RESULTS_DIR / "gate" / f"lora_scores_{bench}.npz")
        y, cos, h = z["y"], z["cos"], z["h"]
        s = -cos + LAM * h
        point = float(roc_auc_score(y, s))
        rng = np.random.default_rng(0)
        vals, diffs = [], []
        idx = np.arange(len(y))
        for _ in range(N_BOOT):
            b = rng.integers(0, len(y), len(y))
            if y[b].min() == y[b].max():
                continue
            a = roc_auc_score(y[b], s[b])
            vals.append(a)
            diffs.append(a - roc_auc_score(y[b], -cos[b]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        dlo, dhi = np.percentile(diffs, [2.5, 97.5])
        out[bench] = {"n": int(len(y)), "auc": point, "ci95": [float(lo), float(hi)],
                      "threshold": THRESHOLDS[bench],
                      "p_pass": float(np.mean(np.asarray(vals) >= THRESHOLDS[bench])),
                      "delta_vs_cosine": point - float(roc_auc_score(y, -cos)),
                      "delta_ci95": [float(dlo), float(dhi)]}
        print(bench, json.dumps(out[bench], ensure_ascii=False), flush=True)

    z = np.load(RESULTS_DIR / "gate" / "lora_scores_foil.npz")
    sp, sn, hp, hn = z["s_pos"], z["s_neg"], z["h_pos"], z["h_neg"]
    A_p, A_n = -sp + LAM * hp, -sn + LAM * hn
    ok = (A_n > A_p).astype(float)
    point = float(ok.mean())
    rng = np.random.default_rng(0)
    vals, dvals = [], []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(ok), len(ok))
        vals.append(ok[b].mean())
        dvals.append(ok[b].mean() - float((sn[b] < sp[b]).mean()))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    out["foil"] = {"n": int(len(ok)), "acc": point, "ci95": [float(lo), float(hi)],
                   "threshold": THRESHOLDS["foil"],
                   "p_pass": float(np.mean(np.asarray(vals) >= THRESHOLDS["foil"])),
                   "delta_vs_cosine": point - float((sn < sp).mean()),
                   "delta_ci95": [float(np.percentile(dvals, 2.5)),
                                  float(np.percentile(dvals, 97.5))]}
    print("foil", json.dumps(out["foil"], ensure_ascii=False), flush=True)
    write_json(RESULTS_DIR / "bench" / "lora_ci.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
