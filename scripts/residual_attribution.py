"""A 的残余错误归因 + Flickr 高分层的稳定性复核。

Part 1: within each s-quartile, report n, share, pos/neg counts, and the
        held-out delta-AUC with an image-clustered bootstrap CI.
Part 2: for each benchmark, split the errors of A into
          (cosine already wrong)  vs  (cosine right, A wrong)
        and count how often r_I / r_c flipped a decision.
Part 3: FOIL group-level 2x2 (cosine correct x A correct) plus the |ds| bin.
No training, no test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402

SEEDS = (1, 2, 3)
LAM = 0.8


def fit_half(x, y, seed=0):
    """Split in half, fit on one, score the other (returned on the scored half)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    h = len(y) // 2
    tr, te = idx[:h], idx[h:]
    if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
        return None, None
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(x[tr], y[tr])
    return m.predict_proba(x[te])[:, 1], te


def main() -> int:
    out = {"bins": {}, "errors": {}, "foil": {}}

    # ---------------- part 1: Flickr (and peers) high-bin stability
    for bench in ("coco", "flickr", "mmimdb"):
        z = np.load(RESULTS_DIR / "seed_audit" / "seed_1" / f"val_scores_{bench}.npz",
                    allow_pickle=True)
        y, cos, rx, rc = z["label"], z["cos"], z["rx"], z["rc"]
        g = z["image_id"].astype(str)
        q = np.quantile(cos, [0.25, 0.5, 0.75])
        bins = np.digitize(cos, q)
        rows = []
        for b in range(4):
            sel = bins == b
            n = int(sel.sum())
            if n < 200 or y[sel].min() == y[sel].max():
                continue
            xs = np.column_stack([cos[sel], rx[sel]])
            xf = np.column_stack([cos[sel], rx[sel], rc[sel]])
            ys = y[sel]
            p1, te = fit_half(xs, ys)
            if p1 is None:
                continue
            p2, _ = fit_half(xf, ys)
            d = roc_auc_score(ys[te], p2) - roc_auc_score(ys[te], p1)
            gg = g[sel][te]
            rng = np.random.default_rng(0)
            uniq = np.unique(gg)
            by = {u: np.where(gg == u)[0] for u in uniq}
            vals = []
            for _ in range(1000):
                pick = rng.choice(uniq, len(uniq), replace=True)
                ii = np.concatenate([by[u] for u in pick])
                if ys[te][ii].min() == ys[te][ii].max():
                    continue
                vals.append(roc_auc_score(ys[te][ii], p2[ii]) -
                            roc_auc_score(ys[te][ii], p1[ii]))
            rows.append({"bin": b, "n": n, "share": n / len(y),
                         "pos": int(y[sel].sum()), "neg": int((1 - y[sel]).sum()),
                         "delta_auc_halfsplit": float(d),
                         "ci95": [float(np.percentile(vals, 2.5)),
                                  float(np.percentile(vals, 97.5))]})
        out["bins"][bench] = rows
        print("BINS", bench, json.dumps(rows, ensure_ascii=False), flush=True)

    # ---------------- part 2: error attribution (threshold at best-F1)
    for bench in ("coco", "flickr", "mmimdb"):
        cells = {"A_FN_cos_wrong": 0, "A_FN_cos_right": 0,
                 "A_FP_cos_wrong": 0, "A_FP_cos_right": 0}
        resc, hurt, tot = 0, 0, 0
        for s in SEEDS:
            z = np.load(RESULTS_DIR / "seed_audit" / f"seed_{s}" / f"val_scores_{bench}.npz",
                        allow_pickle=True)
            y, cos, rx, rc = z["label"], z["cos"], z["rx"], z["rc"]
            a = -cos + LAM * (rx + rc)
            c = -cos

            def best_thr(sc, yy):
                ths = np.quantile(sc, np.linspace(0.2, 0.8, 31))
                best, bt = -1, 0.0
                for t in ths:
                    p = sc > t
                    tp = ((p == 1) & (yy == 1)).sum()
                    fp = ((p == 1) & (yy == 0)).sum()
                    fn = ((p == 0) & (yy == 1)).sum()
                    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                    if f1 > best:
                        best, bt = f1, t
                return bt

            ta, tc = best_thr(a, y), best_thr(c, y)
            pa, pc = a > ta, c > tc
            tot += len(y)
            cells["A_FN_cos_wrong"] += int(((y == 1) & (~pa) & (~pc)).sum())
            cells["A_FN_cos_right"] += int(((y == 1) & (~pa) & pc).sum())
            cells["A_FP_cos_wrong"] += int(((y == 0) & (pa) & (~pc)).sum())
            cells["A_FP_cos_right"] += int(((y == 0) & (pa) & pc).sum())
            resc += int((pc != y).sum() - (pa != y).sum() * 0)  # placeholder
            hurt += int(((pc == y) & (pa != y)).sum())
            resc = int(((pc != y) & (pa == y)).sum()) + resc
        out["errors"][bench] = {"cells": cells, "n": tot,
                                "density_hurt": hurt, "density_rescue": resc}
        print("ERR", bench, json.dumps(out["errors"][bench], ensure_ascii=False), flush=True)

    # ---------------- part 3: FOIL group-level
    z = np.load(RESULTS_DIR / "gate" / "foil_signed_seed1.npz")
    ds, drc = z["ds"], z["drc"]
    a_ok = (ds - LAM * drc) > 0
    c_ok = ds > 0
    foil = {"both_right": int((c_ok & a_ok).sum()),
            "cos_right_A_wrong": int((c_ok & ~a_ok).sum()),
            "cos_wrong_A_right": int((~c_ok & a_ok).sum()),
            "both_wrong": int((~c_ok & ~a_ok).sum()),
            "n": int(len(ds))}
    bins = []
    q = np.quantile(np.abs(ds), [0.25, 0.5, 0.75])
    b = np.digitize(np.abs(ds), q)
    for k in range(4):
        sel = b == k
        bins.append({"bin": k, "n": int(sel.sum()),
                     "cos_right": float(c_ok[sel].mean()),
                     "A_right": float(a_ok[sel].mean()),
                     "flip": float((c_ok & ~a_ok)[sel].sum())})
    foil["bins"] = bins
    out["foil"] = foil
    print("FOIL", json.dumps(foil, ensure_ascii=False), flush=True)
    write_json(RESULTS_DIR / "bench" / "residual_attribution.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
