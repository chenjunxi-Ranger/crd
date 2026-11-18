"""AUC decomposition by subgroup + selection-bias audit for persistent_hard.

AUC = sum_{g,h} (n_g^+ * n_h^- / (N^+ N^-)) * AUC_{g,h}
where g is the group of the noisy sample and h the group of the clean sample.
Also reports within-hard paired bootstrap (image-clustered) for cos vs traced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402

B = 500


def pair_auc(pos, neg):
    """AUC of pos vs neg using mid-ranks."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.r_[pos, neg]
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    vals, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="coco,flickr,mmimdb")
    args = ap.parse_args()
    out = {}
    for bench in args.branches if hasattr(args, "branches") else args.benches.split(","):
        p = RESULTS_DIR / "gate" / f"traced_failure_{bench}.csv"
        if not p.is_file():
            print("skip", bench); continue
        d = pd.read_csv(p, encoding="utf-8-sig")
        if "group" not in d:
            gcol = [c for c in d.columns if c.startswith("group")][0]
            d = d.rename(columns={gcol: "group"})
        d["group"] = d["group"].fillna("both_low_burden")
        y = d["label"].to_numpy()
        d["grp"] = np.where(d["group"] == "persistent_hard", "hard", "easy")

        rec = {"n": int(len(d)), "n_hard": int((d.grp == "hard").sum()),
               "n_noisy": int((y == 1).sum()), "n_noisy_hard": int(((y == 1) & (d.grp == "hard")).sum())}

        for name, col in (("cos", "s0"), ("traced", "p_trace")):
            s = d[col].to_numpy()
            pos, neg = s[y == 1], s[y == 0]
            rec[f"auc_{name}"] = pair_auc(pos, neg)
            mat = {}
            cells = []
            Np, Nn = len(pos), len(neg)
            for g in ("hard", "easy"):
                for h in ("hard", "easy"):
                    mp = (y == 1) & (d.grp == g).to_numpy()
                    mn = (y == 0) & (d.grp == h).to_numpy()
                    a = pair_auc(s[mp], s[mn])
                    w = mp.sum() * mn.sum() / (Np * Nn)
                    mat[f"{g}+/{h}-"] = {"n_pos": int(mp.sum()), "n_neg": int(mn.sum()),
                                         "auc": a, "weight": float(w),
                                         "contrib": None if a is None else float(w * a)}
                    if a is not None:
                        cells.append(w * a)
            rec[f"cells_{name}"] = mat
            rec[f"sum_contrib_{name}"] = float(sum(cells))

        # within-hard paired bootstrap (image clusters)
        hard = d[d.grp == "hard"].reset_index(drop=True)
        yh = hard["label"].to_numpy()
        gid = hard.get("image_id", pd.Series(np.arange(len(hard)))).astype(str).to_numpy()
        c, t = hard["s0"].to_numpy(), hard["p_trace"].to_numpy()
        rng = np.random.default_rng(0)
        uq = np.unique(gid)
        by = {g: np.where(gid == g)[0] for g in uq}
        diffs = []
        for _ in range(B):
            pick = rng.choice(uq, len(uq), replace=True)
            idx = np.concatenate([by[g] for g in pick])
            if yh[idx].min() == yh[idx].max():
                continue
            diffs.append(roc_auc_score(yh[idx], t[idx]) - roc_auc_score(yh[idx], c[idx]))
        rec["hard_diff_traced_minus_cos"] = float(np.mean(diffs)) if diffs else None
        rec["hard_diff_ci"] = ([float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
                               if diffs else None)
        out[bench] = rec
        print(bench, {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in rec.items() if not k.startswith("cells_")}, flush=True)
    write_json(RESULTS_DIR / "gate" / "auc_decomp.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
