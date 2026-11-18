"""Official LEMoN adapter for the five FGBench facets.

The official LEMoN code is written for COCO/Flickr caption datasets. This script
keeps its feature definition (kNN image/text consistency + d1) and its score
formula, but reads FGBench's cached CLIP features and cycle splits. Hyperparameters
are fitted on FGBench validation rows with the official `maximize_metric`.

Output: results/fgbench/lemon_official_fgb.csv
Run (WSL):
  PYTHONPATH=.../_clip_pkg:.../LEMoN-main python fgb_lemon_official.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "experiments" / "p0_baselines"))
sys.path.insert(0, "/mnt/d/PyCharm/PythonProject/异常标签检测/LEMoN/LEMoN-main")
from run_baselines import FACETS, action_splits, load_cache  # noqa: E402
from lib.metrics import utils  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "p0_baselines"
K = 5


def knn_block(U, V, Utr, Vtr, split):
    d = U.shape[1]
    idx_u, idx_v = faiss.IndexFlatIP(d), faiss.IndexFlatIP(d)
    idx_u.add(Utr)
    idx_v.add(Vtr)
    kq = K + 1 if split == "train" else K
    Dn, In = idx_u.search(U, kq)
    Dm, Im = idx_v.search(V, kq)
    if split == "train":
        # query is in its own index; drop self
        Dn, In = Dn[:, 1:], In[:, 1:]
        Dm, Im = Dm[:, 1:], Im[:, 1:]
    d1 = 1.0 - (U * V).sum(1)

    def pair_dist(A, B):
        out = np.empty((len(A), K), np.float32)
        for i in range(0, len(A), 4096):
            out[i:i + 4096] = 1.0 - (A[i:i + 4096, None, :] *
                                     B[i:i + 4096]).sum(-1)
        return out

    dists_n = pair_dist(V, Vtr[In])
    dists_m = pair_dist(U, Utr[Im])
    dists_tr = 1.0 - (Utr * Vtr).sum(1)
    return {
        "sset": split, "idx": np.arange(len(U)), "d_1": d1,
        "dists_n": list(dists_n), "D_n": list(-Dn),
        "dists_tr_n": list(dists_tr[In]),
        "dists_m": list(dists_m), "D_m": list(-Dm),
        "dists_tr_m": list(dists_tr[Im]),
    }


def fit_hparams(val: pd.DataFrame):
    val = val.copy()
    val["is_mislabel"] = val["y"]
    grid = {
        "beta": np.arange(0, 100.01, 5),
        "gamma": np.arange(0, 100.01, 5),
        "tau_1": [0, 1, 5, 10],
        "tau_2": [0, 1, 5, 10],
    }
    hp, best_f1, thres = utils.maximize_metric(
        val, grid, [[0] * 6, [0.5] * 6, [1] * 6, [10] * 6],
        utils.optimize_f1_efficient, {})
    beta, gamma, t1n, t2n, t1m, t2m = hp
    return {"beta": beta, "gamma": gamma, "tau_1_n": t1n, "tau_2_n": t2n,
            "tau_1_m": t1m, "tau_2_m": t2m}, best_f1, thres


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--facet", default="all")
    args = ap.parse_args()
    recs = []
    for ds, facet, cycles in FACETS:
        if args.facet not in ("all", facet):
            continue
        if facet == "Action":
            (dtr, Utr, Vtr), (dva, Uva, Vva), (dte, Ute, Vte) = action_splits()
        else:
            dtr, Utr, Vtr = load_cache(ds, facet, "train", cycles)
            dva, Uva, Vva = load_cache(ds, facet, "val")
            dte, Ute, Vte = load_cache(ds, facet, "test")
        blocks = []
        for split, d, U, V in (("train", dtr, Utr, Vtr), ("val", dva, Uva, Vva),
                               ("test", dte, Ute, Vte)):
            b = knn_block(U, V, Utr, Vtr, split)
            b["y"] = d.pair_label.to_numpy()
            b["cycle_id"] = d.cycle_id.values
            blocks.append(b)
        df = pd.concat([pd.DataFrame({k: v for k, v in b.items()})
                        for b in blocks], ignore_index=True)
        val = df[df.sset == "val"].copy()
        hp, f1, thres = fit_hparams(val)
        score = utils.calc_scores_given_hparams_vectorized(df, hp)
        df["score"] = score
        test = df[df.sset == "test"]
        a = roc_auc_score(test.y, test.score)
        recs.append({"facet": facet, "n_test": int(len(test)),
                     "auroc_official": round(float(a), 4),
                     "val_f1": round(float(f1), 4), **hp})
        print(recs[-1], flush=True)
        df[["sset", "idx", "cycle_id", "y", "score"]].to_csv(
            OUT / f"lemon_official_{facet}.csv", index=False, encoding="utf-8-sig")

    out = pd.DataFrame(recs)
    out.to_csv(OUT / "lemon_official_fgb.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
