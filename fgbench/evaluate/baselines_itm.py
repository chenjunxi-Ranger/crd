"""Baselines in the BLIP-ITM (fusion) score space.

- GMM     : fit 2-component mixture on the ITM score (monotone -> AUROC == raw)
- LEMoN*  : neighbourhood-agreement features from the dual towers, with the ITM
            score appended as an extra feature (hybrid, labelled as such)
CSLS/TRACED are NOT defined here: CSLS needs a dot-product embedding space and
TRACED would need ~25x fusion forwards (computationally prohibitive).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.evaluate.lemon_fgbench import build_features  # noqa: E402

OUT = ROOT / "results" / "fgbench"
FACETS = [("VG", "Attribute"), ("VG", "Relation"), ("COCO", "Quantity")]


def load(prefix, ds, facet, split):
    hits = sorted(glob.glob(str(OUT / f"crdbb_{prefix}_{ds}_{facet}*_{split}.npz")))
    p = Path(hits[0])
    z = np.load(p, allow_pickle=True)
    d = pd.read_parquet(p.with_suffix(".meta.parquet"))
    return d, z["U"], z["V"], z["S"]


def acc2afc(anom, y, cid):
    df = pd.DataFrame({"c": cid, "y": y, "s": anom})
    a = [g[g.y == 0].s.sum() < g[g.y == 1].s.sum()
         for _, g in df.groupby("c") if len(g) == 4]
    return float(np.mean(a)) if a else float("nan")


def main() -> int:
    for ds, facet in FACETS:
        dtr, Utr, Vtr, Str = load("blipitm", ds, facet, "train")
        dva, Uva, Vva, Sva = load("blipitm", ds, facet, "val")
        dte, Ute, Vte, Ste = load("blipitm", ds, facet, "test")
        y = dte.pair_label.to_numpy()
        cid = dte.cycle_id.values

        gm = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(
            Sva.reshape(-1, 1))
        noisy = int(np.argsort(gm.means_.ravel())[0])
        gmm = gm.predict_proba(Ste.reshape(-1, 1))[:, noisy]

        # LEMoN-style: neighbour features from the ITC towers + ITM score
        di, Utr_i, Vtr_i, _ = load("blipitc", ds, facet, "train")
        _, Ute_i, Vte_i, _ = load("blipitc", ds, facet, "test")
        li = sorted(glob.glob(str(OUT / f"crdbb_blipitc_{ds}_{facet}*_train.npz")))[0]
        dtr_i = pd.read_parquet(Path(li).with_suffix(".meta.parquet"))
        keys = [f"{u}|{c}" for u, c in zip(dtr_i.target_image_id,
                                           dtr_i.caption.astype(str))]
        lib_key = {k: i for i, k in enumerate(keys)}
        Ftr = build_features(Utr_i, Vtr_i, Utr_i, Vtr_i, K=32, exclude_self=True,
                             self_keys=keys, lib_keys=lib_key)
        Fte = build_features(Ute_i, Vte_i, Utr_i, Vtr_i, K=32)

        lr = LogisticRegression(max_iter=3000).fit(
            np.c_[Ftr, Str], dtr_i.pair_label.to_numpy())
        lemon = lr.predict_proba(np.c_[Fte, Ste])[:, 1]

        for name, anom in (("itm_raw", -Ste), ("gmm", gmm), ("lemon_hybrid", lemon)):
            print({"dataset": ds, "facet": facet, "method": name,
                   "auroc": round(float(roc_auc_score(y, anom)), 4),
                   "auprc": round(float(average_precision_score(y, anom)), 4),
                   "acc2afc": round(acc2afc(anom, y, cid), 6)}, flush=True)
        np.savez_compressed(OUT / f"score_itmspace_{ds}_{facet}.npz",
                            cycle_id=cid, y=y, gmm=gmm, lemon=lemon, itm=-Ste)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
