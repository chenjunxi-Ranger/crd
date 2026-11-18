"""CSLS + GMM baselines in the BLIP-ITC feature space (or any cached crdbb_* space)."""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fgbench"


def load(ds, facet, split, prefix="blipitc"):
    hits = sorted(glob.glob(str(OUT / f"crdbb_{prefix}_{ds}_{facet}*_{split}.npz")))
    if not hits:
        raise FileNotFoundError(f"no cache {ds}/{facet}/{split}")
    p = Path(hits[0])
    z = np.load(p, allow_pickle=True)
    d = pd.read_parquet(p.with_suffix(".meta.parquet"))
    return d, z["U"], z["V"]


def topk_mean(A, B, k=10, chunk=2048):
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        s = A[i:i + chunk] @ B.T
        kk = min(k, s.shape[1])
        out[i:i + chunk] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="blipitc")
    ap.add_argument("--bank", type=int, default=8000)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    for ds, facet in (("VG", "Attribute"), ("VG", "Relation"), ("COCO", "Quantity")):
        dtr, Utr, Vtr = load(ds, facet, "train", args.prefix)
        dva, Uva, Vva = load(ds, facet, "val", args.prefix)
        dte, Ute, Vte = load(ds, facet, "test", args.prefix)
        rng = np.random.default_rng(0)
        if args.bank and len(Utr) > args.bank:
            idx = rng.choice(len(Utr), args.bank, replace=False)
            Utr, Vtr = Utr[idx], Vtr[idx]

        cos_va = (Uva * Vva).sum(1)
        cos_te = (Ute * Vte).sum(1)

        ri_va = topk_mean(Uva, Vtr, args.k); rc_va = topk_mean(Vva, Utr, args.k)
        ri_te = topk_mean(Ute, Vtr, args.k); rc_te = topk_mean(Vte, Utr, args.k)

        yv, yt = dva.pair_label.to_numpy(), dte.pair_label.to_numpy()
        best = max(np.round(np.arange(0, 2.01, 0.1), 1),
                   key=lambda L: roc_auc_score(yv, -(cos_va - L * (ri_va + rc_va))))
        csls_te = cos_te - best * (ri_te + rc_te)

        gm = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(
            cos_va.reshape(-1, 1))
        noisy = int(np.argsort(gm.means_.ravel())[0])
        gmm_te = gm.predict_proba(cos_te.reshape(-1, 1))[:, noisy]

        def acc2afc(anom):
            df = pd.DataFrame({"c": dte.cycle_id.values, "y": yt, "s": anom})
            a = [g[g.y == 0].s.sum() < g[g.y == 1].s.sum()
                 for _, g in df.groupby("c") if len(g) == 4]
            return float(np.mean(a)) if a else float("nan")

        for name, anom in (("cosine", -cos_te), ("csls", -csls_te), ("gmm", gmm_te)):
            print({"dataset": ds, "facet": facet, "method": name, "lam": best,
                   "auroc": round(float(roc_auc_score(yt, anom)), 4),
                   "auprc": round(float(average_precision_score(yt, anom)), 4),
                   "acc2afc": round(acc2afc(anom), 6)}, flush=True)
        np.savez_compressed(OUT / f"score_{args.prefix}_csls_{ds}_{facet}.npz",
                            cycle_id=dte.cycle_id.values, y=yt, score=-csls_te,
                            cos=-cos_te, gmm=gmm_te)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
