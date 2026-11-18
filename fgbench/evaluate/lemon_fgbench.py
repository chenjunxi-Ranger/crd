"""LEMoN-style baseline on FGBench: cross-modal neighbour agreement.

Mechanism-level reproduction (not the official code):
  library = TRAIN split (images + texts)
  for each pair (i, t):
    I-nbrs = K nearest library images to i  -> how well their texts score against t
    T-nbrs = K nearest library texts to t   -> how well their images score against i
    rank features = where does s(i,t) sit inside those neighbour score lists
  a LogisticRegression is fitted on TRAIN features and applied to TEST.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import BENCH, jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"


def build_features(U, V, LI, LT, K=32, chunk=512, exclude_self=False,
                   self_keys=None, lib_keys=None):
    """Neighbour-agreement features for every row of (U, V)."""
    n = len(U)
    out = np.zeros((n, 8), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Si = U[s:e] @ LI.T                       # (c, L) image -> library images
        St = V[s:e] @ LT.T                       # (c, L) text  -> library texts
        if exclude_self and self_keys is not None:
            for r, key in enumerate(self_keys[s:e]):
                j = lib_keys.get(key)
                if j is not None:
                    Si[r, j] = -1e4
                    St[r, j] = -1e4
        ii = np.argpartition(-Si, K, axis=1)[:, :K]     # top-K library images
        ti = np.argpartition(-St, K, axis=1)[:, :K]     # top-K library texts

        # score current text against the texts of the image-neighbours
        nbr_txt = LT[ii]                                # (c, K, d)
        sc_i = np.einsum("cd,ckd->ck", V[s:e], nbr_txt)
        # score current image against the images of the text-neighbours
        nbr_img = LI[ti]
        sc_t = np.einsum("cd,ckd->ck", U[s:e], nbr_img)

        s_self = (U[s:e] * V[s:e]).sum(1)
        out[s:e, 0] = s_self
        out[s:e, 1] = sc_i.max(1)
        out[s:e, 2] = sc_i.mean(1)
        out[s:e, 3] = sc_t.max(1)
        out[s:e, 4] = sc_t.mean(1)
        # rank of the true pair inside the neighbour score lists (1 = best)
        out[s:e, 5] = (sc_i > s_self[:, None]).sum(1) / K
        out[s:e, 6] = (sc_t > s_self[:, None]).sum(1) / K
        # neighbour agreement margin
        out[s:e, 7] = s_self - 0.5 * (sc_i.mean(1) + sc_t.mean(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="VG")
    ap.add_argument("--facet", default="Attribute")
    ap.add_argument("--tag", default="b16")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--train-limit", type=int, default=20000)
    ap.add_argument("--backbone", default="clip", choices=["clip", "blipitc"],
                    help="which cached feature space to run LEMoN-style in")
    args = ap.parse_args()

    def tensors(split):
        if args.backbone == "blipitc":
            import glob as _glob
            hits = sorted(_glob.glob(str(OUT / f"crdbb_blipitc_{args.dataset}_"
                                                 f"{args.facet}*_{split}.npz")))
            if hits:
                p = Path(hits[0])
                z = np.load(p, allow_pickle=True)
                return pd.read_parquet(p.with_suffix(".meta.parquet")), z["U"], z["V"]
            raise FileNotFoundError(f"no blipitc cache for {split}")
        # prefer the strict-manifest crdbb cache so the rows match the official test set
        cands = [OUT / f"crdbb_clip_{args.dataset}_{args.facet}_{split}.npz",
                 OUT / f"crd_{args.tag}_{args.dataset}_{args.facet}_{split}.npz",
                 OUT / f"crdx_{args.tag}_{split}.npz"]
        for p in cands:
            if p.is_file():
                z = np.load(p, allow_pickle=True)
                d = pd.read_parquet(p.with_suffix(".meta.parquet"))
                return d, z["U"], z["V"]
        raise FileNotFoundError(f"no cached tensors for {split}: {cands}")

    dtr, Utr, Vtr = tensors("train")
    dte, Ute, Vte = tensors("test")
    if args.train_limit and len(dtr) > args.train_limit:
        idx = np.random.default_rng(0).choice(len(dtr), args.train_limit, replace=False)
        dtr, Utr, Vtr = dtr.iloc[idx].reset_index(drop=True), Utr[idx], Vtr[idx]
    print(f"library {len(dtr)}  test {len(dte)}", flush=True)

    # library = train split
    LI, LT = Utr, Vtr
    lib_keys = {f"{u}|{v}": i for i, (u, v) in enumerate(zip(dtr.target_image_id,
                                                            dtr.caption.astype(str)))}
    self_keys = [f"{u}|{v}" for u, v in zip(dtr.target_image_id, dtr.caption.astype(str))]

    Ftr = build_features(Utr, Vtr, LI, LT, K=args.k, exclude_self=True,
                         self_keys=self_keys, lib_keys=lib_keys)
    Fte = build_features(Ute, Vte, LI, LT, K=args.k)
    print("features built", Ftr.shape, flush=True)

    lr = LogisticRegression(max_iter=3000).fit(Ftr, dtr.pair_label.to_numpy())
    p = lr.predict_proba(Fte)[:, 1]
    y = dte.pair_label.to_numpy()
    cos_auc = float(roc_auc_score(y, -Fte[:, 0]))
    auc = float(roc_auc_score(y, p))

    d2 = dte.assign(s=p)
    accs = []
    for cid, g in d2.groupby("cycle_id"):
        if len(g) != 4:
            continue
        accs.append(g[g.pair_label == 0].s.sum() < g[g.pair_label == 1].s.sum())
    res = {"dataset": args.dataset, "facet": args.facet, "k": args.k,
           "library": int(len(dtr)), "n_test": int(len(dte)),
           "cos_auc": round(cos_auc, 4), "lemon_auc": round(auc, 4),
           "delta": round(auc - cos_auc, 4),
           "auprc": round(float(average_precision_score(y, p)), 4),
           "acc2afc": round(float(np.mean(accs)), 4) if accs else None}
    print(res, flush=True)
    np.savez_compressed(OUT / f"score_lemon_{args.backbone}_{args.dataset}_"
                              f"{args.facet}.npz",
                        cycle_id=dte.cycle_id.values, y=y, score=p)
    jdump(res, OUT / f"lemon_{args.dataset}_{args.facet}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
