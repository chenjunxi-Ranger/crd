"""P0-1: Mean-centred cosine and All-but-the-top (ABTT) cosine baselines.

Both baselines only re-estimate feature statistics on the TRAINING split; no
gradient training is involved and no label is used. They are evaluated on the
five FGBench v1-strict facets with the same CLIP ViT-B/16 features and the same
test rows as the main experiment.

Statistics are fitted on UNIQUE image ids and UNIQUE caption ids, because a
reciprocal cycle repeats every image and every caption twice; fitting on rows
would silently double-weight them.

Cache convention (verified): U and V are already L2-normalised, U·V == S.
Order of operations used here:
    x_c = x - mean            (modality-specific training mean)
    x_p = x_c - P_k x_c       (ABTT only, first k principal directions removed)
    x_n = x_p / ||x_p||
    s   = x_n · y_n
`k` is selected on the validation split by AUROC; the test split never enters
any decision.

Run:  python fgbench/experiments/p0_baselines/run_baselines.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

BENCH = ROOT / "fgbench" / "data" / "fgbench_v1_strict"
CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "p0_baselines"
OUT.mkdir(parents=True, exist_ok=True)

VG_ROOT = Path("/mnt/d/PyCharm/PythonProject/异常标签检测/数据集/VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"

# (dataset, facet, train-cycle subset used by CRD)
FACETS = [("COCO", "Entity", 8000), ("COCO", "Quantity", 2500),
          ("VG", "Attribute", 8000), ("VG", "Action", 0),
          ("VG", "Relation", 2000)]
K_GRID = [1, 2, 4, 8, 16, 32]


# ----------------------------------------------------------------- loaders
def load_cache(ds: str, facet: str, split: str, cycles: int = 0):
    tag = f"clip_{ds}_{facet}" + (f"_c{cycles}" if cycles and split == "train" else "")
    npz = CACHE / f"crdbb_{tag}_{split}.npz"
    meta = npz.with_suffix(".meta.parquet")
    z = np.load(npz, allow_pickle=True)
    return pd.read_parquet(meta), z["U"].astype(np.float32), z["V"].astype(np.float32)


def action_splits():
    """Reproduce the internal train/val split of fgbench/models/crd_action.py."""
    sys.path.insert(0, str(ROOT))
    from fgbench.models.crd_action import VAL_FRAC  # noqa
    from fgbench.models.crd_backbone import ClipBB, DATA_SEED

    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()

    def rows(split):
        d = pd.read_parquet(BENCH / f"{split}.parquet")
        d = d[(d.dataset == "VG") & (d.facet == "Action")].copy()
        d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
        d = d[d.iid.isin(vgidx)]
        d = d[d.groupby("cycle_id").target_image_id.transform("nunique") == 2]
        return d.reset_index(drop=True)

    tr_all, te = rows("train"), rows("test")
    cyc2imgs = tr_all.groupby("cycle_id").iid.apply(set).to_dict()
    cids = np.array(sorted(cyc2imgs))
    rng = np.random.default_rng(DATA_SEED)
    rng.shuffle(cids)
    n_val = max(1, int(round(len(cids) * VAL_FRAC)))
    val_ids, train_ids = set(cids[:n_val]), set(cids[n_val:])
    train_imgs = set().union(*[cyc2imgs[c] for c in train_ids])
    for c in [c for c in list(val_ids) if cyc2imgs[c] & train_imgs]:
        val_ids.discard(c)
        train_ids.add(c)
    dtr = tr_all[tr_all.cycle_id.isin(train_ids)].reset_index(drop=True)
    dva = tr_all[tr_all.cycle_id.isin(val_ids)].reset_index(drop=True)

    bb = ClipBB()

    def encode(d):
        imgs = d.iid.unique()
        paths = [str(VG_ROOT / vgidx[i]) for i in imgs]
        U = bb.images(paths)
        pos = {i: k for k, i in enumerate(imgs)}
        return U[[pos[i] for i in d.iid]], bb.texts(d.caption.astype(str).tolist())

    print(f"  [action] internal split: train {len(train_ids)} / val {len(val_ids)} cycles",
          flush=True)
    Utr, Vtr = encode(dtr)
    Uva, Vva = encode(dva)
    Ute, Vte = encode(te)
    return (dtr, Utr, Vtr), (dva, Uva, Vva), (te, Ute, Vte)


def fit_stats(U: np.ndarray, V: np.ndarray, ids_u: np.ndarray, ids_v: np.ndarray):
    """Mean and principal directions per modality, fitted on unique ids only."""
    iu, iv = np.unique(ids_u, return_index=True), np.unique(ids_v, return_index=True)
    Xu, Xv = U[iu[1]], V[iv[1]]
    mu_u, mu_v = Xu.mean(0), Xv.mean(0)
    stats = {"mu_u": mu_u, "mu_v": mu_v, "n_uniq_img": int(len(Xu)),
             "n_uniq_txt": int(len(Xv))}
    for name, X in (("u", Xu), ("v", Xv)):
        _, _, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        stats[f"pc_{name}"] = vt.astype(np.float32)      # rows = principal dirs
    return stats


def project(X: np.ndarray, mu: np.ndarray, pc: np.ndarray | None, k: int) -> np.ndarray:
    Z = X - mu
    if pc is not None and k > 0:
        P = pc[:k]                                       # k x d
        Z = Z - (Z @ P.T) @ P
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(n, 1e-12)


def score(U, V, st, k):
    return (project(U, st["mu_u"], st["pc_u"], k) *
            project(V, st["mu_v"], st["pc_v"], k)).sum(1)


def auc(y, s):
    return float(roc_auc_score(y, -s))      # higher score = cleaner


# ----------------------------------------------------------------- main
def main() -> int:
    rows, per_sample = [], {}
    for ds, facet, cycles in FACETS:
        t0 = time.time()
        if facet == "Action":
            (dtr, Utr, Vtr), (dva, Uva, Vva), (dte, Ute, Vte) = action_splits()
        else:
            dtr, Utr, Vtr = load_cache(ds, facet, "train", cycles)
            dva, Uva, Vva = load_cache(ds, facet, "val")
            dte, Ute, Vte = load_cache(ds, facet, "test")

        st = fit_stats(Utr, Vtr, dtr.target_image_id.values, dtr.caption_id.values)
        yva, yte = dva.pair_label.to_numpy(), dte.pair_label.to_numpy()

        # ---- mean-centred cosine (no hyper-parameter, no training)
        s_va = score(Uva, Vva, st, 0)
        s_te = score(Ute, Vte, st, 0)
        c_auc = round(auc(yte, s_te), 4)
        rows.append({"facet": facet, "method": "centered", "k": 0,
                     "auroc": c_auc,
                     "auprc": round(float(average_precision_score(yte, -s_te)), 4),
                     "val_auroc": round(auc(yva, s_va), 4),
                     "n_test": int(len(yte))})
        per_sample[f"{facet}_centered"] = (dte, s_te)

        # ---- ABTT: k chosen on validation AUROC only
        best = None
        for k in K_GRID:
            if k >= min(Utr.shape[1], len(np.unique(dtr.target_image_id))) - 1:
                continue
            a = auc(yva, score(Uva, Vva, st, k))
            rows.append({"facet": facet, "method": f"abtt_k{k}", "k": k,
                         "auroc": None, "auprc": None,
                         "val_auroc": round(a, 4), "n_test": int(len(yte))})
            if best is None or a > best[1]:
                best = (k, a)
        k_best = best[0]
        s_te = score(Ute, Vte, st, k_best)
        for r in rows:
            if r["facet"] == facet and r["method"] == f"abtt_k{k_best}":
                r["auroc"] = round(auc(yte, s_te), 4)
                r["auprc"] = round(float(average_precision_score(yte, -s_te)), 4)
        per_sample[f"{facet}_abtt"] = (dte, s_te)

        print(f"{facet:16s} centered {c_auc:.4f} | "
              f"abtt k={k_best} {auc(yte, s_te):.4f} | {time.time()-t0:.0f}s",
              flush=True)

        s_raw = (Ute * Vte).sum(1)
        s_cent = score(Ute, Vte, st, 0)
        # keep the exact row order used for the per-sample scores
        dte[["cycle_id", "target_image_id", "caption_id", "pair_label"]].to_parquet(
            OUT / f"{ds}_{facet}_testmeta.parquet", index=False)
        np.savez_compressed(
            OUT / f"{ds}_{facet}_scores.npz",
            cycle_id=dte.cycle_id.values, y=yte,
            raw_cos=s_raw, centered=s_cent, abtt=s_te, k=np.array([k_best]),
            val_raw=(Uva * Vva).sum(1), val_centered=score(Uva, Vva, st, 0),
            val_abtt=score(Uva, Vva, st, k_best),
            va_cycle_id=dva.cycle_id.values, va_y=yva)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baselines_raw.csv", index=False, encoding="utf-8-sig")
    print()
    print(df[df.auroc.notna()].to_string(index=False))

    cfg = {"features": "CLIP ViT-B/16, cached U/V are L2-normalised (U·V == S)",
           "order": "centre -> (ABTT projection) -> L2 renormalise -> cosine",
           "fit": "training split only; mean and PCA fitted on UNIQUE image ids "
                  "and UNIQUE caption ids",
           "k_grid": K_GRID, "k_selection": "validation AUROC",
           "score_direction": "AUROC computed as roc_auc_score(y, -s)",
           "train_subset": {f: c for _, f, c in FACETS},
           "action_protocol": "internal train/val split reproduced from "
                              "fgbench/models/crd_action.py (data_seed=0, val_frac=0.2)"}
    (OUT / "baselines_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
