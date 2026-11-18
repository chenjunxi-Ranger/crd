"""Main table on FGBench v1-strict: cosine / CSLS / LEMoN / TRACED / CRD."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))

from fgbench.common import jdump  # noqa: E402

OUT = ROOT / "scores"
STRICT = ROOT / "data" / "manifests"
def load_lams():
    """CSLS lambda per facet, selected on VALIDATION and then frozen."""
    p = OUT / "csls_val_selected_b16.csv"
    if p.is_file():
        d = pd.read_csv(p)
        return {r.facet: float(r.lambda_from_val) for r in d.itertuples()}
    return {"Entity": 0.5, "Quantity": 0.6, "Action": 0.4,
            "Attribute": 0.2, "Relation": 0.3}


LAM = load_lams()


def topk_mean(a, b, k=10, chunk=1024):
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        s = a[i:i + chunk] @ b.T
        kk = min(k, s.shape[1])
        out[i:i + chunk] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def metrics(y, s, cyc):
    auc = float(roc_auc_score(y, -s))
    ap = float(average_precision_score(y, -s))
    df = pd.DataFrame({"c": cyc, "y": y, "s": -s})
    acc = [g[g.y == 0].s.sum() < g[g.y == 1].s.sum()
           for _, g in df.groupby("c") if len(g) == 4]
    return auc, ap, (float(np.mean(acc)) if acc else float("nan"))


def load_tensors(facet, ds):
    """Return (meta, U, V) for the FULL v1 test order of this facet."""
    pref = "crdx_b16_test" if (ds, facet) == ("VG", "Attribute") else \
        f"crd_b16_{ds}_{facet}_test"
    p = OUT / f"{pref}.npz"
    z = np.load(p, allow_pickle=True)
    meta = pd.read_parquet(p.with_suffix(".meta.parquet"))
    return meta, z["U"], z["V"]


def main() -> int:
    te = pd.read_parquet(STRICT / "test.parquet")
    keep = set(te.cycle_id)
    rows = []

    for ds, facet in (("COCO", "Entity"), ("COCO", "Quantity"),
                      ("VG", "Attribute"), ("VG", "Action"), ("VG", "Relation")):
        strict_ids = set(te[(te.dataset == ds) & (te.facet == facet)].cycle_id)
        if not strict_ids:
            continue
        meta, U, V = load_tensors(facet, ds)
        cos = (U * V).sum(1)
        bank = np.load(OUT / f"bank_b16_{ds}.npz", allow_pickle=True)
        csls = None
        if facet in LAM:
            ri = topk_mean(U, bank["txt"]); rc = topk_mean(V, bank["img"])
            csls = cos - LAM[facet] * (ri + rc)
        base = pd.DataFrame({"cycle_id": meta.cycle_id.values,
                             "y": meta.pair_label.values, "cos": cos,
                             "csls": csls if csls is not None
                             else np.full(len(meta), np.nan)})
        base["rank"] = base.groupby("cycle_id").cumcount()
        base = base[base.cycle_id.isin(strict_ids)].reset_index(drop=True)
        print(f"  {ds}/{facet}: {len(base)} rows kept", flush=True)

        def align(z, col):
            """Scores are stored in the same manifest order as `base`; keep only the
            rows whose cycle survives the strict filter (order is preserved)."""
            mask = np.isin(z["cycle_id"], np.array(list(strict_ids)))
            arr = np.asarray(z[col])[mask]
            if len(arr) != len(base):
                raise ValueError(f"align mismatch {len(arr)} vs {len(base)} ({ds}/{facet})")
            return arr

        seeds = [0, 1, 2]                       # three seeds wherever available
        vals = []
        if facet == "Action":                   # internal-val protocol
            for s in seeds:
                p = OUT / f"crd_action_scores_s{s}.npz"
                if p.is_file():
                    vals.append(align(np.load(p, allow_pickle=True), "crd"))
        for s in ([] if vals else seeds):
            # standardised pipeline (official): crdbb_scores_<cfg>_s<seed>.npz
            for name in (f"crdbb_scores_clip_{ds}_{facet}_both_row_s{s}.npz",
                         f"crd_scores_b16_{ds}_{facet}_both_s{s}.npz",
                         f"crd_scores_b16_{ds}_{facet}_s{s}.npz"):
                p = OUT / name
                if p.is_file():
                    vals.append(align(np.load(p, allow_pickle=True), "crd"))
                    break
        crd_per_seed = []
        if vals:
            base["crd"] = np.mean(vals, axis=0)
            for v in vals:                       # per-seed metrics, averaged afterwards
                crd_per_seed.append(metrics(base.y.to_numpy(), v,
                                            base.cycle_id.to_numpy()))

        # per-row base scores so the paired bootstrap can reuse identical rows
        np.savez_compressed(OUT / f"strict_base_{ds}_{facet}.npz",
                            cycle_id=base.cycle_id.values, y=base.y.values,
                            cos=-base["cos"].to_numpy(),
                            csls=(-base["csls"].to_numpy()
                                  if not base["csls"].isna().all() else None))

        gmm_p = OUT / f"score_gmm_{ds}_{facet}.npz"
        if gmm_p.is_file():
            base["gmm"] = -align(np.load(gmm_p, allow_pickle=True), "score")

        for name in ("lemon", "traced"):
            # prefer the CLIP-space scores; the untagged files were overwritten by BLIP runs
            for fname in (f"score_{name}_clip_{ds}_{facet}.npz",
                          f"score_{name}_{ds}_{facet}.npz"):
                p = OUT / fname
                if p.is_file():
                    # these files store an ANOMALY score (P(noisy)); convert to similarity
                    base[name] = -align(np.load(p, allow_pickle=True), "score")
                    break

        for method in ("cos", "csls", "gmm", "lemon", "traced", "crd"):
            if method not in base.columns or base[method].isna().all():
                continue
            a, ap, acc = metrics(base.y.to_numpy(), base[method].to_numpy(),
                                 base.cycle_id.to_numpy())
            row = {"dataset": ds, "facet": facet, "method": method,
                   "n_noisy": int((base.y == 1).sum()),
                   "cycles": int(base.cycle_id.nunique()),
                   "auroc": round(a, 4), "auprc": round(ap, 4),
                   "acc2afc": round(acc, 4)}
            if method == "crd" and crd_per_seed:
                row["n_seeds"] = len(crd_per_seed)
                row["auroc_seedmean"] = round(float(np.mean([x[0] for x in crd_per_seed])), 4)
                row["auroc_seedsd"] = round(float(np.std([x[0] for x in crd_per_seed],
                                                         ddof=1)) if len(crd_per_seed) > 1 else 0.0, 4)
            rows.append(row)
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "strict_main_table.csv", index=False, encoding="utf-8-sig")
    jdump({"rows": rows}, OUT / "strict_main_table.json")
    print(df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
