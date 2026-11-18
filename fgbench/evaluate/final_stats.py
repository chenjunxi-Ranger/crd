"""Final statistics: 10k cycle bootstrap + Holm correction + threshold utility.

Threshold rules are fitted on VALIDATION only and then frozen for TEST:
  - t@1%FPR : 99th percentile of clean val scores
  - t@Youden: argmax(TPR - FPR) on val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
BANK_LAM = {"Attribute": 0.2, "Relation": 0.3, "Quantity": 0.6}


def topk_mean(a, b, k=10, chunk=1024):
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        s = a[i:i + chunk] @ b.T
        kk = min(k, s.shape[1])
        out[i:i + chunk] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def load_facet(ds, facet):
    """val/test tensors for one facet (B/16 caches)."""
    if ds == "VG" and facet == "Attribute":
        pref = "crdx_b16"
        va_p = OUT / "crd_b16_val.npz"
    else:
        pref = f"crd_b16_{ds}_{facet}"
        va_p = OUT / f"{pref}_val.npz"
    te_p = OUT / (f"{pref}_test.npz" if pref == "crdx_b16"
                  else f"{pref}_test.npz")
    out = {}
    for name, p in (("val", va_p), ("test", te_p)):
        z = np.load(p, allow_pickle=True)
        d = pd.read_parquet(p.with_suffix(".meta.parquet"))
        out[name] = (d, z["U"], z["V"])
    return out


def scores_for(bundle, ds, facet):
    bank = np.load(OUT / f"bank_b16_{ds}.npz", allow_pickle=True)
    res = {}
    for split, (d, U, V) in bundle.items():
        cos = (U * V).sum(1)
        ri = topk_mean(U, bank["txt"]); rc = topk_mean(V, bank["img"])
        res[split] = {"cos": cos,
                      "csls": cos - BANK_LAM[facet] * (ri + rc)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--target", default="row", choices=["cycle", "row"],
                    help="which CRD supervision target's score files to use")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows, tests = [], []

    for ds, facet in (("VG", "Attribute"), ("VG", "Relation"), ("COCO", "Quantity")):
        bundle = load_facet(ds, facet)
        sc = scores_for(bundle, ds, facet)

        # CRD test scores: official both-parts, three seeds averaged
        seeds = [0, 1, 2]
        crd_te, crd_va = [], []
        for s in seeds:
            p = OUT / f"crd_scores_b16_{ds}_{facet}_both_{args.target}_s{s}.npz"
            if not p.is_file():
                p = OUT / f"crd_scores_b16_{ds}_{facet}_both_s{s}.npz"
            z = np.load(p, allow_pickle=True)
            crd_te.append(z["crd"])
            if "va_crd" in z:
                crd_va.append(z["va_crd"])
        crd_te = np.mean(crd_te, axis=0)
        crd_va = np.mean(crd_va, axis=0) if crd_va else None

        dte = bundle["test"][0].reset_index(drop=True)
        dva = bundle["val"][0].reset_index(drop=True)
        yt = dte.pair_label.to_numpy()
        yv = dva.pair_label.to_numpy()

        # ---- threshold utility (val-fitted, test-applied) ----
        def thr_report(name, s_tr, s_te):
            nonlocal rows
            clean = s_tr[yv == 0]
            t99 = float(np.quantile(clean, 0.01))
            # Youden
            grid = np.quantile(s_tr, np.linspace(0.80, 0.995, 400))
            best, bt = -1, None
            for t in grid:
                tp = int(((s_tr >= t) & (yv == 1)).sum())
                fp = int(((s_tr >= t) & (yv == 0)).sum())
                fn = int(((s_tr < t) & (yv == 1)).sum())
                tpr = tp / max(1, tp + fn); fpr = fp / max(1, fp + int(((s_tr < t) & (yv == 0)).sum()))
                if tpr - fpr > best:
                    best, bt = tpr - fpr, float(t)
            pred = s_te >= bt
            tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum())
            fn = int((~pred & (yt == 1)).sum())
            prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            tpr1 = float((s_te[yt == 1] >= t99).mean())
            fpr1_test = float((s_te[yt == 0] >= t99).mean())   # real test FPR at the val threshold
            rows.append({"dataset": ds, "facet": facet, "method": name,
                         "auroc": round(float(roc_auc_score(yt, s_te)), 4),
                         "auprc": round(float(average_precision_score(yt, s_te)), 4),
                         "tpr@1%fpr": round(tpr1, 4),
                         "fpr_at_that_thr": round(fpr1_test, 4),
                         "f1": round(f1, 4), "precision": round(prec, 4),
                         "recall": round(rec, 4),
                         "thr_val_youden": round(bt, 4), "thr_val_1%": round(t99, 4)})

        # convert to ANOMALY space (higher = noisier) once, for both AUC and thresholds
        thr_report("cosine", -sc["val"]["cos"], -sc["test"]["cos"])
        thr_report("csls", -sc["val"]["csls"], -sc["test"]["csls"])
        if crd_va is not None:
            thr_report("crd", -crd_va, -crd_te)

        # ---- 10k clustered bootstrap on Delta AUROC ----
        cyc = dte.cycle_id.values
        gids = pd.Series(np.arange(len(yt))).groupby(cyc).apply(lambda x: x.to_numpy()).to_dict()
        keys = list(gids)
        base = {"cosine": -sc["test"]["cos"], "csls": -sc["test"]["csls"],
                "crd": -crd_te}
        boot = {k: [] for k in ("cosine", "csls")}
        for _ in range(args.boot):
            pick = rng.choice(len(keys), len(keys), replace=True)
            ii = np.concatenate([gids[keys[t]] for t in pick])
            yy = yt[ii]
            if len(np.unique(yy)) < 2:
                continue
            a_crd = roc_auc_score(yy, base["crd"][ii])
            for k in boot:
                boot[k].append(a_crd - roc_auc_score(yy, base[k][ii]))
        for k, v in boot.items():
            v = np.array(v) * 100
            p = float(np.mean(v <= 0))          # one-sided p for CRD > baseline
            tests.append({"dataset": ds, "facet": facet, "baseline": k,
                          "delta_pp": round(float(v.mean()), 2),
                          "ci_lo": round(float(np.percentile(v, 2.5)), 2),
                          "ci_hi": round(float(np.percentile(v, 97.5)), 2),
                          "p_one_sided": round(max(p, 1.0 / args.boot), 5)})
            print(tests[-1], flush=True)

    tdf = pd.DataFrame(tests)
    tdf["p_holm"] = np.round(holm(tdf.p_one_sided.to_numpy()), 5)
    tdf["significant_holm"] = tdf.p_holm < 0.05
    print(tdf.to_string(index=False), flush=True)
    tdf.to_csv(OUT / "final_stats_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "final_stats_thresholds.csv", index=False,
                              encoding="utf-8-sig")
    jdump({"tests": tests, "thresholds": rows}, OUT / "final_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
