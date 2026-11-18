"""Recompute metrics on the STRICT rule subsets, reusing saved predictions.

Relation strict  = Tier A  (on|under, behind|in front of)
Relation explore = Tier B  (above|below, over|under, left of|right of)
Relation excluded= unverified (inside|outside)
Action  strict    = sitting on|standing on, walking on|running on
Action  cond.     = riding|walking
Action  excluded  = eating|drinking
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import BENCH, jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
TIER_A = {"on | under", "behind | in front of"}
TIER_B = {"above | below", "over | under", "left of | right of"}
UNVER = {"inside | outside"}
# NB: rule strings are sorted alphabetically, so "running on" precedes "walking on"
ACT_STRICT = {"sitting on | standing on", "running on | walking on"}
ACT_COND = {"riding | walking"}
ACT_EXCL = {"drinking | eating", "eating | drinking"}


def topk_mean(a, b, k=10, chunk=1024):
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        s = a[i:i + chunk] @ b.T
        kk = min(k, s.shape[1])
        out[i:i + chunk] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def metrics(y, s, cyc, boot=10000, rng=None):
    auc = roc_auc_score(y, -s)
    ap = average_precision_score(y, -s)
    df = pd.DataFrame({"c": cyc, "y": y, "s": -s})
    acc = [g[g.y == 0].s.sum() < g[g.y == 1].s.sum()
           for _, g in df.groupby("c") if len(g) == 4]
    return float(auc), float(ap), float(np.mean(acc)) if acc else float("nan")


def main() -> int:
    rng = np.random.default_rng(0)
    d = pd.read_parquet(BENCH / "test.parquet")
    rows = []

    # ---------- Relation ----------
    rel = d[d.dataset == "VG"]
    rel = rel[rel.facet == "Relation"].copy()
    rel["rule"] = [" | ".join(sorted(map(str, e["exclusive_pair"])))
                   if isinstance(e, dict) and e.get("exclusive_pair") is not None else None
                   for e in rel.evidence]
    z = np.load(OUT / "crd_b16_VG_Relation_test.npz", allow_pickle=True)
    bank = np.load(OUT / "bank_b16_VG.npz", allow_pickle=True)
    cos = (z["U"] * z["V"]).sum(1)
    ri = topk_mean(z["U"], bank["txt"]); rc = topk_mean(z["V"], bank["img"])
    csls = cos - 0.3 * (ri + rc)
    crds = []
    for s in (0, 1, 2):
        p = OUT / f"crd_scores_b16_VG_Relation_both_s{s}.npz"
        if not p.is_file():
            p = OUT / f"crd_scores_b16_VG_Relation_s{s}.npz"
        if p.is_file():
            crds.append(np.load(p, allow_pickle=True)["crd"])
    crd = np.mean(crds, axis=0) if crds else None
    assert len(rel) == len(cos), (len(rel), len(cos))

    for name, keep in (("TierA_strict", TIER_A), ("TierB_explore", TIER_B),
                       ("unverified", UNVER),
                       ("ALL", TIER_A | TIER_B | UNVER)):
        m = rel.rule.isin(keep).to_numpy()
        if m.sum() == 0:
            continue
        y = rel.pair_label.to_numpy()[m]; c = rel.cycle_id.to_numpy()[m]
        out = {"dataset": "VG", "facet": "Relation", "subset": name,
               "n_noisy": int((y == 1).sum()), "n_cycles": int(pd.Series(c).nunique())}
        out["auc_cos"] = round(metrics(y, cos[m], c)[0], 4)
        out["auc_csls"] = round(metrics(y, csls[m], c)[0], 4)
        out["acc2afc_cos"] = round(metrics(y, cos[m], c)[2], 4)
        if crd is not None:
            a, ap, acc = metrics(y, crd[m], c)
            out["auc_crd"] = round(a, 4)
            out["auprc_crd"] = round(ap, 4)
            out["acc2afc_crd"] = round(acc, 4)
            # bootstrap CRD - cosine / CSLS
            gids = pd.Series(np.arange(len(y))).groupby(c).apply(lambda x: x.to_numpy()).to_dict()
            keys = list(gids); bs = []
            for _ in range(2000):
                pk = rng.choice(len(keys), len(keys), replace=True)
                ii = np.concatenate([gids[keys[t]] for t in pk])
                yy = y[ii]
                if len(np.unique(yy)) < 2:
                    continue
                bs.append((roc_auc_score(yy, -crd[m][ii])
                           - roc_auc_score(yy, -cos[m][ii])) * 100)
            bs = np.array(bs)
            out["d_crd_cos_pp"] = round(float(bs.mean()), 2)
            out["d_ci"] = [round(float(np.percentile(bs, 2.5)), 2),
                           round(float(np.percentile(bs, 97.5)), 2)]
        rows.append(out)
        print(out, flush=True)

    # ---------- Action (CRD was never trained on this facet) ----------
    act = d[(d.dataset == "VG") & (d.facet == "Action")].copy()
    act["rule"] = [" | ".join(sorted(map(str, e["exclusive_pair"])))
                   if isinstance(e, dict) and e.get("exclusive_pair") is not None else None
                   for e in act.evidence]
    za = np.load(OUT / "crd_b16_VG_Action_test.npz", allow_pickle=True) \
        if (OUT / "crd_b16_VG_Action_test.npz").is_file() else None
    if za is not None:
        cosa = (za["U"] * za["V"]).sum(1)
        for name, keep in (("Strict", ACT_STRICT), ("Conditional", ACT_COND),
                           ("Excluded", {"eating | drinking"})):
            m = act.rule.isin(keep).to_numpy()
            if m.sum() == 0:
                continue
            a, ap, acc = metrics(act.pair_label.to_numpy()[m], cosa[m],
                                 act.cycle_id.to_numpy()[m])
            rows.append({"dataset": "VG", "facet": "Action", "subset": name,
                         "n_noisy": int((act.pair_label.to_numpy()[m] == 1).sum()),
                         "n_cycles": int(act.cycle_id[m].nunique()),
                         "auc_cos": round(a, 4), "auprc_cos": round(ap, 4),
                         "acc2afc_cos": round(acc, 4)})
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "strict_subset_metrics.csv", index=False, encoding="utf-8-sig")
    jdump({"rows": rows}, OUT / "strict_subset_metrics.json")
    print(df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
