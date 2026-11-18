"""CRD v2 external gate: FGBench Attribute (CRD v1 + BCE interaction) -> five
external protocols from the main external table.

Everything is frozen at the source:

    mu, sigma, CRD head, lambda   from the FGBench VG Attribute run
    g                             BCE pairwise head on uvdot, 3 seeds
    eta                           selected on the FGBench Attribute val split

Nothing is refitted on a target: no target label, no target mean/std.

Scores (similarity space, higher = cleaner):

    cosine   s
    E1       (s - mu) / sigma - lambda * q_add
    E2       g
    E3       E1 + eta * g

Targets and metrics are the ones of the main external table: SugarCrepe
(within-pair accuracy), Flickr30k random / noun (row AUROC), MM-IMDb (row
AUROC), FOIL-COCO (within-pair accuracy).

Run:  python fgbench/experiments/external/run_v2_external.py --n-boot 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "fgbench" / "experiments" / "p0_baselines",
          ROOT / "fgbench" / "experiments" / "crd_v2"):
    sys.path.insert(0, str(p))

from run_additive_interaction import (  # noqa: E402
    ETA_GRID, auc, crd_v1, feats, load_split, train_head)
from run_probe_external import (  # noqa: E402
    EXT, load_flickr, load_foil, load_mmimdb, load_sugarcrepe, pair_acc)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HEADS = EXT / "v2_heads"
HEADS.mkdir(parents=True, exist_ok=True)
N_BOOT = 5000


def build_source(facet: str = "Attribute", seeds=(0, 1, 2)):
    """Train (or load) the three BCE heads and select eta on FGBench val."""
    mu, sd, lam, crd_sim = crd_v1(facet)
    tag = facet.lower()
    Utr, Vtr, _, mtr = load_split(facet, "train", mu, sd)
    Uva, Vva, _, mva = load_split(facet, "val", mu, sd)
    S1_va = crd_sim(Uva, Vva)
    ytr = mtr.pair_label.to_numpy().astype(np.int64)
    yva = mva.pair_label.to_numpy().astype(np.int64)
    Xtr, Xva = feats(Utr, Vtr, "uvdot"), feats(Uva, Vva, "uvdot")
    out = []
    for s in seeds:
        path = HEADS / f"v2_{tag}_bce_s{s}.pt"
        model, gm, gs, _, gva = train_head("uvdot", facet, s, "B_bce", Xtr, ytr,
                                           mtr, Xva, yva)
        torch.save({"state": model.state_dict(), "dim": int(Xtr.shape[1]),
                    "mu": gm, "sd": gs, "facet": facet, "seed": s}, path)
        va_auc = [auc(yva, S1_va + e * gva) for e in ETA_GRID]
        eta = float(ETA_GRID[int(np.argmax(va_auc))])
        out.append({"model": model, "gm": gm, "gs": gs, "eta": eta,
                    "val_eta": round(float(max(va_auc)), 4)})
        print(f"  seed {s}: eta {eta:.2f} | val E3 {max(va_auc):.4f}", flush=True)
    return mu, sd, lam, crd_sim, out


def row_metrics(y, scores):
    return {k: round(float(roc_auc_score(y, a)), 4) for k, a in scores.items()}


def pair_metrics(y, pair, scores):
    return {k: round(pair_acc(pair, y, a), 4) for k, a in scores.items()}


def boot_row(y, ref, alt, rng, n):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yy = y[i]
        if yy.min() == yy.max():
            continue
        d.append(roc_auc_score(yy, alt[i]) - roc_auc_score(yy, ref[i]))
    return np.asarray(d)


def boot_pair(pair, y, ref, alt, rng, n):
    def sides(a):
        df = pd.DataFrame({"p": pair, "y": y, "a": a})
        return df[df.y == 0].groupby("p").a.first(), df[df.y == 1].groupby("p").a.first()
    p0, n0 = sides(ref)
    p1, n1 = sides(alt)
    ids = np.asarray(p0.index.intersection(n0.index))
    d = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids), replace=True)
        d.append((p1[pick] < n1[pick]).mean() - (p0[pick] < n0[pick]).mean())
    return np.asarray(d)


def pack(rec, key, d):
    rec[f"d_{key}_pp"] = round(float(np.mean(d)) * 100, 2)
    rec[f"ci_{key}"] = [round(float(np.percentile(d, 2.5)) * 100, 2),
                        round(float(np.percentile(d, 97.5)) * 100, 2)]
    rec[f"p_{key}"] = round(min(1.0, 2 * min(float((d <= 0).mean()),
                                             float((d >= 0).mean()))), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--facet", default="Attribute", choices=["Attribute", "Entity"])
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    print(f"== source: FGBench {args.facet} ==", flush=True)
    mu, sd, lam, crd_sim, heads = build_source(args.facet)

    def scores(U, V):
        S = np.einsum("ij,ij->i", U, V)
        S1 = crd_sim(U, V)          # already standardised and calibrated
        X = torch.tensor(feats(U, V, "uvdot").astype(np.float32), device=DEV)
        gs = []
        with torch.no_grad():
            for h in heads:
                gs.append((h["model"](X).cpu().numpy() - h["gm"]) / h["gs"])
        E2 = np.mean(gs, axis=0)
        E3 = S1 + np.mean([h["eta"] * g for h, g in zip(heads, gs)], axis=0)
        return {"cosine": -S, "E1": -S1, "E2": -E2, "E3": -E3}

    recs = []

    def emit(name, protocol, y, U, V, pair=None):
        sc = scores(U, V)
        if pair is None:
            rec = {"target": name, "protocol": protocol, "n": int(len(y))}
            rec.update(row_metrics(y, sc))
            d31 = boot_row(y, sc["E1"], sc["E3"], rng, args.n_boot)
            d32 = boot_row(y, sc["E2"], sc["E3"], rng, args.n_boot)
        else:
            rec = {"target": name, "protocol": protocol, "n": int(len(y))}
            rec.update(pair_metrics(y, pair, sc))
            d31 = boot_pair(pair, y, sc["E1"], sc["E3"], rng, args.n_boot)
            d32 = boot_pair(pair, y, sc["E2"], sc["E3"], rng, args.n_boot)
        pack(rec, "E3_vs_E1", d31)
        pack(rec, "E3_vs_E2", d32)
        recs.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    print("\n== SugarCrepe ==", flush=True)
    df, U, V = load_sugarcrepe()
    emit("SugarCrepe-comp", "same-image 2-way",
         df.is_mislabel.to_numpy().astype(int), U, V, df.pair_id.to_numpy())

    for noise in ("random", "noun"):
        print(f"\n== Flickr30k {noise} ==", flush=True)
        df, U, V = load_flickr(noise)
        m = (df.split == "test").to_numpy()
        emit(f"Flickr30k-{noise}", "row-level",
             df.is_mislabel.to_numpy()[m].astype(int), U[m], V[m])

    print("\n== MM-IMDb ==", flush=True)
    df, U, V = load_mmimdb()
    m = (df.split == "test").to_numpy()
    emit("MM-IMDb-cat", "row-level",
         df.is_mislabel.to_numpy()[m].astype(int), U[m], V[m])

    print("\n== FOIL-COCO ==", flush=True)
    Uf, Vp, Vn = load_foil()
    n = len(Uf)
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    pair = np.r_[np.arange(n), np.arange(n)]
    emit("FOIL-COCO", "same-image 2-way", y,
         np.vstack([Uf, Uf]), np.vstack([Vp, Vn]), pair)

    df = pd.DataFrame(recs)
    df.to_csv(EXT / f"v2_external_{args.facet}.csv", index=False,
              encoding="utf-8-sig")
    (EXT / f"v2_external_{args.facet}.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8")
    cols = ["target", "protocol", "cosine", "E1", "E2", "E3",
            "d_E3_vs_E1_pp", "ci_E3_vs_E1", "p_E3_vs_E1",
            "d_E3_vs_E2_pp", "ci_E3_vs_E2", "p_E3_vs_E2"]
    print("\n== summary ==")
    print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
