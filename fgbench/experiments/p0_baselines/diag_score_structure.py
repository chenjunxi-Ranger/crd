"""Diagnosis: is there a predictable modality-specific additive component?

Part A - exact 2x2 factorial decomposition of the raw similarity inside every
reciprocal cycle:

        s_ij = m + alpha_i + beta_j + gamma_ij

    m       cycle level
    alpha_i image main effect   (image-conditioned marginal)
    beta_j  text  main effect   (text-conditioned marginal)
    gamma   interaction, i.e. what cannot be explained by the two marginals

    Identity verified below: the clean-minus-noisy sum equals
    sum(gamma over the two clean cells) - sum(gamma over the two noisy cells),
    i.e. the marginals cancel exactly at this level.

Part B - held-out R^2 of predicting the pair's own standardised score from the
features, for the image-only / text-only / additive heads of the A1 ablation.

Part C - do those heads actually recover the marginals? Correlation between the
image head output a(u) and the cycle's alpha, and between b(v) and beta.

Run:  python fgbench/experiments/p0_baselines/diag_score_structure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.models.crd_backbone import Off  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "p0_baselines"
FACETS = [("COCO", "Entity"), ("COCO", "Quantity"), ("VG", "Attribute"),
          ("VG", "Action"), ("VG", "Relation")]


def load(ds, facet, split):
    tag = f"clip_{ds}_{facet}"
    npz = CACHE / f"crdbb_{tag}_{split}.npz"
    if not npz.is_file():                      # Action keeps its own cache
        pref = f"crd_b16_{ds}_{facet}_{split}"
        npz = CACHE / f"{pref}.npz"
        meta = CACHE / f"{pref}.meta.parquet"
    else:
        meta = npz.with_suffix(".meta.parquet")
    z = np.load(npz, allow_pickle=True)
    return pd.read_parquet(meta), z["U"].astype(np.float32), z["V"].astype(np.float32)


def decompose(s: np.ndarray) -> dict:
    """s is the 2x2 matrix, rows = images, columns = captions."""
    m = s.mean()
    alpha = s.mean(1) - m
    beta = s.mean(0) - m
    gamma = s - m - alpha[:, None] - beta[None, :]
    return {"m": m, "alpha": alpha, "beta": beta, "gamma": gamma}


def part_a() -> pd.DataFrame:
    rows = []
    for ds, facet in FACETS:
        meta = pd.read_parquet(OUT / f"{ds}_{facet}_testmeta.parquet")
        z = np.load(OUT / f"{ds}_{facet}_scores.npz", allow_pickle=True)
        d = pd.DataFrame({"cycle": z["cycle_id"], "y": z["y"],
                          "img": meta.target_image_id.values,
                          "cap": meta.caption_id.values,
                          "s": z["raw_cos"]})
        recs = []
        for cid, g in d.groupby("cycle"):
            if len(g) != 4 or g.img.nunique() != 2 or g.cap.nunique() != 2:
                continue
            M = g.pivot_table(index="img", columns="cap", values="s", aggfunc="first")
            if M.isna().any().any():
                continue
            dec = decompose(M.to_numpy())
            clean = g[g.y == 0].s.sum()
            noisy = g[g.y == 1].s.sum()
            gm = dec["gamma"]
            # which cells are clean is given by the label, not by the pivot order
            ipos = {v: k for k, v in enumerate(M.index)}
            jpos = {v: k for k, v in enumerate(M.columns)}
            g_clean = sum(gm[ipos[r.img], jpos[r.cap]]
                          for r in g[g.y == 0].itertuples())
            g_noisy = sum(gm[ipos[r.img], jpos[r.cap]]
                          for r in g[g.y == 1].itertuples())
            recs.append({
                "alpha2": float((dec["alpha"] ** 2).sum()),
                "beta2": float((dec["beta"] ** 2).sum()),
                "gamma2": float((gm ** 2).sum()),
                "delta": float(clean - noisy),
                "gamma_delta": float(g_clean - g_noisy),
                "abs_alpha": float(np.abs(dec["alpha"]).mean()),
                "abs_beta": float(np.abs(dec["beta"]).mean()),
                "abs_gamma": float(np.abs(gm).mean()),
            })
        r = pd.DataFrame(recs)
        tot = r.alpha2 + r.beta2 + r.gamma2
        rows.append({
            "facet": facet, "cycles": len(r),
            "rms_alpha": float(np.sqrt(r.alpha2.mean() / 2)),
            "rms_beta": float(np.sqrt(r.beta2.mean() / 2)),
            "rms_gamma": float(np.sqrt(r.gamma2.mean() / 4)),
            "share_alpha": float(r.alpha2.sum() / tot.sum()),
            "share_beta": float(r.beta2.sum() / tot.sum()),
            "share_gamma": float(r.gamma2.sum() / tot.sum()),
            "identity_max_err": float((r.delta - r.gamma_delta).abs().max()),
        })
    return pd.DataFrame(rows)


def head_r2() -> pd.DataFrame:
    """Held-out R^2 for image-only / text-only / additive on VG Attribute."""
    ds, facet = "VG", "Attribute"
    meta_tr, Utr, Vtr = load(ds, facet, "train")
    _, Ute, Vte = load(ds, facet, "test")
    meta_te = pd.read_parquet(OUT / f"{ds}_{facet}_testmeta.parquet")
    S_tr = (Utr * Vtr).sum(1)
    mu, sd = float(S_tr.mean()), float(S_tr.std() + 1e-8)
    z_te = ((Ute * Vte).sum(1) - mu) / sd
    sst = float(((z_te - z_te.mean()) ** 2).sum())

    rows = []
    for parts in ("image", "text", "both"):
        for seed in (0, 1, 2):
            p = CACHE / f"crdbb_clip_{ds}_{facet}_{parts}_row_s{seed}_head.pt"
            if not p.is_file():
                continue
            ck = torch.load(p, map_location="cpu")
            head = Off(d=int(ck["dim"]), parts=parts)
            head.load_state_dict(ck["state"])
            head.eval()
            with torch.no_grad():
                pred = head(torch.tensor(Ute), torch.tensor(Vte)).numpy().ravel()
            sse = float(((z_te - pred) ** 2).sum())
            r2 = 1 - sse / sst
            # correlation with the corresponding marginal
            rows.append({"parts": parts, "seed": seed, "r2_test": round(r2, 4),
                         "pred": pred})
    out = pd.DataFrame([{k: v for k, v in r.items() if k != "pred"} for r in rows])
    return out, rows, meta_te, z_te


def part_c(rows, meta_te, ds="VG", facet="Attribute") -> pd.DataFrame:
    """Correlate head outputs with the per-cycle marginal effects."""
    z = np.load(OUT / f"{ds}_{facet}_scores.npz", allow_pickle=True)
    d = pd.DataFrame({"cycle": z["cycle_id"], "y": z["y"],
                      "img": meta_te.target_image_id.values,
                      "cap": meta_te.caption_id.values, "s": z["raw_cos"]})
    # per (cycle, image) marginal and per (cycle, caption) marginal
    img_m, cap_m, keep = {}, {}, []
    for cid, g in d.groupby("cycle"):
        if len(g) != 4 or g.img.nunique() != 2 or g.cap.nunique() != 2:
            continue
        M = g.pivot_table(index="img", columns="cap", values="s", aggfunc="first")
        dec = decompose(M.to_numpy())
        for i, im in enumerate(M.index):
            img_m[(cid, im)] = dec["alpha"][i]
        for j, cp in enumerate(M.columns):
            cap_m[(cid, cp)] = dec["beta"][j]
        keep.append(cid)
    mask = d.cycle.isin(keep).to_numpy()
    dd = d[mask].reset_index(drop=True)
    alpha = np.array([img_m[(c, i)] for c, i in zip(dd.cycle, dd.img)])
    beta = np.array([cap_m[(c, i)] for c, i in zip(dd.cycle, dd.cap)])

    res = []
    for r in rows:
        pred = r["pred"][np.isin(z["cycle_id"], keep)]
        tgt = alpha if r["parts"] == "image" else (beta if r["parts"] == "text" else None)
        if tgt is None:
            continue
        if len(pred) != len(tgt):
            continue
        res.append({"parts": r["parts"], "seed": r["seed"],
                    "pearson_r": round(float(np.corrcoef(pred, tgt)[0, 1]), 4)})
    return pd.DataFrame(res)


def main() -> int:
    a = part_a()
    print("== Part A: 2x2 factorial decomposition (raw cosine) ==")
    print(a.to_string(index=False))
    a.to_csv(OUT / "diag_decomposition.csv", index=False, encoding="utf-8-sig")

    b, rows, meta_te, z_te = head_r2()
    print("\n== Part B: held-out R^2 on VG Attribute test ==")
    print(b.groupby("parts").r2_test.agg(["mean", "std"]).round(4).to_string())
    b.to_csv(OUT / "diag_r2.csv", index=False, encoding="utf-8-sig")

    c = part_c(rows, meta_te)
    print("\n== Part C: head output vs cycle marginal ==")
    if len(c):
        print(c.groupby("parts").pearson_r.agg(["mean", "std"]).round(4).to_string())
        c.to_csv(OUT / "diag_marginal_corr.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
