"""Analytic additive-interaction decomposition of the CLIP cosine score.

No training, no labels, no Huber/MSE. The frozen, L2-normalised features are
already cached; every quantity below is a closed-form function of two means.

    s(u,v) = u'v
    a(u)   = (u - u_bar)' v_bar          image main effect
    b(v)   = u_bar' (v - v_bar)          text  main effect
    h(u,v) = (u - u_bar)' (v - v_bar)    image-text interaction
    mu     = u_bar' v_bar                global term

    s = mu + a + b + h                   exact identity, not a fit

Means use UNIQUE image ids and UNIQUE caption ids from the TRAIN split only; a
reciprocal cycle repeats every image and every caption twice, so averaging over
rows would double-weight them. Validation and test reuse the train means.

Detectors compared, all under the same convention (higher score = cleaner, i.e.
AUROC is computed as roc_auc_score(y_noisy, -score)):

    E0  s              raw cosine
    E1  a              image main effect
    E2  b              text main effect
    E3  h              interaction
    E4  a+b            additive marginals

Run:  python fgbench/experiments/decomposition/run_decomposition.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))
sys.path.insert(0, str(ROOT / "fgbench" / "experiments" / "p0_baselines"))

from run_baselines import FACETS, action_splits, load_cache  # noqa: E402

OUT = ROOT / "results" / "fgbench" / "decomposition"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 1000


def moments(U, V, img_ids, cap_ids):
    """Means over unique images / unique captions of the training split."""
    _, iu = np.unique(img_ids, return_index=True)
    _, iv = np.unique(cap_ids, return_index=True)
    u_bar = U[iu].mean(0)
    v_bar = V[iv].mean(0)
    return u_bar, v_bar, len(iu), len(iv)


def components(U, V, u_bar, v_bar):
    """Return mu, a, b, h, s for every row."""
    du, dv = U - u_bar, V - v_bar
    mu = float(u_bar @ v_bar)
    a = du @ v_bar
    b = dv @ u_bar
    h = np.einsum("ij,ij->i", du, dv)
    s = np.einsum("ij,ij->i", U, V)
    return mu, a, b, h, s


def auc(y, x):
    return float(roc_auc_score(y, -x))


def cyc_2afc(y, s, cyc):
    df = pd.DataFrame({"c": cyc, "y": y, "s": -s})
    acc = [g[g.y == 0].s.sum() < g[g.y == 1].s.sum()
           for _, g in df.groupby("c") if len(g) == 4]
    return float(np.mean(acc)) if acc else float("nan")


def holm(p):
    order = np.argsort(p)
    m, out, run = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def main() -> int:
    rng = np.random.default_rng(0)
    summary, mom_rows, boot_rows, samples = [], [], [], []

    for ds, facet, cycles in FACETS:
        t0 = time.time()
        if facet == "Action":
            (dtr, Utr, Vtr), (dva, Uva, Vva), (dte, Ute, Vte) = action_splits()
        else:
            dtr, Utr, Vtr = load_cache(ds, facet, "train", cycles)
            dva, Uva, Vva = load_cache(ds, facet, "val")
            dte, Ute, Vte = load_cache(ds, facet, "test")

        u_bar, v_bar, n_img, n_txt = moments(
            Utr, Vtr, dtr.target_image_id.values, dtr.caption_id.values)
        mu = float(u_bar @ v_bar)
        mom_rows.append({"facet": facet, "n_unique_images": n_img,
                         "n_unique_captions": n_txt, "mu": round(mu, 6),
                         "norm_u_bar": round(float(np.linalg.norm(u_bar)), 6),
                         "norm_v_bar": round(float(np.linalg.norm(v_bar)), 6)})

        # identity check on every split
        err = 0.0
        for (d, U, V) in ((dtr, Utr, Vtr), (dva, Uva, Vva), (dte, Ute, Vte)):
            m_, a_, b_, h_, s_ = components(U, V, u_bar, v_bar)
            err = max(err, float(np.abs(s_ - (m_ + a_ + b_ + h_)).max()))
        assert err < 1e-4, f"identity broken on {facet}: max err {err}"

        mu_te, a_te, b_te, h_te, s_te = components(Ute, Vte, u_bar, v_bar)
        y = dte.pair_label.to_numpy()
        cyc = dte.cycle_id.values

        scores = {"E0_cosine": s_te, "E1_image_main": a_te, "E2_text_main": b_te,
                  "E3_interaction": h_te, "E4_additive": a_te + b_te}
        for name, x in scores.items():
            summary.append({"facet": facet, "detector": name,
                            "auroc": round(auc(y, x), 4),
                            "auprc": round(float(average_precision_score(y, -x)), 4),
                            "acc2afc": round(cyc_2afc(y, x, cyc), 4),
                            "n_test": int(len(y))})

        # ---- paired cycle bootstrap: E3 vs E0
        ids = pd.unique(cyc)
        idx_by_cyc = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
            lambda s: s.to_numpy()).to_dict()
        deltas = []
        for _ in range(N_BOOT):
            pick = rng.choice(ids, len(ids), replace=True)
            idx = np.concatenate([idx_by_cyc[c] for c in pick])
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            deltas.append(auc(yy, h_te[idx]) - auc(yy, s_te[idx]))
        deltas = np.asarray(deltas)
        d_pt = auc(y, h_te) - auc(y, s_te)
        boot_rows.append({
            "facet": facet,
            "auroc_E0": round(auc(y, s_te), 4),
            "auroc_E3": round(auc(y, h_te), 4),
            "delta_pp": round(d_pt * 100, 2),
            "ci_lo": round(float(np.percentile(deltas, 2.5)) * 100, 2),
            "ci_hi": round(float(np.percentile(deltas, 97.5)) * 100, 2),
            "p_E3_better": float((deltas <= 0).mean()),
            "p_E3_worse": float((deltas >= 0).mean()),
        })

        # ---- per-sample dump
        samples.append(pd.DataFrame({
            "split": "test", "cycle_id": cyc, "image_id": dte.target_image_id.values,
            "caption_id": dte.caption_id.values, "label": y,
            "raw_cosine": s_te, "image_main_a": a_te, "text_main_b": b_te,
            "interaction_h": h_te, "additive_ab": a_te + b_te, "mu": mu_te,
        }))
        print(f"{facet:16s} E0 {auc(y, s_te):.4f}  E3 {auc(y, h_te):.4f}  "
              f"Δ {d_pt*100:+.2f}pp  err {err:.2e}  {time.time()-t0:.0f}s",
              flush=True)

    s_df = pd.DataFrame(summary)
    b_df = pd.DataFrame(boot_rows)
    # the observed effect is negative, so the meaningful tail is "E3 is worse";
    # both directions are corrected over the same family of five facets
    b_df["p_holm_better"] = holm(b_df.p_E3_better.to_numpy())
    b_df["p_holm_worse"] = holm(b_df.p_E3_worse.to_numpy())
    b_df["verdict"] = np.where(b_df.p_holm_worse < 0.05, "E3 worse",
                               np.where(b_df.p_holm_better < 0.05, "E3 better",
                                        "tie"))

    s_df.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mom_rows).to_csv(OUT / "moments.csv", index=False,
                                  encoding="utf-8-sig")
    b_df.to_csv(OUT / "bootstrap_E3_vs_E0.csv", index=False, encoding="utf-8-sig")
    pd.concat(samples, ignore_index=True).to_parquet(
        OUT / "samples_test.parquet", index=False)

    print("\n== AUROC by detector ==")
    print(s_df.pivot(index="detector", columns="facet", values="auroc").to_string())
    print("\n== E3 vs E0 ==")
    print(b_df.to_string(index=False))
    print("\n2AFC (cycle level) - E0 vs E3 must coincide:")
    print(s_df.pivot(index="detector", columns="facet", values="acc2afc").to_string())
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
