"""MM-IMDb under category-replacement noise: cosine / CSLS / LEMoN / CRD.

Different corpus and a different image domain (movie posters plus plot text)
from FGBench's COCO/VG photographs, with a full train/val/test split, so the
protocol-internal comparison is meaningful:

    train   fit CRD heads (label-free, row-level standardised cosine)
            fit the LEMoN-style logistic regression (uses train labels)
            provide the reference bank for CSLS
    val     choose lambda (CRD) and lambda (CSLS)
    test    scored once

GMM is not run: its posterior is a monotone function of the cosine, so its AUROC
equals the cosine by construction (fgbench/models/gmm_filter.py).
TRACED-style is not run: it needs a deletion trajectory over every caption.

Run:  python fgbench/experiments/external/run_mmimdb.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fgbench.models.crd_backbone import Off  # noqa: E402

DATA = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\derived\mmimdb"
            r"\mmimdb__cat__lvl0.40__seed0.csv")
OUT = ROOT / "results" / "fgbench" / "external"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT = 2000
K_LEMON = 32


def auc(y, x):
    return float(roc_auc_score(y, -x))


def ap(y, x):
    return float(average_precision_score(y, -x))


def encode(df: pd.DataFrame):
    p = OUT / "feat_mmimdb.pkl"
    if p.is_file():
        with open(p, "rb") as f:
            U, V = pickle.load(f)
        print(f"  cache: {U.shape[0]} rows", flush=True)
        return U, V
    from fgbench.models.crd_backbone import ClipBB
    bb = ClipBB()
    t0 = time.time()
    U = bb.images(df.path.tolist())
    print(f"  {len(U)} posters in {time.time()-t0:.0f}s", flush=True)
    V = bb.texts(df.sentence.tolist())
    print(f"  {len(V)} texts in {time.time()-t0:.0f}s", flush=True)
    with open(p, "wb") as f:
        pickle.dump((U, V), f)
    return U, V


def topk_mean(a, b, k=10, chunk=512):
    return np.concatenate([
        np.sort(a[s:s + chunk] @ b.T, axis=1)[:, -k:].mean(1)
        for s in range(0, len(a), chunk)])


def lemon_features(U, V, LI, LT, k=K_LEMON, chunk=512):
    out = np.zeros((len(U), 8), dtype=np.float32)
    for s in range(0, len(U), chunk):
        e = min(s + chunk, len(U))
        Si = U[s:e] @ LI.T
        St = V[s:e] @ LT.T
        ii = np.argpartition(-Si, k, axis=1)[:, :k]
        ti = np.argpartition(-St, k, axis=1)[:, :k]
        sc_i = np.einsum("cd,ckd->ck", V[s:e], LT[ii])
        sc_t = np.einsum("cd,ckd->ck", U[s:e], LI[ti])
        r = np.arange(e - s)
        out[s:e] = np.c_[(U[s:e] * V[s:e]).sum(1), Si[r, ii[:, 0]], St[r, ti[:, 0]],
                         sc_i.mean(1), sc_t.mean(1),
                         sc_i.max(1) - sc_i.min(1), sc_t.max(1) - sc_t.min(1),
                         sc_i.mean(1) - sc_t.mean(1)]
    return out


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--steps", type=int, default=1500)
    a = ap_.parse_args()

    df = pd.read_csv(DATA, encoding="utf-8-sig")
    print(f"{len(df)} rows, {df.image_id.nunique()} posters")
    print(df.groupby("split").agg(rows=("sentence", "size"),
                                  mis=("is_mislabel", "mean")).round(3).to_string())
    U, V = encode(df)
    S = np.einsum("ij,ij->i", U, V)
    y = df.is_mislabel.to_numpy()

    m_tr = (df.split == "train").to_numpy()
    m_va = (df.split == "val").to_numpy()
    m_te = (df.split == "test").to_numpy()
    y_va, y_te = y[m_va], y[m_te]
    mu, sd = float(S[m_tr].mean()), float(S[m_tr].std() + 1e-8)
    Z = (S - mu) / sd
    LI, LT = U[m_tr], V[m_tr]

    rows = []
    rows.append(("CLIP cosine", auc(y_te, Z[m_te]), ap(y_te, Z[m_te]), ""))

    # centred cosine (parameter-free control)
    Ub, Vb = U - U[m_tr].mean(0), V - V[m_tr].mean(0)
    Ub /= np.linalg.norm(Ub, axis=1, keepdims=True) + 1e-8
    Vb /= np.linalg.norm(Vb, axis=1, keepdims=True) + 1e-8
    Sc = np.einsum("ij,ij->i", Ub, Vb)
    Sc = (Sc - Sc[m_tr].mean()) / (Sc[m_tr].std() + 1e-8)
    rows.append(("Mean-centred cosine", auc(y_te, Sc[m_te]), ap(y_te, Sc[m_te]), ""))
    print("cosine/centred done", flush=True)

    # CSLS (grid widened to 3.0: it hit the boundary on Flickr30k)
    ri_va, rc_va = topk_mean(U[m_va], LT), topk_mean(V[m_va], LI)
    ri_te, rc_te = topk_mean(U[m_te], LT), topk_mean(V[m_te], LI)
    grid = np.round(np.arange(0, 6.01, 0.05), 2)
    lam = max(grid, key=lambda L: auc(y_va, Z[m_va] - L * (ri_va + rc_va)))
    s_csls = Z[m_te] - lam * (ri_te + rc_te)
    rows.append(("CLIP + CSLS", auc(y_te, s_csls), ap(y_te, s_csls),
                 f"lam={lam}" + (" (boundary)" if lam >= 6.0 else "")))
    print("csls", rows[-1][1], flush=True)

    # LEMoN-style
    Ftr = lemon_features(U[m_tr], V[m_tr], LI, LT)
    Fte = lemon_features(U, V, LI, LT)[m_te]
    clf = LogisticRegression(max_iter=2000).fit(Ftr, y[m_tr])
    p_te = clf.predict_proba(Fte)[:, 1]
    rows.append(("LEMoN-style", auc(y_te, -p_te), ap(y_te, -p_te), "LR, K=32"))
    print("lemon", rows[-1][1], flush=True)

    # CRD, 3 seeds
    scores, per_seed = {}, []
    for seed in (int(s_) for s_ in a.seeds.split(",")):
        torch.manual_seed(seed)
        m = Off(d=U.shape[1]).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        Ut = torch.tensor(U[m_tr], device=DEV); Vt = torch.tensor(V[m_tr], device=DEV)
        Mt = torch.tensor(Z[m_tr], device=DEV, dtype=torch.float32)
        rg = np.random.default_rng(seed)
        for _ in range(a.steps):
            i = torch.tensor(rg.integers(0, len(Ut), 256), device=DEV)
            loss = F.huber_loss(m(Ut[i], Vt[i]), Mt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            qva = m(torch.tensor(U[m_va], device=DEV),
                    torch.tensor(V[m_va], device=DEV)).cpu().numpy().ravel()
            qte = m(torch.tensor(U[m_te], device=DEV),
                    torch.tensor(V[m_te], device=DEV)).cpu().numpy().ravel()
        L = max(np.round(np.arange(0, 2.01, 0.1), 1),
                key=lambda x: auc(y_va, Z[m_va] - x * qva))
        crd = Z[m_te] - L * qte
        scores[seed] = crd
        per_seed.append({"seed": seed, "lam": float(L), "auroc": round(auc(y_te, crd), 4),
                         "auprc": round(ap(y_te, crd), 4)})
        print(f"  crd s{seed} auroc {per_seed[-1]['auroc']} lam {L}", flush=True)

    crd_mean = np.mean(list(scores.values()), axis=0)
    rows.append(("CRD", auc(y_te, crd_mean), ap(y_te, crd_mean),
                 f"lam={np.mean([p['lam'] for p in per_seed]):.2f}, 3 seeds"))

    # paired bootstrap: CRD vs cosine (rows are one-per-poster, so rows == images)
    rng = np.random.default_rng(0)
    base = Z[m_te]
    d = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y_te), len(y_te))
        yy = y_te[idx]
        if yy.min() == yy.max():
            continue
        d.append(auc(yy, crd_mean[idx]) - auc(yy, base[idx]))
    d = np.asarray(d)

    # paired bootstraps of CRD against every other scorer
    for nm, ref in (("centred", Sc[m_te]), ("csls", s_csls), ("lemon", -p_te)):
        dd = []
        r2 = np.random.default_rng(1)
        for _ in range(N_BOOT):
            idx = r2.integers(0, len(y_te), len(y_te))
            yy = y_te[idx]
            if yy.min() == yy.max():
                continue
            dd.append(auc(yy, crd_mean[idx]) - auc(yy, ref[idx]))
        dd = np.asarray(dd)
        print("CRD - %-8s %+.2f pp  95%% CI [%+.2f, %+.2f]  p=%.4f"
              % (nm, (auc(y_te, crd_mean) - auc(y_te, ref)) * 100,
                 np.percentile(dd, 2.5) * 100, np.percentile(dd, 97.5) * 100,
                 (dd <= 0).mean()), flush=True)

    res = pd.DataFrame(rows, columns=["method", "auroc", "auprc", "note"])
    res.to_csv(OUT / "mmimdb_methods.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(OUT / "mmimdb_test_scores.npz", y=y_te, cosine=base,
                        centred=Sc[m_te], csls=s_csls, lemon=-p_te, crd=crd_mean)
    pd.DataFrame(per_seed).to_csv(OUT / "mmimdb_crd_seeds.csv", index=False,
                                  encoding="utf-8-sig")
    print()
    print(res.to_string(index=False))
    print(f"\nCRD - cosine: {(auc(y_te, crd_mean) - auc(y_te, base))*100:+.2f} pp  "
          f"95% CI [{np.percentile(d,2.5)*100:+.2f}, {np.percentile(d,97.5)*100:+.2f}]  "
          f"p={(d <= 0).mean():.4f}")
    print(f"per-seed sd: {np.std([p['auroc'] for p in per_seed], ddof=1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
