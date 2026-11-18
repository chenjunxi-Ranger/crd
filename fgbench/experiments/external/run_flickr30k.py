"""External generalisation on Flickr30k under LEMoN-style random noise.

Why this benchmark: FGBench is COCO+VG with reciprocal 2x2 cycles. This corpus
is Flickr30k images, one row per image, 40% of captions swapped in from another
image (LEMoN "random" construction). Rows are independent and the label margins
are NOT balanced by construction, so both the corpus and the noise family differ
from FGBench.

Layer-1 protocol (retrain inside the external protocol):
    train  fit the additive heads on the pair's own standardised score (no labels)
    val    choose lambda by AUROC (labels used here only)
    test   score once

Baselines: raw cosine, mean-centred cosine (no training), CRD.

Run:  python fgbench/experiments/external/run_flickr30k.py --noise random
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fgbench.models.crd_backbone import Off  # noqa: E402

DATA = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\derived\flickr30k")
OUT = ROOT / "results" / "fgbench" / "external"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = np.round(np.arange(0, 2.01, 0.1), 1)
N_BOOT = 1000


def auc(y, x):
    return float(roc_auc_score(y, -x))


def ap(y, x):
    return float(average_precision_score(y, -x))


def encode(df: pd.DataFrame, tag: str):
    """Images are shared across noise types; captions are not."""
    img_cache = OUT / "feat_flickr_images.pkl"
    txt_cache = OUT / f"feat_flickr_txt_{tag}.pkl"
    if img_cache.is_file() and txt_cache.is_file():
        with open(img_cache, "rb") as f:
            U = pickle.load(f)
        with open(txt_cache, "rb") as f:
            V = pickle.load(f)
        print(f"  cache hit: {U.shape[0]} rows", flush=True)
        return U, V

    from fgbench.models.crd_backbone import ClipBB
    bb = ClipBB()
    t0 = time.time()
    if img_cache.is_file():
        with open(img_cache, "rb") as f:
            U = pickle.load(f)
        print(f"  image cache hit ({len(U)} rows)", flush=True)
    else:
        U = bb.images(df.path.tolist())
        print(f"  {len(U)} images in {time.time()-t0:.0f}s", flush=True)
        with open(img_cache, "wb") as f:
            pickle.dump(U, f)
    V = bb.texts(df.sentence.tolist())
    print(f"  {len(V)} captions in {time.time()-t0:.0f}s", flush=True)
    with open(txt_cache, "wb") as f:
        pickle.dump(V, f)
    return U, V


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--noise", default="random", choices=["random", "noun"])
    ap_.add_argument("--level", default="0.40")
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--steps", type=int, default=1500)
    a = ap_.parse_args()

    f = DATA / f"flickr30k__{a.noise}__lvl{a.level}__seed0.csv"
    df = pd.read_csv(f, encoding="utf-8-sig")
    print(f"{f.name}: {len(df)} rows, {df.image_id.nunique()} images")
    print(df.groupby("split").agg(rows=("sentence", "size"),
                                  mis=("is_mislabel", "mean")).round(3).to_string())

    U, V = encode(df, f"{a.noise}{a.level}")
    S = np.einsum("ij,ij->i", U, V)
    y = df.is_mislabel.to_numpy()

    m_tr = (df.split == "train").to_numpy()
    m_va = (df.split == "val").to_numpy()
    m_te = (df.split == "test").to_numpy()
    mu, sd = float(S[m_tr].mean()), float(S[m_tr].std() + 1e-8)
    Z = (S - mu) / sd
    y_va, y_te = y[m_va], y[m_te]

    # mean-centred cosine (no training, no labels)
    u_bar = U[m_tr].mean(0)
    v_bar = V[m_tr].mean(0)
    Uc = U - u_bar
    Vc = V - v_bar
    Uc /= np.linalg.norm(Uc, axis=1, keepdims=True) + 1e-8
    Vc /= np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-8
    Sc = np.einsum("ij,ij->i", Uc, Vc)

    rng = np.random.default_rng(0)
    rows = []
    for seed in (int(s) for s in a.seeds.split(",")):
        torch.manual_seed(seed)
        m = Off(d=U.shape[1]).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        Ut = torch.tensor(U[m_tr], device=DEV); Vt = torch.tensor(V[m_tr], device=DEV)
        Mt = torch.tensor(Z[m_tr], device=DEV, dtype=torch.float32)
        rg = np.random.default_rng(seed)
        t0 = time.time()
        for _ in range(a.steps):
            idx = torch.tensor(rg.integers(0, len(Ut), 256), device=DEV)
            loss = F.huber_loss(m(Ut[idx], Vt[idx]), Mt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            qva = m(torch.tensor(U[m_va], device=DEV),
                    torch.tensor(V[m_va], device=DEV)).cpu().numpy().ravel()
            qte = m(torch.tensor(U[m_te], device=DEV),
                    torch.tensor(V[m_te], device=DEV)).cpu().numpy().ravel()
        lam = float(GRID[int(np.argmax([auc(y_va, Z[m_va] - L * qva) for L in GRID]))])
        crd = Z[m_te] - lam * qte

        d = []
        idx_te = np.arange(int(m_te.sum()))
        for _ in range(N_BOOT):
            pick = rng.choice(idx_te, len(idx_te), replace=True)
            yy = y_te[pick]
            if yy.min() == yy.max():
                continue
            d.append(auc(yy, crd[pick]) - auc(yy, Z[m_te][pick]))
        d = np.asarray(d)

        rows.append({
            "noise": a.noise, "seed": seed, "lam": lam,
            "auroc_cos": round(auc(y_te, Z[m_te]), 4),
            "auroc_centered": round(auc(y_te, (Sc[m_te] - Sc[m_tr].mean())
                                        / Sc[m_tr].std()), 4),
            "auroc_crd": round(auc(y_te, crd), 4),
            "auprc_cos": round(ap(y_te, Z[m_te]), 4),
            "auprc_crd": round(ap(y_te, crd), 4),
            "delta_pp": round((auc(y_te, crd) - auc(y_te, Z[m_te])) * 100, 2),
            "ci_lo": round(float(np.percentile(d, 2.5)) * 100, 2),
            "ci_hi": round(float(np.percentile(d, 97.5)) * 100, 2),
            "p": round(float((d <= 0).mean()), 3),
        })
        print(f"  s{seed} cos {rows[-1]['auroc_cos']:.4f} "
              f"cent {rows[-1]['auroc_centered']:.4f} "
              f"crd {rows[-1]['auroc_crd']:.4f} lam {lam} "
              f"({time.time()-t0:.0f}s)", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"flickr_{a.noise}_{a.level}.csv", index=False,
             encoding="utf-8-sig")
    print("\n== mean over seeds ==")
    print(r[["auroc_cos", "auroc_centered", "auroc_crd", "auprc_cos",
             "auprc_crd", "delta_pp", "ci_lo", "ci_hi"]].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
