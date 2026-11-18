"""Experiment 3: does the MSE projection residual survive outside a 2x2 design?

FGBench's reciprocal cycles make every image and caption appear once per class,
which forces structural zero covariance between any additive score and the label.
The LEMoN-protocol corpus (random noun-matched caption swaps, independent rows)
has no such balance, so it is a cleaner test of whether the correction is doing
something real.

Compares, on the same rows and the same training budget:
    cosine, Huber with lambda_val, Huber with lambda=1, MSE with lambda=1.
Bootstrap resamples images, since captions share images.

Run:  python fgbench/experiments/crd_v2/external_check.py --frac 0.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fgbench.models.crd_backbone import Off  # noqa: E402

LEMON = ROOT / "results" / "fgbench" / "lemon_protocol"
OUT = ROOT / "results" / "fgbench" / "crd_v2"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT = 1000


def auc(y, x):
    return float(roc_auc_score(y, -x))


def ap(y, x):
    return float(average_precision_score(y, -x))


def train(U, V, z, loss_name, seed, steps=1500, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    m = Off(d=U.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    Ut = torch.tensor(U, device=DEV); Vt = torch.tensor(V, device=DEV)
    Mt = torch.tensor(z, device=DEV, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = torch.tensor(rng.integers(0, len(Ut), batch), device=DEV)
        p = m(Ut[idx], Vt[idx])
        loss = F.mse_loss(p, Mt[idx]) if loss_name == "mse" else F.huber_loss(p, Mt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


def pred(m, U, V):
    with torch.no_grad():
        return m(torch.tensor(U, device=DEV),
                 torch.tensor(V, device=DEV)).cpu().numpy().ravel()


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--frac", type=float, default=0.2)
    ap_.add_argument("--seeds", default="0,1,2")
    a = ap_.parse_args()

    d = pd.read_parquet(LEMON / f"rows_frac{a.frac:.2f}.parquet")
    z = np.load(LEMON / "features.npz", allow_pickle=True)
    img = dict(zip(z["img_ids"].astype(str), z["img_vec"]))
    txt = dict(zip(z["txt_keys"], z["txt_vec"]))
    U = np.stack([img[str(i)] for i in d.image_id])
    V = np.stack([txt[c] for c in d.caption])
    S = np.einsum("ij,ij->i", U, V)

    m_tr = (d.split == "train").to_numpy()
    m_va = (d.split == "val").to_numpy()
    m_te = (d.split == "test").to_numpy()
    mu, sd = float(S[m_tr].mean()), float(S[m_tr].std() + 1e-8)
    Z = (S - mu) / sd
    y_va, y_te = d.label[m_va].to_numpy(), d.label[m_te].to_numpy()

    rng = np.random.default_rng(0)
    rows, boot = [], []
    for seed in (int(s) for s in a.seeds.split(",")):
        res = {"frac": a.frac, "seed": seed}
        scores = {"cosine": Z[m_te]}
        for loss_name in ("huber", "mse"):
            m = train(U[m_tr], V[m_tr], Z[m_tr], loss_name, seed)
            qva, qte = pred(m, U[m_va], V[m_va]), pred(m, U[m_te], V[m_te])
            lam = max(np.round(np.arange(0, 2.01, 0.1), 1),
                      key=lambda L: auc(y_va, Z[m_va] - L * qva))
            scores[f"{loss_name}_val"] = Z[m_te] - lam * qte
            scores[f"{loss_name}_one"] = Z[m_te] - qte
            res[f"lam_{loss_name}"] = float(lam)
            res[f"auroc_{loss_name}_val_lam"] = round(
                auc(y_te, scores[f"{loss_name}_val"]), 4)
            res[f"auprc_{loss_name}_val_lam"] = round(
                ap(y_te, scores[f"{loss_name}_val"]), 4)
            res[f"auroc_{loss_name}_full"] = round(
                auc(y_te, scores[f"{loss_name}_one"]), 4)
            res[f"auprc_{loss_name}_full"] = round(
                ap(y_te, scores[f"{loss_name}_one"]), 4)
        res["auroc_cosine"] = round(auc(y_te, Z[m_te]), 4)
        res["auprc_cosine"] = round(ap(y_te, Z[m_te]), 4)
        rows.append(res)

        img_te = d.image_id[m_te].to_numpy()
        by = pd.Series(np.arange(len(y_te))).groupby(img_te).apply(
            lambda s: s.to_numpy()).to_dict()
        ids = list(by)
        for name, xa, xb in (
                # loss effect, holding the lambda strategy fixed
                ("MSE_val_vs_Huber_val", scores["mse_val"], scores["huber_val"]),
                ("MSE_one_vs_Huber_one", scores["mse_one"], scores["huber_one"]),
                # lambda effect, holding the loss fixed
                ("Huber_one_vs_val", scores["huber_one"], scores["huber_val"]),
                ("MSE_one_vs_val", scores["mse_one"], scores["mse_val"]),
                # deployed variants
                ("MSE_one_vs_Huber_val", scores["mse_one"], scores["huber_val"]),
                ("MSE_val_vs_cosine", scores["mse_val"], Z[m_te])):
            ds = []
            for _ in range(N_BOOT):
                pick = rng.choice(ids, len(ids), replace=True)
                ix = np.concatenate([by[i] for i in pick])
                yy = y_te[ix]
                if yy.min() == yy.max():
                    continue
                ds.append(auc(yy, xa[ix]) - auc(yy, xb[ix]))
            ds = np.asarray(ds)
            boot.append({"frac": a.frac, "seed": seed, "vs": name,
                         "delta_pp": round((auc(y_te, xa) - auc(y_te, xb)) * 100, 2),
                         "ci_lo": round(float(np.percentile(ds, 2.5)) * 100, 2),
                         "ci_hi": round(float(np.percentile(ds, 97.5)) * 100, 2),
                         "p": round(float((ds <= 0).mean()), 3)})
        print(res, flush=True)

    r = pd.DataFrame(rows)
    b = pd.DataFrame(boot)
    r.to_csv(OUT / f"external_lemon_{a.frac:.2f}.csv", index=False,
             encoding="utf-8-sig")
    b.to_csv(OUT / f"external_boot_{a.frac:.2f}.csv", index=False,
             encoding="utf-8-sig")
    cols = ["auroc_cosine", "auroc_huber_val_lam", "auroc_huber_full",
            "auroc_mse_val_lam", "auroc_mse_full",
            "auprc_cosine", "auprc_huber_val_lam", "auprc_mse_val_lam"]
    print("\n== mean over seeds ==")
    print(r[cols].mean().round(4).to_string())
    print("\n== bootstrap ==")
    print(b.groupby("vs")[["delta_pp", "ci_lo", "ci_hi"]].mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
