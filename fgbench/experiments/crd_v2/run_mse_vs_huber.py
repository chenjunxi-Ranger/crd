"""Experiment 2: Huber vs MSE under an identical architecture and budget.

Three variants, all with the same two-head additive model, the same data, the
same seeds and the same number of optimisation steps; only the loss and the
deployed combination differ.

    V1        Huber loss,  score = z - lambda_val * q     (what the paper does)
    V1-full   Huber loss,  score = z - q                  (lambda fixed to 1)
    V2        MSE   loss,  score = z - q                  (projection residual)

Rationale for V2: minimising E[(z - a(u) - b(v) - b0)^2] over the additive family
gives first-order conditions E[r|U] = 0 and E[r|V] = 0 at the population optimum,
i.e. the residual is orthogonal to any single-modality function. Huber only gives
E[psi_delta(r)|U] = 0, which does not imply a projection in L2.

Reported per facet and seed: held-out R^2, residual std ratio, AUROC/AUPRC of the
deployed score, and a cycle-clustered paired bootstrap of V2 against V1/V1-full.

Run:  python fgbench/experiments/crd_v2/run_mse_vs_huber.py --facets Attribute,Quantity
"""
from __future__ import annotations

import argparse
import json
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

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "crd_v2"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FACET_MAP = {"Entity": ("COCO", "Entity"), "Quantity": ("COCO", "Quantity"),
             "Attribute": ("VG", "Attribute"), "Relation": ("VG", "Relation")}
N_BOOT = 1000


def auc(y, x):
    return float(roc_auc_score(y, -x))


def ap(y, x):
    return float(average_precision_score(y, -x))


def r2(z, pred):
    return 1.0 - float(((z - pred) ** 2).sum()) / float(((z - z.mean()) ** 2).sum())


def load_facet(facet: str):
    ds, fc = FACET_MAP[facet]
    prov = json.loads(
        (CACHE / f"crdbb_prov_clip_{ds}_{fc}_both_row_s0.json").read_text("utf-8"))
    mu, sd = float(prov["mu"]), float(prov["sd"])

    def one(name):
        if name == "train":
            fn = sorted(CACHE.glob(f"crdbb_clip_{ds}_{fc}_c*_train.npz"))[0]
        else:
            fn = CACHE / f"crdbb_clip_{ds}_{fc}_{name}.npz"
        z = np.load(fn, allow_pickle=True)
        U, V = z["U"].astype(np.float32), z["V"].astype(np.float32)
        meta = pd.read_parquet(fn.with_suffix(".meta.parquet")) \
            if fn.with_suffix(".meta.parquet").is_file() else None
        return U, V, ((U * V).sum(1) - mu) / sd, meta

    return one("train"), one("val"), one("test")


def train_head(U, V, z, loss_name, seed, steps=1500, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    m = Off(d=U.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    Ut = torch.tensor(U, device=DEV); Vt = torch.tensor(V, device=DEV)
    Mt = torch.tensor(z, device=DEV, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = torch.tensor(rng.integers(0, len(Ut), batch), device=DEV)
        pred = m(Ut[idx], Vt[idx])
        loss = (F.mse_loss(pred, Mt[idx]) if loss_name == "mse"
                else F.huber_loss(pred, Mt[idx]))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


def predict(m, U, V):
    with torch.no_grad():
        return m(torch.tensor(U, device=DEV),
                 torch.tensor(V, device=DEV)).cpu().numpy().ravel()


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--facets", default="Attribute,Quantity")
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--steps", type=int, default=1500)
    ap_.add_argument("--tag", default="all")
    a = ap_.parse_args()

    rng = np.random.default_rng(0)
    rows, boot = [], []
    for facet in a.facets.split(","):
        (Utr, Vtr, ztr, _), (Uva, Vva, zva, _), (Ute, Vte, zte, mte) = load_facet(facet)
        print(f"\n{facet}: train {len(ztr)} rows, test {len(zte)}", flush=True)
        y_te = mte.pair_label.to_numpy()
        cyc = mte.cycle_id.values

        for seed in (int(s) for s in a.seeds.split(",")):
            t0 = time.time()
            out = {}
            for loss_name in ("huber", "mse"):
                m = train_head(Utr, Vtr, ztr, loss_name, seed, steps=a.steps)
                qva, qte = predict(m, Uva, Vva), predict(m, Ute, Vte)
                lam = max(np.round(np.arange(0, 2.01, 0.1), 1),
                          key=lambda L: auc(mte.pair_label.to_numpy(), zva - L * qva) if False
                          else 0)
                # lambda must be chosen on VALIDATION, with val labels only
                prov_val = pd.read_parquet(
                    sorted(CACHE.glob(f"crdbb_clip_{FACET_MAP[facet][0]}_"
                                      f"{FACET_MAP[facet][1]}_val.npz"))[0]
                    .with_suffix(".meta.parquet"))
                yva = prov_val.pair_label.to_numpy()
                lam = max(np.round(np.arange(0, 2.01, 0.1), 1),
                          key=lambda L: auc(yva, zva - L * qva))
                out[loss_name] = {"q_te": qte, "lam": float(lam)}

            res = {"facet": facet, "seed": seed}
            for loss_name in ("huber", "mse"):
                qte = out[loss_name]["q_te"]
                lam = out[loss_name]["lam"]
                res[f"r2_{loss_name}"] = round(r2(zte, qte), 4)
                res[f"resid_{loss_name}"] = round(float((zte - qte).std() / zte.std()), 4)
                res[f"lam_{loss_name}"] = lam
                res[f"auroc_{loss_name}_val_lam"] = round(auc(y_te, zte - lam * qte), 4)
                res[f"auroc_{loss_name}_full"] = round(auc(y_te, zte - qte), 4)
                res[f"auprc_{loss_name}_full"] = round(ap(y_te, zte - qte), 4)

            qh, qm = out["huber"]["q_te"], out["mse"]["q_te"]
            idx_by = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
                lambda s: s.to_numpy()).to_dict()
            ids = list(idx_by)
            for name, xa, xb in (("V2_vs_V1full", zte - qm, zte - qh),
                                 ("V2_vs_V1", zte - qm,
                                  zte - out["huber"]["lam"] * qh)):
                d = []
                for _ in range(N_BOOT):
                    pick = rng.choice(ids, len(ids), replace=True)
                    ix = np.concatenate([idx_by[c] for c in pick])
                    yy = y_te[ix]
                    if yy.min() == yy.max():
                        continue
                    d.append(auc(yy, xa[ix]) - auc(yy, xb[ix]))
                d = np.asarray(d)
                boot.append({"facet": facet, "seed": seed, "vs": name,
                             "delta_pp": round((auc(y_te, xa) - auc(y_te, xb)) * 100, 2),
                             "ci_lo": round(float(np.percentile(d, 2.5)) * 100, 2),
                             "ci_hi": round(float(np.percentile(d, 97.5)) * 100, 2)})

            rows.append(res)
            print(f"  s{seed} R2 H {res['r2_huber']:.4f} M {res['r2_mse']:.4f} | "
                  f"AUROC full H {res['auroc_huber_full']:.4f} "
                  f"M {res['auroc_mse_full']:.4f} | {time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    b = pd.DataFrame(boot)
    d.to_csv(OUT / f"mse_vs_huber_{a.tag}.csv", index=False, encoding="utf-8-sig")
    b.to_csv(OUT / f"bootstrap_{a.tag}.csv", index=False, encoding="utf-8-sig")
    print("\n== mean over seeds ==")
    cols = ["r2_huber", "r2_mse", "resid_huber", "resid_mse",
            "auroc_huber_val_lam", "auroc_huber_full", "auroc_mse_full"]
    print(d.groupby("facet")[cols].mean().round(4).to_string())
    print("\n== bootstrap (V2 minus Huber) ==")
    print(b.groupby(["facet", "vs"])[["delta_pp", "ci_lo", "ci_hi"]].mean().round(2).to_string())
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
