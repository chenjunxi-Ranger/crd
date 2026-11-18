"""CRD v2 candidate: modality-specific gated MoE, against its own controls.

Three configurations, identical data / seeds / budget / loss / lambda protocol:

    mlp64    CRD v1  (2 x 512-64-1)                       65,795 params
    mlp128   parameter-matched control (2 x 512-128-1)   131,587 params
    moe2     two gated experts per modality               134,667 params

The control is the point: if mlp128 already matches moe2, the gain (if any) is
capacity, not the gating or the per-modality banks. Gate statistics are logged,
because a collapsed gate means the mixture degenerated into one expert.

Run:  python fgbench/experiments/crd_v2/run_moe.py --facets Attribute --seeds 0
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
from fgbench.models.crd_moe import MoEAdditive, param_count  # noqa: E402

CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "crd_v2"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = np.round(np.arange(0, 2.01, 0.1), 1)
N_BOOT = 1000
FACETS = {"Entity": ("COCO", "Entity"), "Quantity": ("COCO", "Quantity"),
          "Attribute": ("VG", "Attribute"), "Relation": ("VG", "Relation")}


def auc(y, x):
    return float(roc_auc_score(y, -x))


def ap(y, x):
    return float(average_precision_score(y, -x))


def r2(z, p):
    return 1.0 - float(((z - p) ** 2).sum()) / float(((z - z.mean()) ** 2).sum())


def load(facet, name, mu, sd):
    ds, fc = FACETS[facet]
    fn = (sorted(CACHE.glob(f"crdbb_clip_{ds}_{fc}_c*_train.npz"))[0]
          if name == "train" else CACHE / f"crdbb_clip_{ds}_{fc}_{name}.npz")
    z = np.load(fn, allow_pickle=True)
    U, V = z["U"].astype(np.float32), z["V"].astype(np.float32)
    meta = pd.read_parquet(fn.with_suffix(".meta.parquet"))
    return U, V, ((U * V).sum(1) - mu) / sd, meta


def build(kind: str):
    if kind == "mlp64":
        return Off(d=512, width=64)
    if kind == "mlp128":
        return Off(d=512, width=128)
    if kind == "moe2":
        return MoEAdditive(d=512, k=2, width=64)
    if kind == "moe2_w40":
        return MoEAdditive(d=512, k=2, width=40)
    raise ValueError(kind)


def train(m, U, V, z, seed, steps=1500, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    m = m.to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    Ut = torch.tensor(U, device=DEV); Vt = torch.tensor(V, device=DEV)
    Mt = torch.tensor(z, device=DEV, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = torch.tensor(rng.integers(0, len(Ut), batch), device=DEV)
        loss = F.huber_loss(m(Ut[idx], Vt[idx]), Mt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


@torch.no_grad()
def qpred(m, U, V):
    return m(torch.tensor(U, device=DEV),
             torch.tensor(V, device=DEV)).cpu().numpy().ravel()


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--facets", default="Attribute,Quantity")
    ap_.add_argument("--seeds", default="0,1,2")
    ap_.add_argument("--configs", default="mlp64,mlp128,moe2")
    ap_.add_argument("--steps", type=int, default=1500)
    ap_.add_argument("--tag", default="moe")
    a = ap_.parse_args()

    rng = np.random.default_rng(0)
    rows, gates, boot = [], [], []

    for facet in a.facets.split(","):
        ds, fc = FACETS[facet]
        prov = json.loads((CACHE / f"crdbb_prov_clip_{ds}_{fc}_both_row_s0.json")
                          .read_text(encoding="utf-8"))
        mu, sd = float(prov["mu"]), float(prov["sd"])
        Utr, Vtr, ztr, _ = load(facet, "train", mu, sd)
        Uva, Vva, zva, mva = load(facet, "val", mu, sd)
        Ute, Vte, zte, mte = load(facet, "test", mu, sd)
        y_va, y_te = mva.pair_label.to_numpy(), mte.pair_label.to_numpy()
        cyc = mte.cycle_id.values
        print(f"\n{facet}: {len(ztr)} train / {len(zte)} test rows", flush=True)

        for seed in (int(s) for s in a.seeds.split(",")):
            scores = {}
            for kind in a.configs.split(","):
                t0 = time.time()
                net = build(kind)
                n_par = param_count(net)
                m = train(net, Utr, Vtr, ztr, seed, steps=a.steps)
                qva, qte = qpred(m, Uva, Vva), qpred(m, Ute, Vte)
                lam = float(GRID[int(np.argmax([auc(y_va, zva - L * qva)
                                                for L in GRID]))])
                te = zte - lam * qte
                scores[kind] = te
                rows.append({"facet": facet, "seed": seed, "config": kind,
                             "params": n_par, "lam": lam,
                             "r2_test": round(r2(zte, qte), 4),
                             "auroc": round(auc(y_te, te), 4),
                             "auprc": round(ap(y_te, te), 4)})
                if kind.startswith("moe"):
                    st = m.gate_stats(torch.tensor(Ute, device=DEV),
                                      torch.tensor(Vte, device=DEV))
                    gates.append({"facet": facet, "seed": seed, "config": kind, **st})
                print(f"  s{seed} {kind:9s} {n_par:>8,}p  R2 {r2(zte,qte):.4f}  "
                      f"AUROC {auc(y_te, te):.4f}  lam {lam}  {time.time()-t0:.0f}s",
                      flush=True)

            by = pd.Series(np.arange(len(cyc))).groupby(cyc).apply(
                lambda s: s.to_numpy()).to_dict()
            ids = list(by)
            for kind in a.configs.split(","):
                if kind == "mlp64":
                    continue
                d = []
                for _ in range(N_BOOT):
                    pick = rng.choice(ids, len(ids), replace=True)
                    ix = np.concatenate([by[c] for c in pick])
                    yy = y_te[ix]
                    if yy.min() == yy.max():
                        continue
                    d.append(auc(yy, scores[kind][ix]) - auc(yy, scores["mlp64"][ix]))
                d = np.asarray(d)
                boot.append({"facet": facet, "seed": seed, "config": kind,
                             "delta_pp": round((auc(y_te, scores[kind])
                                                - auc(y_te, scores["mlp64"])) * 100, 2),
                             "ci_lo": round(float(np.percentile(d, 2.5)) * 100, 2),
                             "ci_hi": round(float(np.percentile(d, 97.5)) * 100, 2)})

    r = pd.DataFrame(rows)
    b = pd.DataFrame(boot)
    r.to_csv(OUT / f"{a.tag}_results.csv", index=False, encoding="utf-8-sig")
    b.to_csv(OUT / f"{a.tag}_boot.csv", index=False, encoding="utf-8-sig")
    if gates:
        pd.DataFrame(gates).to_csv(OUT / f"{a.tag}_gates.csv", index=False,
                                   encoding="utf-8-sig")

    print("\n== test AUROC (mean over seeds) ==")
    print(r.groupby(["facet", "config"]).auroc.mean().unstack().round(4).to_string())
    print("\n== test R^2 of the additive fit ==")
    print(r.groupby(["facet", "config"]).r2_test.mean().unstack().round(4).to_string())
    if not b.empty:
        print("\n== bootstrap vs mlp64 ==")
        print(b.groupby(["facet", "config"])[["delta_pp", "ci_lo", "ci_hi"]]
              .mean().round(2).to_string())
    if gates:
        g = pd.DataFrame(gates)
        print("\n== gate usage (mean over seeds and rows) ==")
        print(g.groupby(["facet", "config"])[["image_gate", "text_gate",
                                              "image_entropy", "text_entropy"]]
              .first().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
