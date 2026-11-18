"""CRD v2 on VG-Action: the facet with no cached features.

Action uses the internal train/val split of `fgbench/models/crd_action.py`
(319 / 80 / 166 cycles), so features are re-encoded once and reused for all six
heads.  Everything else is identical to `run_additive_interaction.py --stage 3`:

    features   uvdot = [u; v; u*v]
    heads      B_rank (pairwise ranking) and B_bce (BCE on the pair label)
    score      S = z - lam * q_add + eta * g, eta on the internal validation split

Run:  python fgbench/experiments/crd_v2/run_action_interaction.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "models"))
sys.path.insert(0, str(ROOT / "fgbench" / "experiments" / "p0_baselines"))
sys.path.insert(0, str(ROOT / "fgbench" / "experiments" / "crd_v2"))

from run_baselines import action_splits  # noqa: E402
from run_additive_interaction import (  # noqa: E402
    CACHE, DEV, ETA_GRID, OUT, acc2afc, auc, feats, train_head)
from fgbench.models.crd_backbone import Off  # noqa: E402


def crd_action():
    ck = torch.load(CACHE / "crd_action_head_s0.pt", map_location=DEV,
                    weights_only=False)
    head = Off(d=int(ck["dim"]), parts="both").to(DEV)
    head.load_state_dict(ck["state"])
    head.eval()
    mu, sd, lam = float(ck["mu"]), float(ck["sd"]), float(ck["lam"])

    @torch.no_grad()
    def score(U, V):
        S = np.einsum("ij,ij->i", U, V)
        q = head(torch.tensor(U.astype(np.float32), device=DEV),
                 torch.tensor(V.astype(np.float32), device=DEV)).cpu().numpy().ravel()
        return (S - mu) / sd - lam * q
    return score


def main() -> int:
    t0 = time.time()
    (dtr, Utr, Vtr), (dva, Uva, Vva), (dte, Ute, Vte) = action_splits()
    print(f"encoded in {time.time()-t0:.0f}s: {len(dtr)}/{len(dva)}/{len(dte)} rows",
          flush=True)
    S1_va, S1_te = crd_action()(Uva, Vva), crd_action()(Ute, Vte)
    ytr = dtr.pair_label.to_numpy().astype(np.int64)          # 1 = noisy
    yva = dva.pair_label.to_numpy().astype(np.int64)
    yte = dte.pair_label.to_numpy().astype(np.int64)
    Xtr, Xva, Xte = feats(Utr, Vtr, "uvdot"), feats(Uva, Vva, "uvdot"), \
        feats(Ute, Vte, "uvdot")
    print(f"cos {auc(yte, (Ute*Vte).sum(1)):.4f} | CRD {auc(yte, S1_te):.4f}",
          flush=True)

    rows, payload = [], {}
    for scheme in ("B_rank", "B_bce"):
        for seed in (0, 1, 2):
            model, gm, gs, gtr, gva = train_head(
                "uvdot", "Action", seed, scheme, Xtr, ytr, dtr, Xva, yva)
            with torch.no_grad():
                gte = (model(torch.tensor(Xte, device=DEV)).cpu().numpy() - gm) / gs
            va_auc = [auc(yva, S1_va + e * gva) for e in ETA_GRID]
            eta = float(ETA_GRID[int(np.argmax(va_auc))])
            S3 = S1_te + eta * gte
            rec = {
                "facet": "Action", "scheme": scheme, "features": "uvdot",
                "seed": seed,
                "cosine": round(auc(yte, (Ute * Vte).sum(1)), 4),
                "E1_crd": round(auc(yte, S1_te), 4),
                "E2_interaction": round(auc(yte, gte), 4),
                "eta": eta, "val_E3": round(float(max(va_auc)), 4),
                "E3_crd_plus": round(auc(yte, S3), 4),
                "E3_minus_E1_pp": round((auc(yte, S3) - auc(yte, S1_te)) * 100, 2),
                "E2_auprc": round(float(average_precision_score(yte, -gte)), 4),
                "E3_auprc": round(float(average_precision_score(yte, -S3)), 4),
                "E1_2afc": round(acc2afc(-S1_te, dte.cycle_id.values, yte), 6),
                "E3_2afc": round(acc2afc(-S3, dte.cycle_id.values, yte), 6),
            }
            rows.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            k = f"{scheme}_uvdot_{seed}"
            payload[f"{k}__cycle"] = dte.cycle_id.values
            payload[f"{k}__y"] = yte
            payload[f"{k}__E1"] = S1_te
            payload[f"{k}__E3"] = S3
            payload[f"{k}__g"] = gte

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "action_interaction.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(OUT / "interaction_scores_Action_stage3.npz", **payload)
    print("\n", df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
