"""CRD on VG-Action with an INTERNAL validation split (test set untouched).

Protocol
  - test = the frozen strict test split (166 cycles)
  - train cycles are split 80/20 into (train, internal-val) by DATA_SEED
  - any val cycle sharing an image with the train part is moved back to train,
    so the two parts are image-disjoint
  - mu/sd come from the train part only; lambda is chosen on the internal val
  - three seeds; identical head/target/inference formulas as crd_backbone
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH, jdump  # noqa: E402
from fgbench.models.crd_backbone import ClipBB, Off, sha_text, DATA_SEED  # noqa: E402

OUT = ROOT / "results" / "fgbench"
VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
VAL_FRAC = 0.2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    args = ap.parse_args()

    torch.manual_seed(0)
    bb = ClipBB()
    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()

    def rows(split):
        d = pd.read_parquet(BENCH / f"{split}.parquet")
        d = d[(d.dataset == "VG") & (d.facet == "Action")].copy()
        d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
        d = d[d.iid.isin(vgidx)]
        d = d[d.groupby("cycle_id").target_image_id.transform("nunique") == 2]
        return d.reset_index(drop=True)

    def encode(d):
        imgs = d.iid.unique()
        paths = [str(VG_ROOT / vgidx[i]) for i in imgs]
        U = bb.images(paths)
        pos = {i: k for k, i in enumerate(imgs)}
        U = U[[pos[i] for i in d.iid]]
        V = bb.texts(d.caption.astype(str).tolist())
        return U, V

    tr_all, te = rows("train"), rows("test")
    print(f"train {tr_all.cycle_id.nunique()} cycles | test {te.cycle_id.nunique()}",
          flush=True)

    # ---- split train cycles, then enforce image-disjointness ----
    cyc2imgs = tr_all.groupby("cycle_id").iid.apply(set).to_dict()
    cids = np.array(sorted(cyc2imgs))
    rng = np.random.default_rng(DATA_SEED)
    rng.shuffle(cids)
    n_val = max(1, int(round(len(cids) * args.val_frac)))
    val_ids, train_ids = set(cids[:n_val]), set(cids[n_val:])
    train_imgs = set().union(*[cyc2imgs[c] for c in train_ids])
    moved = [c for c in list(val_ids) if cyc2imgs[c] & train_imgs]
    for c in moved:
        val_ids.discard(c); train_ids.add(c)
    val_imgs = set().union(*[cyc2imgs[c] for c in val_ids]) if val_ids else set()
    print(f"internal split: train {len(train_ids)} cycles / val {len(val_ids)} cycles",
          flush=True)
    print(f"  val cycles moved to train (image overlap): {len(moved)}", flush=True)
    print(f"  train/val image overlap after fix: {len(train_imgs & val_imgs)}", flush=True)

    dtr = tr_all[tr_all.cycle_id.isin(train_ids)].reset_index(drop=True)
    dva = tr_all[tr_all.cycle_id.isin(val_ids)].reset_index(drop=True)
    t0 = time.time()
    Utr, Vtr = encode(dtr)
    Uva, Vva = encode(dva)
    Ute, Vte = encode(te)
    print(f"  encoded in {time.time()-t0:.0f}s: {len(dtr)}/{len(dva)}/{len(te)} rows",
          flush=True)

    Str = (Utr * Vtr).sum(1)
    Sva = (Uva * Vva).sum(1)
    Ste = (Ute * Vte).sum(1)
    mu, sd = float(Str.mean()), float(Str.std() + 1e-8)
    Str, Sva, Ste = (Str - mu) / sd, (Sva - mu) / sd, (Ste - mu) / sd
    print(f"  mu {mu:.4f} sd {sd:.4f} (train part only)", flush=True)

    yv, yt = dva.pair_label.to_numpy(), te.pair_label.to_numpy()
    prov = {"facet": "Action", "dataset": "VG", "backbone": "clip",
            "protocol": "internal_val_split", "data_seed": DATA_SEED,
            "val_frac": args.val_frac, "n_train_cycles": len(train_ids),
            "n_val_cycles": len(val_ids), "n_test_cycles": int(te.cycle_id.nunique()),
            "overlap_images_after_fix": len(train_imgs & val_imgs),
            "train_order_sha": sha_text(dtr.cycle_id.astype(str).values),
            "mu": mu, "sd": sd, "steps": args.steps}

    rows_out = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        rng_t = np.random.default_rng(seed)
        head = Off(d=Utr.shape[1]).to("cuda" if torch.cuda.is_available() else "cpu")
        dev = next(head.parameters()).device
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        Ut = torch.tensor(Utr, device=dev); Vt = torch.tensor(Vtr, device=dev)
        Mt = torch.tensor(Str, device=dev, dtype=torch.float32)
        for _ in range(args.steps):
            idx = torch.tensor(rng_t.integers(0, len(Ut), args.batch), device=dev)
            loss = F.huber_loss(head(Ut[idx], Vt[idx]), Mt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            mh_va = head(torch.tensor(Uva, device=dev),
                         torch.tensor(Vva, device=dev)).cpu().numpy()
            mh_te = head(torch.tensor(Ute, device=dev),
                         torch.tensor(Vte, device=dev)).cpu().numpy()
        lam = max(np.round(np.arange(0, 2.01, 0.1), 1),
                  key=lambda L: roc_auc_score(yv, -(Sva - L * mh_va)))
        raw, crd = -Ste, -(Ste - lam * mh_te)

        def a2(score):
            df = pd.DataFrame({"c": te.cycle_id.values, "y": yt, "s": score})
            acc = [g[g.y == 0].s.sum() < g[g.y == 1].s.sum()
                   for _, g in df.groupby("c") if len(g) == 4]
            return float(np.mean(acc)) if acc else float("nan")

        rec = {"seed": seed, "lam": float(lam),
               "auroc_raw": round(float(roc_auc_score(yt, raw)), 4),
               "auroc_crd": round(float(roc_auc_score(yt, crd)), 4),
               "auprc_raw": round(float(average_precision_score(yt, raw)), 4),
               "auprc_crd": round(float(average_precision_score(yt, crd)), 4),
               "acc2afc_raw": round(a2(raw), 6), "acc2afc_crd": round(a2(crd), 6)}
        rec["delta_pp"] = round((rec["auroc_crd"] - rec["auroc_raw"]) * 100, 2)
        print(rec, flush=True)
        rows_out.append(rec)
        # save in SIMILARITY space (same convention as crd_backbone's npz)
        np.savez_compressed(OUT / f"crd_action_scores_s{seed}.npz",
                            cycle_id=te.cycle_id.values, y=yt,
                            raw=Ste, crd=Ste - lam * mh_te,
                            va_cycle_id=dva.cycle_id.values, va_y=yv, va_raw=Sva,
                            va_crd=Sva - lam * mh_va, lam=np.array([lam]))
        torch.save({"state": head.state_dict(), "mu": mu, "sd": sd,
                    "dim": int(Utr.shape[1]), "lam": float(lam)},
                   OUT / f"crd_action_head_s{seed}.pt")

    prov["runs"] = rows_out
    prov["auroc_crd_mean"] = round(float(np.mean([r["auroc_crd"] for r in rows_out])), 4)
    prov["auroc_crd_sd"] = round(float(np.std([r["auroc_crd"] for r in rows_out],
                                              ddof=1)), 4)
    jdump(prov, OUT / "crd_action_prov.json")
    print("summary", {k: prov[k] for k in ("auroc_crd_mean", "auroc_crd_sd")}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
