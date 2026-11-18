"""Compare detectors on FGBench v0.2c: Cosine / CSLS / (optional) R200 adapter.

Bank for CSLS is built ONLY from the train split (images + captions), never test.
Reports Overall / Entity / Quantity / Entity-easy / Entity-hard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fgbench.common import BENCH, PROJ   # noqa: E402

IMG_ROOT = Path(os.environ.get("COCO_IMG", str(PROJ / "数据集" / "MS-COCO" / "val2014")))


def topk_mean(a, b, k=10, chunk=1024):
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        s = a[i:i + chunk] @ b.T
        kk = min(k, s.shape[1])
        out[i:i + chunk] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=int, default=5000)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--dataset", default="", help="restrict to COCO / VG")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    enc = JointEncoder(CLIP_DIR)
    model = enc.model
    if args.adapter:
        import lora_stage1 as L
        nv = len(model.vision_model.encoder.layers)
        nt = len(model.text_model.encoder.layers)
        L.BLOCKS = (min(nv, nt) - 2, min(nv, nt) - 1)
        L.inject(model)
        sd = __import__("torch").load(ROOT / args.adapter, map_location=enc.device)
        model.load_state_dict(sd, strict=False)
    model.eval()

    def ei(paths, bs=64):
        return enc.encode_images(paths, batch_size=bs, num_workers=4)

    def et(caps):
        return enc.encode_texts(list(caps))

    te = pd.read_parquet(BENCH / "test.parquet")
    tr = pd.read_parquet(BENCH / "train.parquet")
    if args.dataset:
        te = te[te.dataset == args.dataset].reset_index(drop=True)
        tr = tr[tr.dataset == args.dataset].reset_index(drop=True)
    sys.path.insert(0, str(ROOT / "fgbench" / "audit"))
    from _cache import resolve as _resolve
    te_paths = [_resolve(f) for f in te.target_file]
    vi = ei(te_paths); vt = et(te.caption.astype(str))
    cos = (vi * vt).sum(1)

    bank = tr.drop_duplicates("target_image_id").head(args.bank)
    bi = ei([_resolve(f) for f in bank.target_file])
    bt = et(bank.caption.astype(str))
    ri = topk_mean(vi, bt)
    rc = topk_mean(vt, bi)

    rows = []
    y = te.pair_label.to_numpy()

    def report(name, score, mask=None, facet="Overall"):
        m = np.ones(len(te), bool) if mask is None else mask
        yy, ss = y[m], score[m]
        if len(np.unique(yy)) < 2:
            return
        uq = np.unique(te.target_image_id.values[m])
        idx = np.where(m)[0]
        by = {g: idx[te.target_image_id.values[idx] == g] for g in uq}
        vals = []
        for _ in range(300):
            pk = rng.choice(uq, len(uq), replace=True)
            ii = np.concatenate([by[g] for g in pk])
            if len(np.unique(y[ii])) < 2:
                continue
            vals.append(roc_auc_score(y[ii], score[ii]))
        rows.append({"detector": name, "facet": facet, "n_noisy": int((yy == 1).sum()),
                     "auc": float(roc_auc_score(yy, ss)),
                     "ci_lo": float(np.percentile(vals, 2.5)),
                     "ci_hi": float(np.percentile(vals, 97.5))})

    dets = {"cosine": -cos,
            "csls_0.8": -cos + 0.8 * (ri + rc),
            "csls_grid_best": None}
    best = (-9, None)
    for lam in (0.2, 0.4, 0.6, 0.8, 1.0):
        s = -cos + lam * (ri + rc)
        a = roc_auc_score(y, s)
        if a > best[0]:
            best = (a, lam)
        dets[f"csls_{lam}"] = s
    dets["csls_grid_best"] = dets[f"csls_{best[1]}"]
    print(f"best lambda (on test, reference only): {best[1]} -> {best[0]:.4f}", flush=True)

    for name in ("cosine", "csls_0.8", "csls_grid_best"):
        s = dets[name]
        report(name, s, None, "Overall")
        for f in ("Entity", "Quantity"):
            report(name, s, (te.facet == f).to_numpy(), f)
        for d in ("easy", "hard"):
            report(name, s, ((te.facet == "Entity") & (te.difficulty == d)).to_numpy(),
                   f"Entity-{d}")

    out = pd.DataFrame(rows)
    p = ROOT / "results" / "fgbench" / ("compare_" + (Path(args.adapter).stem if args.adapter else "clip") + ".csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
