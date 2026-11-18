"""FOIL score decomposition: which term destroys the within-pair ordering?

Scores (higher = more suspicious):
    cosine      -s
    image-only  -s + lam * r_I
    text-only   -s + lam * r_c
    full        -s + lam * (r_I + r_c)

Reports both
  * within-pair accuracy (the C5 metric; same image, two candidate captions)
  * pooled AUROC over all 2N records (different images get compared)
with image-clustered bootstrap intervals for both.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lora_stage1 import inject                              # noqa: E402
from rat_gate import topk_mean                              # noqa: E402
from src.encoders import JointEncoder                       # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_foil_val, write_json  # noqa: E402

LAM, K = 0.8, 10


def boot(fn, groups, n_boot=2000, seed=0):
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        v = fn(sel)
        if v is not None:
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    model = enc.model
    inject(model)
    ck = torch.load(RESULTS_DIR / "seed_audit" / f"seed_{args.seed}" / "adapter.pt",
                    map_location="cpu")
    model.load_state_dict(ck, strict=False)
    model.to(enc.device).eval()

    with np.load(RESULTS_DIR / "seed_audit" / f"seed_{args.seed}" / "bank_foil.npz") as z:
        bi, bt = z["img"], z["txt"]

    df = load_foil_val(args.limit)
    vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
    tp = enc.encode_texts(df["positive"].astype(str).tolist())
    tn = enc.encode_texts(df["negative"].astype(str).tolist())
    gid = df["image_id"].astype(str).to_numpy()

    sp, sn = (vi * tp).sum(1), (vi * tn).sum(1)
    ri = topk_mean(vi, bt)                     # image side: identical for + and -
    rcp = topk_mean(tp, bi)
    rcn = topk_mean(tn, bi)

    scores = {
        "cosine": (-sp, -sn),
        "image_only": (-sp + LAM * ri, -sn + LAM * ri),
        "text_only": (-sp + LAM * rcp, -sn + LAM * rcn),
        "full": (-sp + LAM * (ri + rcp), -sn + LAM * (ri + rcn)),
    }

    out = {"seed": args.seed, "n_pairs": int(len(df))}
    for name, (Ap, An) in scores.items():
        within = float(np.mean(An > Ap))
        lo_w, hi_w = boot(lambda s: float(np.mean(An[s] > Ap[s])), gid)
        y = np.r_[np.zeros(len(df)), np.ones(len(df))]
        pooled = np.concatenate([Ap, An])
        auc = float(roc_auc_score(y, pooled))
        lo_p, hi_p = boot(lambda s: roc_auc_score(y[s], pooled[s]), np.r_[gid, gid])
        out[name] = {"within_pair": within, "within_ci": [lo_w, hi_w],
                     "pooled_auc": auc, "pooled_ci": [lo_p, hi_p]}
        print(name, out[name], flush=True)

    # how much of the image-side term differs between pos and neg
    out["diag"] = {"r_I_pos_equals_neg": bool(np.allclose(ri, ri)),
                   "mean_abs_drc": float(np.mean(np.abs(rcp - rcn))),
                   "mean_abs_ds": float(np.mean(np.abs(sp - sn)))}
    print("diag", out["diag"], flush=True)
    write_json(RESULTS_DIR / "gate" / "foil_decomp.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
