"""Signed component attribution on FOIL + text-retrieval-contrast distribution.

delta_s  = s_pos - s_neg        (>0 means cosine ranks the clean caption first)
delta_rc = r_c(pos) - r_c(neg)  (>0 means the CLEAN caption has higher text density)
A_foil - A_true = delta_s - lam * delta_rc

g_c(c) = mean_topk sim(c, bank_img) - mean_{rest} sim(c, bank_img)
(the second term is the exact background mean over all non-neighbour banks)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lora_stage1 import inject                              # noqa: E402
from src.encoders import JointEncoder                       # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_foil_val, write_json  # noqa: E402

K, LAM = 10, 0.8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--limit", type=int, default=5000)
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
    bank_img = torch.as_tensor(bi, device=enc.device)

    df = load_foil_val(args.limit)
    vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
    tp = enc.encode_texts(df["positive"].astype(str).tolist())
    tn = enc.encode_texts(df["negative"].astype(str).tolist())
    sp, sn = (vi * tp).sum(1), (vi * tn).sum(1)

    def text_stats(t):
        """top-k mean, background mean over the rest, contrast."""
        tk, bg = [], []
        for i in range(0, len(t), 500):
            q = torch.as_tensor(t[i:i + 500], device=enc.device)
            sim = q @ bank_img.t()
            vals, _ = torch.topk(sim, K, dim=1)
            s_all = sim.sum(1)
            tk.append(vals.mean(1).cpu().numpy())
            bg.append(((s_all - vals.sum(1)) / (sim.shape[1] - K)).cpu().numpy())
        tk, bg = np.concatenate(tk), np.concatenate(bg)
        return tk, bg, tk - bg

    rp, bp, gp = text_stats(tp)
    rn, bn, gn = text_stats(tn)

    ds = sp - sn
    drc = rp - rn
    flip = (ds > 0) & ((ds - LAM * drc) < 0)          # cosine right, full CSLS wrong

    out = {"seed": args.seed, "n": int(len(ds)),
           "mean_ds": float(ds.mean()), "mean_drc": float(drc.mean()),
           "median_drc": float(np.median(drc)),
           "p_drc_pos": float((drc > 0).mean()),
           "corr_ds_drc": float(np.corrcoef(ds, drc)[0, 1]),
           "flip_rate": float(flip.mean()),
           "flip_mean_drc": float(drc[flip].mean()) if flip.any() else None,
           "cos_acc": float((ds > 0).mean()),
           "full_acc": float((ds - LAM * drc > 0).mean()),
           "gc_pos": {"mean": float(gp.mean()), "sd": float(gp.std()),
                      "q": [float(x) for x in np.percentile(gp, [5, 50, 95])]},
           "gc_neg": {"mean": float(gn.mean()), "sd": float(gn.std()),
                      "q": [float(x) for x in np.percentile(gn, [5, 50, 95])]},
           "gc_mean_diff": float((gp - gn).mean()),
           "p_gc_pos_gt_neg": float((gp > gn).mean()),
           "corr_gc_ds": float(np.corrcoef(gp - gn, ds)[0, 1]),
           "rc_ds": {"mean": float(rp.mean()), "sd": float(rp.std())},
           "gc_ds": {"mean": float(gp.mean()), "sd": float(gp.std())},
           "corr_rc_gc": float(np.corrcoef(np.r_[rp, rn], np.r_[gp, gn])[0, 1])}
    write_json(RESULTS_DIR / "gate" / f"foil_signed_seed{args.seed}.json", out)
    np.savez_compressed(RESULTS_DIR / "gate" / f"foil_signed_seed{args.seed}.npz",
                        ds=ds, drc=drc, rc_pos=rp, rc_neg=rn, gc_pos=gp, gc_neg=gn)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
