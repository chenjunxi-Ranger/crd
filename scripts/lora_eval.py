"""Stage 2/3: four-column evaluation of the adapted CLIP with CSLS.

A(I,c) = -cos_theta + lambda * h_theta,   h = r_I + r_c  (bank side, adapted)
One global lambda, same formula for row-level and within-pair protocols.
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
from rat_gate import topk_mean                              # noqa: E402
from src.encoders import JointEncoder                       # noqa: E402
from src.protocol import (CLIP_DIR, RESULTS_DIR, THRESHOLDS, load_foil_val,  # noqa: E402
                          load_split, write_json)

LAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--foil-limit", type=int, default=5000)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    model = enc.model
    inject(model)
    ck = torch.load(RESULTS_DIR / "gate" / "lora_weights.pt", map_location="cpu")
    model.load_state_dict(ck["state"], strict=False)
    model.to(enc.device).eval()
    print("adapted model loaded", flush=True)

    out = {"lambda_sweep": {}}
    rows = {}
    for bench in ("coco", "flickr", "mmimdb"):
        d = load_split(bench, "val", args.limit or None)
        vi = enc.encode_images(d["path"].tolist(), batch_size=64, num_workers=4)
        vt = enc.encode_texts(d["caption"].tolist())
        cos = (vi * vt).sum(1)
        with np.load(RESULTS_DIR / "gate" / f"lora_bank_{bench}.npz") as z:
            bi, bt = z["img"], z["txt"]
        h = topk_mean(vi, bt) + topk_mean(vt, bi)
        y = d["label"].to_numpy()
        rows[bench] = (y, cos, h)
        np.savez_compressed(RESULTS_DIR / "gate" / f"lora_scores_{bench}.npz",
                            y=y, cos=cos, h=h)
        out[bench] = {"n": int(len(y)), "auc_cos": float(roc_auc_score(y, -cos)),
                      "sweep": {}}
        for lam in LAMS:
            out[bench]["sweep"][str(lam)] = float(roc_auc_score(y, -cos + lam * h))
        print(bench, json.dumps({k: v for k, v in out[bench].items() if k != "sweep"}),
              flush=True)

    foil = load_foil_val(args.foil_limit)
    vi = enc.encode_images(foil["path"].tolist(), batch_size=64, num_workers=4)
    tp = enc.encode_texts(foil["positive"].astype(str).tolist())
    tn = enc.encode_texts(foil["negative"].astype(str).tolist())
    with np.load(RESULTS_DIR / "gate" / "lora_bank_foil.npz") as z:
        bi, bt = z["img"], z["txt"]
    sp, sn = (vi * tp).sum(1), (vi * tn).sum(1)
    hp = topk_mean(vi, bt) + topk_mean(tp, bi)
    hn = topk_mean(vi, bt) + topk_mean(tn, bi)
    np.savez_compressed(RESULTS_DIR / "gate" / "lora_scores_foil.npz",
                        s_pos=sp, s_neg=sn, h_pos=hp, h_neg=hn)
    out["foil"] = {"n": int(len(foil)),
                   "acc_cos": float(np.mean(np.where(sn < sp, 1.0,
                                                     np.where(sn > sp, 0.0, 0.5)))),
                   "sweep": {}}
    for lam in LAMS:
        out["foil"]["sweep"][str(lam)] = float(np.mean(np.where(
            (-sn + lam * hn) > (-sp + lam * hp), 1.0,
            np.where((-sn + lam * hn) < (-sp + lam * hp), 0.0, 0.5))))
    print("foil", json.dumps({k: v for k, v in out["foil"].items() if k != "sweep"}),
          flush=True)

    # global lambda: maximise the minimum margin to the four thresholds
    best = None
    for lam in LAMS:
        m = {"coco": out["coco"]["sweep"][str(lam)],
             "flickr": out["flickr"]["sweep"][str(lam)],
             "mmimdb": out["mmimdb"]["sweep"][str(lam)],
             "foil": out["foil"]["sweep"][str(lam)]}
        margin = min(m[k] - THRESHOLDS[k] for k in m)
        if best is None or margin > best["margin"]:
            best = {"lambda": lam, "metrics": m, "margin": margin,
                    "n_pass": sum(m[k] >= THRESHOLDS[k] for k in m)}
    out["global_lambda"] = best
    print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)
    write_json(RESULTS_DIR / "bench" / "lora_four.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
