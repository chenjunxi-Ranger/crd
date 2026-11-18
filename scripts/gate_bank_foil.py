"""Gate: bank-side statistics on FOIL pairs with the full image-disjoint bank.

Bank = FOIL official train minus every image used by the frozen validation
split (62,345 images / 187,788 pairs).  No test data, no labels.

Analysis only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import (CLIP_DIR, RESULTS_DIR, THRESHOLDS, load_foil_val,  # noqa: E402
                          write_json)
from gate_bank_ops import _ops                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--bank", type=int, default=0, help="0 = prepared full bank")
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    df = load_foil_val(args.limit)
    vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
    tp = enc.encode_texts(df["positive"].tolist())
    tn = enc.encode_texts(df["negative"].tolist())

    image_key = None
    if args.bank <= 0:
        with np.load(RESULTS_DIR / "gate" / "bank_foil_all.npz") as z:
            bi, bt = z["img"], z["txt"]
            image_key = z["image_key"]
        bank_note = "full image-disjoint FOIL train"
    else:
        from gate_bank_ops import _load_bank
        bi, bt = _load_bank("coco", args.bank, 0, enc)
        bank_note = f"COCO train subsample {args.bank}"

    fp = _ops(vi, tp, bi, bt, chunk=args.chunk)
    fn = _ops(vi, tn, bi, bt, chunk=args.chunk)

    def acc(diff: np.ndarray) -> float:
        return float(np.mean(np.where(diff < 0, 1.0, np.where(diff > 0, 0.0, 0.5))))

    single = {k: acc(fn[k] - fp[k]) for k in fp}
    out = {"bench": "foil", "n_pairs": int(len(df)), "n_bank_rows": int(len(bi)),
           "n_bank_images": int(len(np.unique(image_key))) if image_key is not None else None,
           "bank": bank_note, "bank_image_disjoint": True,
           "threshold": THRESHOLDS["foil"], "single": single,
           "best_single": max(single.values()), "best_name": max(single, key=single.get),
           "gap": float(THRESHOLDS["foil"] - max(single.values()))}
    write_json(RESULTS_DIR / "gate" / "foil_bank_ops_full.json", out)
    print(pd.Series(single).sort_values(ascending=False).to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "single"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
