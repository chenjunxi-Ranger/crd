"""FOIL robustness across image-disjoint validation splits (frozen constants)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gate_final_grid import _logp, _topk_means            # noqa: E402
from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, PROJECT, RESULTS_DIR, write_json  # noqa: E402

GPT2 = PROJECT / "LEMoN+CLIPsim" / "weights" / "gpt2"
A, B = 6.0, 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(GPT2))
    tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(str(GPT2)).to(dev).eval()
    enc = JointEncoder(CLIP_DIR)

    df = pd.read_csv(args.split_csv, encoding="utf-8-sig")
    with np.load(RESULTS_DIR / "gate" / "bank_foil_all.npz") as z:
        bi, bt = z["img"], z["txt"]
    vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
    out = {}
    for tag, caps in (("pos", df["positive_caption"].astype(str).tolist()),
                      ("neg", df["negative_caption"].astype(str).tolist())):
        vt = enc.encode_texts(caps)
        cos = (vi * vt).sum(1)
        rx, _ = _topk_means(vi, bt)
        rc, _ = _topk_means(vt, bi)
        lp = _logp(tok, lm, caps, dev)
        out[tag] = -cos + A * ((rx + rc) / 2 - cos) + B * (-lp)
    d = out["neg"] - out["pos"]
    acc = float(np.mean(np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))))
    payload = {"bench": "foil", "tag": args.tag, "n_pairs": int(len(df)),
               "accuracy": acc, "threshold": 0.92,
               "gap": float(0.92 - acc), "pass": bool(acc >= 0.92),
               "tie_rate": float((d == 0).mean())}
    write_json(RESULTS_DIR / "bench" / f"seed_foil_{args.tag}.json", payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
