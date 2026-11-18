"""Split/seed robustness check for v2.0.

Row benches: evaluate on a different noise realisation of the same protocol
(mscoco/flickr30k seed1/seed2 CSVs) with the SAME frozen constants and the same
reference bank.

FOIL: evaluate on a different image-disjoint validation split (seed1/seed2).

No constant is re-selected here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gate_final_grid import _logp, _topk_means            # noqa: E402
from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, PROJECT, RESULTS_DIR, write_json  # noqa: E402

GPT2 = PROJECT / "LEMoN+CLIPsim" / "weights" / "gpt2"
A, B = 6.0, 0.05


def _rows(csv: Path, split: str, limit=None) -> pd.DataFrame:
    df = pd.read_csv(csv, encoding="utf-8-sig",
                     usecols=["split", "image_id", "path", "sentence", "is_mislabel"])
    sub = df[df["split"] == split]
    out = pd.DataFrame({"path": sub["path"].astype(str).to_numpy(),
                        "caption": sub["sentence"].astype(str).to_numpy(),
                        "label": sub["is_mislabel"].astype(int).to_numpy()})
    return out.head(limit) if limit else out.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(GPT2))
    tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(str(GPT2)).to(dev).eval()
    enc = JointEncoder(CLIP_DIR)

    df = _rows(Path(args.csv), "val", args.limit)
    bank_path = RESULTS_DIR / "gate" / f"bank_{args.bench}_all.npz"
    with np.load(bank_path) as z:
        bi, bt = z["img"], z["txt"]
    vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
    vt = enc.encode_texts(df["caption"].tolist())
    cos = (vi * vt).sum(1)
    rx, _ = _topk_means(vi, bt)
    rc, _ = _topk_means(vt, bi)
    lp = _logp(tok, lm, df["caption"].tolist(), dev)
    s = -cos + A * ((rx + rc) / 2 - cos) + B * (-lp)
    auc = float(roc_auc_score(df["label"], s))
    out = {"bench": args.bench, "tag": args.tag, "n": int(len(df)),
           "auroc": auc, "threshold": 0.96,
           "gap": float(0.96 - auc), "pass": bool(auc >= 0.96)}
    write_json(RESULTS_DIR / "bench" / f"seed_{args.bench}_{args.tag}.json", out)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
