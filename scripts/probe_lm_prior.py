"""Stage-3 iteration 3: pretraining-corpus prior (C4 removed by user).

GPT-2 (local weights) gives a per-caption mean token log-probability.  Two ways
to use it in ONE formula with one global lambda:

  A  s = -cos - lam * logp          (prior re-weights the image evidence)
  B  s = -cos + lam * (-logp)       (low-probability caption is suspicious)

Measured on COCO val (AUROC) and FOIL val (within-pair accuracy) with the same
lambda, to see whether the Pareto conflict of the witness term disappears.

Analysis only.
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

from src.protocol import PROJECT, RESULTS_DIR, load_foil_val, load_split  # noqa: E402

GPT2 = PROJECT / "LEMoN+CLIPsim" / "weights" / "gpt2"


@torch.no_grad()
def logp(model, tok, caps, device, batch=64, max_len=64):
    out = []
    for i in range(0, len(caps), batch):
        chunk = [c if c.strip() else "a photo" for c in caps[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[:, :-1]
        tgt = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        lp = (lp * mask).sum(1) / mask.sum(1).clamp_min(1)
        out.append(lp.cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    tok = AutoTokenizer.from_pretrained(str(GPT2))
    model = AutoModelForCausalLM.from_pretrained(str(GPT2)).to(dev).eval()
    tok.pad_token = tok.eos_token

    # COCO
    cdf = load_split("coco", "val", args.limit)
    cfeat = pd.read_csv(RESULTS_DIR / "gate" / "coco_val_features.csv", encoding="utf-8-sig")
    cfeat = cfeat.head(len(cdf))
    lp_c = logp(model, tok, cdf["caption"].tolist(), dev)
    cos_c = cfeat["cosine"].to_numpy()
    y = cdf["label"].to_numpy()

    # FOIL
    fdf = load_foil_val(args.limit)
    ffeat = pd.read_csv(RESULTS_DIR / "gate" / "foil_val_features.csv", encoding="utf-8-sig")
    ffeat = ffeat.head(len(fdf))
    lp_p = logp(model, tok, fdf["positive"].tolist(), dev)
    lp_n = logp(model, tok, fdf["negative"].tolist(), dev)
    cos_p = ffeat["cosine_pos"].to_numpy()
    cos_n = ffeat["cosine_neg"].to_numpy()

    rows = []
    for lam in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        for mode in ("A", "B"):
            if mode == "A":
                sc = -cos_c - lam * lp_c
                sp, sn = -cos_p - lam * lp_p, -cos_n - lam * lp_n
            else:
                sc = -cos_c + lam * (-lp_c)
                sp, sn = -cos_p + lam * (-lp_p), -cos_n + lam * (-lp_n)
            auc = float(roc_auc_score(y, sc))
            d = sn - sp
            acc = float(np.mean(np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))))
            rows.append({"lam": lam, "mode": mode, "coco_auc": auc, "foil_acc": acc,
                         "min_margin": min(auc - 0.96, acc - 0.92)})
    df = pd.DataFrame(rows).sort_values("min_margin", ascending=False)
    df.to_csv(RESULTS_DIR / "gate" / "lm_prior_grid.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print("perplexity proxy: coco mean logp", float(lp_c.mean()),
          "foil pos/neg", float(lp_p.mean()), float(lp_n.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
