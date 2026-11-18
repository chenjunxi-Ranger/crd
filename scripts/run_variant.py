"""Full-iteration evaluator for the two iteration candidates (NOT the deliverable).

v1.1-attn : s = -cos + 0.5*(1 - attention-weighted coverage)
v1.2-lm   : s = -cos - 0.05*mean_token_logp(GPT-2)

Each is one formula with one constant set; they exist only to complete the
Stage 3 -> 4 -> 5 loop that the plan requires.  The deliverable remains v1.0.
"""
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

from src.encoders import JointEncoder                       # noqa: E402
from src.protocol import (CLIP_DIR, PROJECT, RESULTS_DIR, THRESHOLDS,  # noqa: E402
                          bootstrap_acc, bootstrap_auc, load_foil_test,
                          load_foil_val, load_split, write_json)
from src.score import FrozenScorer                          # noqa: E402
from probe_attention import encode_att, features            # noqa: E402

GPT2 = PROJECT / "LEMoN+CLIPsim" / "weights" / "gpt2"
CHUNK = 2000


class LMScorer:
    def __init__(self, lam: float = 0.05, device: str = "auto") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.tok = AutoTokenizer.from_pretrained(str(GPT2))
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(str(GPT2)).to(self.dev).eval()
        self.lam = lam

    @torch.no_grad()
    def logp(self, caps) -> np.ndarray:
        out = []
        for i in range(0, len(caps), 64):
            chunk = [c if c.strip() else "a photo" for c in caps[i:i + 64]]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=64)
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            logits = self.model(**enc).logits[:, :-1]
            tgt = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].bool()
            lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lp = (lp * mask).sum(1) / mask.sum(1).clamp_min(1)
            out.append(lp.cpu().numpy())
        return np.concatenate(out)


def score_attn(enc, paths, caps):
    st = encode_att(enc, paths, caps)
    f = features(st, enc.tokenizer)
    cos = (st["pimg"] * st["ptxt"]).sum(1)
    return -cos + 0.5 * (1 - f["coverage_attn"])


def score_lm(enc, lm, paths, caps):
    st = enc.encode(paths, caps, batch_size=64)
    cos = (st.pooled_img * st.pooled_txt).sum(1)
    return -cos - lm.lam * lm.logp(caps)


def run(bench: str, split: str, variant: str, limit=None) -> dict:
    enc = JointEncoder(CLIP_DIR)
    lm = LMScorer() if variant == "lm" else None
    scorer = (lambda p, c: score_attn(enc, p, c)) if variant == "attn" else \
             (lambda p, c: score_lm(enc, lm, p, c))

    if bench == "foil":
        df = load_foil_val(limit) if split == "val" else load_foil_test(limit)
        n = len(df)
        sp = np.concatenate([scorer(df["path"].iloc[i:i + CHUNK].tolist(),
                                    df["positive"].iloc[i:i + CHUNK].tolist())
                             for i in range(0, n, CHUNK)])
        sn = np.concatenate([scorer(df["path"].iloc[i:i + CHUNK].tolist(),
                                    df["negative"].iloc[i:i + CHUNK].tolist())
                             for i in range(0, n, CHUNK)])
        d = sn - sp
        correct = np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))
        acc, lo, hi = bootstrap_acc(correct)
        payload = {"variant": variant, "bench": bench, "split": split,
                   "n_pairs": int(n), "accuracy": acc, "accuracy_ci95": [lo, hi],
                   "tie_rate": float((d == 0).mean()),
                   "threshold": THRESHOLDS[bench],
                   "gap": float(THRESHOLDS[bench] - acc),
                   "pass": bool(acc >= THRESHOLDS[bench])}
        pd.DataFrame({"pair_id": df["pair_id"], "score_pos": sp,
                      "score_neg": sn}).to_csv(
            RESULTS_DIR / "bench" / f"foil_{split}_{variant}.csv",
            index=False, encoding="utf-8-sig")
    else:
        df = load_split(bench, split, limit)
        s = np.concatenate([scorer(df["path"].iloc[i:i + CHUNK].tolist(),
                                   df["caption"].iloc[i:i + CHUNK].tolist())
                            for i in range(0, len(df), CHUNK)])
        auc, lo, hi = bootstrap_auc(df["label"].to_numpy(), s)
        payload = {"variant": variant, "bench": bench, "split": split,
                   "n": int(len(df)), "auroc": auc, "auroc_ci95": [lo, hi],
                   "threshold": THRESHOLDS[bench],
                   "gap": float(THRESHOLDS[bench] - auc),
                   "pass": bool(auc >= THRESHOLDS[bench])}
        pd.DataFrame({"image_id": df["image_id"], "label": df["label"],
                      "score": s}).to_csv(
            RESULTS_DIR / "bench" / f"{bench}_{split}_{variant}.csv",
            index=False, encoding="utf-8-sig")
    write_json(RESULTS_DIR / "bench" / f"{bench}_{split}_{variant}.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--variant", required=True, choices=["attn", "lm"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.bench, args.split, args.variant, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
