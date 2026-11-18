"""Iteration 4 (C7 removed): one formula, one global constant set, four columns.

    s = -cos + alpha * ( (rx + rc)/2 - cos ) + beta * ( -ell_GPT2(c) )

* ``rx`` = mean top-k similarity of the image to the reference captions
* ``rc`` = mean top-k similarity of the caption to the reference images
* ``ell`` = GPT-2 mean token log-probability (C4 removed)

Stage 1 builds and caches per-benchmark arrays (cos / rx / rc / ell); Stage 2
sweeps (alpha, beta) with ONE set for all four columns.

Analysis only; labels are used to score the grid.
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

from src.encoders import JointEncoder                        # noqa: E402
from src.protocol import (CLIP_DIR, PROJECT, RESULTS_DIR, THRESHOLDS,  # noqa: E402
                          load_foil_val, load_split, write_json)

GPT2 = PROJECT / "LEMoN+CLIPsim" / "weights" / "gpt2"
CHUNK = 512
K = 10


@torch.no_grad()
def _logp(tok, model, caps, dev, batch=64, max_len=64):
    out = []
    for i in range(0, len(caps), batch):
        chunk = [c if c.strip() else "a photo" for c in caps[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len)
        enc = {k: v.to(dev) for k, v in enc.items()}
        logits = model(**enc).logits[:, :-1]
        tgt = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        out.append(((lp * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu().numpy())
    return np.concatenate(out)


def _topk_means(q: np.ndarray, bank: np.ndarray, k: int = K):
    """Return (mean of top-k, max) similarities against the bank."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bt = torch.as_tensor(bank, device=dev)
    mean = np.empty(len(q), dtype=np.float64)
    top1 = np.empty(len(q), dtype=np.float64)
    for s in range(0, len(q), CHUNK):
        sim = torch.as_tensor(q[s:s + CHUNK], device=dev) @ bt.T
        vals = torch.topk(sim, k, dim=1).values
        mean[s:s + CHUNK] = vals.mean(1).cpu().numpy()
        top1[s:s + CHUNK] = vals[:, 0].cpu().numpy()
    return mean, top1


def build(bench: str, limit: int | None) -> dict:
    cache = RESULTS_DIR / "gate" / f"final_feats2_{bench}.npz"
    if cache.is_file():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(GPT2))
    tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(str(GPT2)).to(dev).eval()
    enc = JointEncoder(CLIP_DIR)

    bank_path = RESULTS_DIR / "gate" / f"bank_{bench}_all.npz"
    with np.load(bank_path) as z:
        bi, bt = z["img"], z["txt"]

    if bench == "foil":
        df = load_foil_val(limit)
        n = len(df)
        paths = df["path"].tolist()
        vi = enc.encode_images(paths, batch_size=64, num_workers=4)
        cap_p, cap_n = df["positive"].tolist(), df["negative"].tolist()
        tp, tn = enc.encode_texts(cap_p), enc.encode_texts(cap_n)
        out = {}
        for tag, vt, caps in (("pos", tp, cap_p), ("neg", tn, cap_n)):
            out[f"cos_{tag}"] = (vi * vt).sum(1)
            out[f"rx_{tag}"], out[f"mxt_{tag}"] = _topk_means(vi, bt)
            out[f"rc_{tag}"], out[f"mxi_{tag}"] = _topk_means(vt, bi)
            out[f"lp_{tag}"] = _logp(tok, lm, caps, dev)
        out["label"] = np.zeros(n)
    else:
        df = load_split(bench, "val", limit)
        vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
        vt = enc.encode_texts(df["caption"].tolist())
        rx, mxt = _topk_means(vi, bt)
        rc, mxi = _topk_means(vt, bi)
        out = {"cos": (vi * vt).sum(1), "rx": rx, "rc": rc, "mxt": mxt, "mxi": mxi,
               "lp": _logp(tok, lm, df["caption"].tolist(), dev),
               "label": df["label"].to_numpy()}
    np.savez_compressed(cache, **out)
    return out


def score(f: dict, alpha: float, beta: float, gamma: float = 0.0,
          tag: str | None = None) -> np.ndarray:
    sfx = "" if tag is None else f"_{tag}"
    cos, rx, rc = f["cos" + sfx], f["rx" + sfx], f["rc" + sfx]
    mxt, mxi, lp = f["mxt" + sfx], f["mxi" + sfx], f["lp" + sfx]
    return (-cos + alpha * ((rx + rc) / 2 - cos)
            + gamma * ((mxt - cos) + (mxi - cos)) / 2 + beta * (-lp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "grid"])
    ap.add_argument("--bench", default="coco")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.stage == "build":
        f = build(args.bench, args.limit)
        print({k: float(np.mean(v)) for k, v in f.items() if k != "label"})
        return 0

    feats = {b: build(b, None) for b in ("coco", "flickr", "mmimdb", "foil")}
    rows = []
    for alpha in (2.0, 4.0, 6.0):
        for gamma in (0.0, 0.5, 1.0, 2.0):
            for beta in (0.0, 0.02, 0.05):
                res = {}
                for b in ("coco", "flickr", "mmimdb"):
                    s = score(feats[b], alpha, beta, gamma)
                    res[b] = float(roc_auc_score(feats[b]["label"], s))
                sp = score(feats["foil"], alpha, beta, gamma, "pos")
                sn = score(feats["foil"], alpha, beta, gamma, "neg")
                d = sn - sp
                res["foil"] = float(np.mean(np.where(d > 0, 1.0,
                                                     np.where(d < 0, 0.0, 0.5))))
                res["alpha"], res["beta"], res["gamma"] = alpha, beta, gamma
                res["min_margin"] = min(res["coco"] - 0.96, res["flickr"] - 0.96,
                                        res["mmimdb"] - 0.88, res["foil"] - 0.92)
                res["n_pass"] = sum([res["coco"] >= 0.96, res["flickr"] >= 0.96,
                                     res["mmimdb"] >= 0.88, res["foil"] >= 0.92])
                rows.append(res)
    df = pd.DataFrame(rows).sort_values(["n_pass", "min_margin"], ascending=False)
    df.to_csv(RESULTS_DIR / "gate" / "final_grid.csv", index=False, encoding="utf-8-sig")
    print(df.head(15).to_string(index=False))
    best = df.iloc[0].to_dict()
    write_json(RESULTS_DIR / "gate" / "final_grid_best.json",
               {"best": best, "selection": "one global (alpha,beta) for all four columns"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
