"""RSG diagnostics D1-D3: is reference stability a usable reliability signal?

h_b = rx_b + rc_b on a leave-one-fold-out reference bank (5 folds, fixed hash).
    h_med = median_b h_b,  d = median_b |h_b - h_med|,  g = |h_med|/(|h_med|+1.4826 d + eps)

D1  distribution of g (dynamic range, not all ~1)
D2  conditional MCA of the bank-side CSLS within g quintiles
D3  rescue / hurt of CSLS vs CLIP within g quintiles
+   matched-band coverage = #matched pairs / (#clean x #noisy)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402

K = 10
FOLDS = 5
EPS = 0.01


def topk_mean(q: np.ndarray, bank: np.ndarray, k: int = K, chunk: int = 256) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bt = torch.as_tensor(bank, device=dev)
    out = np.empty(len(q), dtype=np.float64)
    for i in range(0, len(q), chunk):
        sim = torch.as_tensor(q[i:i + chunk], device=dev) @ bt.T
        out[i:i + chunk] = torch.topk(sim, k, dim=1).values.mean(1).cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--limit", type=int, default=1200)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    df = load_split(args.bench, "val", args.limit)
    vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
    vt = enc.encode_texts(df["caption"].tolist())
    cos = (vi * vt).sum(1)
    y = df["label"].to_numpy()

    with np.load(RESULTS_DIR / "gate" / f"bank_{args.bench}_all.npz") as z:
        bi, bt = z["img"], z["txt"]
    n_bank = len(bi)
    fold = np.arange(n_bank) % FOLDS          # fixed, label-free

    h = np.zeros((FOLDS, len(df)))
    for b in range(FOLDS):
        keep = fold != b
        rx = topk_mean(vi, bt[keep])
        rc = topk_mean(vt, bi[keep])
        h[b] = rx + rc
    h_med = np.median(h, axis=0)
    d = np.median(np.abs(h - h_med[None, :]), axis=0)
    g = np.abs(h_med) / (np.abs(h_med) + 1.4826 * d + 1e-8)

    clsls = h_med - 2 * cos                   # suspicion, higher = more suspicious

    # matched pairs (same rule as the audit)
    clean, noisy = np.where(y == 0)[0], np.where(y == 1)[0]
    order = np.argsort(cos[clean])
    cs = cos[clean][order]
    p = np.searchsorted(cs, cos[noisy])
    left = np.clip(p - 1, 0, len(cs) - 1)
    right = np.clip(p, 0, len(cs) - 1)
    choose = np.where(np.abs(cs[left] - cos[noisy]) <= np.abs(cs[right] - cos[noisy]),
                      left, right)
    partner = order[choose]
    delta = np.abs(cos[clean][partner] - cos[noisy])
    keep = delta < EPS

    q = np.quantile(g[noisy][keep], [0.2, 0.4, 0.6, 0.8])
    quint = np.digitize(g[noisy][keep], q)
    ok = clsls[clean][partner][keep] < clsls[noisy][keep]      # clean less suspicious
    clip_ok = cos[clean][partner][keep] < cos[noisy][keep]

    res = {"bench": args.bench, "n": int(len(df)), "n_matched": int(keep.sum()),
           "coverage": float(keep.sum() / (len(clean) * len(noisy))),
           "g_quantiles": [float(x) for x in np.quantile(g, [0, .1, .25, .5, .75, .9, 1])],
           "g_mean": float(g.mean()), "g_std": float(g.std())}
    rows = []
    for k in range(5):
        sel = quint == k
        if sel.sum() == 0:
            continue
        rows.append({"quintile": k, "n": int(sel.sum()),
                     "g_mean": float(g[noisy][keep][sel].mean()),
                     "mca_csls": float(ok[sel].mean()),
                     "mca_clip": float(clip_ok[sel].mean()),
                     "rescue": int(((~clip_ok) & ok)[sel].sum()),
                     "hurt": int((clip_ok & (~ok))[sel].sum())})
    res["quintiles"] = rows
    write_json(RESULTS_DIR / "gate" / f"rsg_{args.bench}.json", res)
    np.savez_compressed(RESULTS_DIR / "gate" / f"rsg_{args.bench}.npz",
                        cos=cos, g=g, h=h_med, label=y)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
