"""Unified conditional residual-information audit over all tested candidates.

Same samples, same CLIP baseline, same matching rule for every candidate:
  * index = the BCSC run (first N rows of each val split)
  * candidate scores are aligned by row order and oriented so that
    HIGHER = MORE SUSPICIOUS
  * pairs: every noisy row is 1-NN matched to a clean row on s, kept if
    |s_clean - s_noisy| < eps
  * MCA(candidate) = P(susp_clean < susp_noisy | matched)
  * Oracle          = P(any candidate orders the pair correctly | matched)

Candidates
  CLIP       : -cos
  PCPES-like : rx + rc - 2cos                (CSLS form, bank-side)
  +prior     : v2.0 combination (bank + GPT-2)
  CDUS       : -D4
  BCSC       : R_{I->T}
  VES        : -E  (per-dataset selector, recomputed on the SAME rows)
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
sys.path.insert(0, str(ROOT / "scripts"))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402
from ves_train import Selector                            # noqa: E402

K_TOP = 10


def boot_ci(v: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0.005,0.01")
    args = ap.parse_args()
    eps_list = [float(x) for x in args.eps.split(",")]

    enc = JointEncoder(CLIP_DIR)
    dev = enc.device
    out = {}

    for bench in ("coco", "flickr", "mmimdb"):
        b = np.load(RESULTS_DIR / "gate" / f"bcsc_{bench}.npz")
        n = len(b["s"])
        bank = np.load(RESULTS_DIR / "gate" / f"final_feats2_{bench}.npz")
        d = np.load(RESULTS_DIR / "gate" / f"cdus_scores_{bench}.npz")
        y = b["label"]
        cos = b["s"]

        sel = Selector().to(dev)
        sel.load_state_dict(torch.load(RESULTS_DIR / "gate" / f"selector_{bench}.pt",
                                       map_location=dev))
        sel.eval()
        df = load_split(bench, "val", n)
        e = []
        for i in range(0, n, 500):
            st = enc.encode(df["path"].iloc[i:i + 500].tolist(),
                            df["caption"].iloc[i:i + 500].tolist(),
                            batch_size=64, num_workers=4)
            with torch.no_grad():
                s_ = sel(torch.as_tensor(st.patch, device=dev),
                         torch.as_tensor(st.pooled_txt, device=dev))
                e.append(s_.topk(K_TOP, dim=1).values.mean(1).cpu().numpy())
        e = np.concatenate(e)

        cand = {
            "CLIP": -cos,
            "PCPES_like": bank["rx"][:n] + bank["rc"][:n] - 2 * cos,
            "v2.0": (-cos + 6.0 * ((bank["rx"][:n] + bank["rc"][:n]) / 2 - cos)
                     + 0.05 * (-bank["lp"][:n])),
            "CDUS": -d["d4"][:n],
            "BCSC": b["rit"],
            "VES": -e,
        }

        clean = np.where(y == 0)[0]
        noisy = np.where(y == 1)[0]
        order = np.argsort(cos[clean])
        cs = cos[clean][order]
        p = np.searchsorted(cs, cos[noisy])
        left = np.clip(p - 1, 0, len(cs) - 1)
        right = np.clip(p, 0, len(cs) - 1)
        choose = np.where(np.abs(cs[left] - cos[noisy]) <= np.abs(cs[right] - cos[noisy]),
                          left, right)
        partner = order[choose]
        delta = np.abs(cos[clean][partner] - cos[noisy])

        res = {"n": int(n), "auc": {}, "mca": {}}
        for name, sc in cand.items():
            from sklearn.metrics import roc_auc_score
            res["auc"][name] = float(roc_auc_score(y, sc))
            res["mca"][name] = {}
            for eps in eps_list:
                keep = delta < eps
                ok = (sc[clean][partner][keep] < sc[noisy][keep]).astype(float)
                m, lo, hi = boot_ci(ok)
                res["mca"][name][str(eps)] = {"n": int(keep.sum()), "mca": m,
                                              "ci": [lo, hi], "delta": m - 0.5}
        # oracle over candidates on the matched pairs
        for eps in eps_list:
            keep = delta < eps
            mat = np.stack([(sc[clean][partner][keep] < sc[noisy][keep])
                            for sc in cand.values()])
            anyok = mat.any(0).astype(float)
            allok = mat.all(0).astype(float)
            m, lo, hi = boot_ci(anyok)
            res.setdefault("oracle", {})[str(eps)] = {
                "n": int(keep.sum()), "any_correct": m, "ci": [lo, hi],
                "all_correct": float(allok.mean()),
                "clip_correct_within_band": float(np.mean(cos[clean][partner][keep]
                                                          < cos[noisy][keep]))}
        out[bench] = res
        print(bench, json.dumps({"n": n, "auc": {k: round(v, 3) for k, v in res["auc"].items()}},
                                ensure_ascii=False), flush=True)
        for name in cand:
            print("  ", name, {k: (v["n"], round(v["mca"], 3),
                                   [round(x, 3) for x in v["ci"]])
                               for k, v in res["mca"][name].items()}, flush=True)
        print("   oracle", {k: (v["n"], round(v["any_correct"], 3),
                               round(v["all_correct"], 3),
                               round(v["clip_correct_within_band"], 3))
                            for k, v in res["oracle"].items()}, flush=True)

    write_json(RESULTS_DIR / "gate" / "residual_audit.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
