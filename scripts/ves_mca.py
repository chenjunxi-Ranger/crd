"""MCA diagnostic: does E still rank correctly when CLIP cannot tell?

Matched-pair accuracy
    MCA(eps) = P(E_clean > E_noisy | |s_clean - s_noisy| < eps)

Row-level benches: noisy rows are 1-NN matched (on s) to clean rows of the same
benchmark; only matched pairs within eps are used.
FOIL: pairs are matched by construction (same image).

Stop rule (agreed with the user): if MCA is ~0.5 on the three row-level
benchmarks, the current patch-evidence selector carries no information
independent of the CLIP ordering, and no amount of extra data / capacity / a
hard-negative retrain is expected to fix it.

No training, no parameter update.
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

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402
from ves_train import Selector                            # noqa: E402

K_TOP = 10


def _boot_mca(correct: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(correct)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(correct[idx].mean())
    return float(np.mean(correct)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--eps", default="0.002,0.005,0.01,0.02")
    args = ap.parse_args()
    eps_list = [float(x) for x in args.eps.split(",")]

    enc = JointEncoder(CLIP_DIR)
    dev = enc.device
    out = {"eps": eps_list, "benches": {}}

    for bench in ("coco", "flickr", "mmimdb"):
        df = load_split(bench, "val", args.limit)
        y = df["label"].to_numpy()
        sel = Selector().to(dev)
        sel.load_state_dict(torch.load(RESULTS_DIR / "gate" / f"selector_{bench}.pt",
                                       map_location=dev))
        sel.eval()
        cos, energy = [], []
        for i in range(0, len(df), 500):
            st = enc.encode(df["path"].iloc[i:i + 500].tolist(),
                            df["caption"].iloc[i:i + 500].tolist(),
                            batch_size=64, num_workers=4)
            cos.append((st.pooled_img * st.pooled_txt).sum(1))
            with torch.no_grad():
                e = sel(torch.as_tensor(st.patch, device=dev),
                        torch.as_tensor(st.pooled_txt, device=dev))
                energy.append(e.topk(K_TOP, dim=1).values.mean(1).cpu().numpy())
        s = np.concatenate(cos)
        e = np.concatenate(energy)

        clean = np.where(y == 0)[0]
        noisy = np.where(y == 1)[0]
        res = {"n": int(len(df)), "n_clean": int(len(clean)), "n_noisy": int(len(noisy)),
               "auc_cos": float(roc_auc_score(y, -s)),
               "auc_energy": float(roc_auc_score(y, -e)),
               "by_eps": {}}
        for eps in eps_list:
            sc, sn = s[clean], s[noisy]
            order = np.argsort(sc)
            scs = sc[order]
            pos = np.searchsorted(scs, sn)
            left = np.clip(pos - 1, 0, len(scs) - 1)
            right = np.clip(pos, 0, len(scs) - 1)
            choose = np.where(np.abs(scs[left] - sn) <= np.abs(scs[right] - sn), left, right)
            partner = order[choose]
            delta = np.abs(sc[partner] - sn)
            keep = delta < eps
            if keep.sum() == 0:
                res["by_eps"][str(eps)] = {"n_pairs": 0}
                continue
            correct = (e[clean][partner][keep] > e[noisy][keep]).astype(float)
            m, lo, hi = _boot_mca(correct)
            res["by_eps"][str(eps)] = {"n_pairs": int(keep.sum()), "mca": m,
                                       "ci": [lo, hi],
                                       "mean_abs_ds": float(delta[keep].mean())}
        out["benches"][bench] = res
        print(bench, json.dumps({k: v for k, v in res.items() if k != "by_eps"},
                                ensure_ascii=False), flush=True)
        for k, v in res["by_eps"].items():
            print("   eps", k, json.dumps(v, ensure_ascii=False), flush=True)

    # FOIL: use the saved per-pair scores of the FOIL-specific selector
    p = RESULTS_DIR / "gate" / "ves_pairs_VES-1.npz"
    if p.is_file():
        with np.load(p) as z:
            sp, sn_ = z["s_pos"], z["s_neg"]
            ep, en = z["e_pos"], z["e_neg"]
        res = {"n_pairs": int(len(sp)), "by_eps": {}}
        for eps in eps_list:
            keep = np.abs(sp - sn_) < eps
            if keep.sum() == 0:
                res["by_eps"][str(eps)] = {"n_pairs": 0}
                continue
            correct = (ep[keep] > en[keep]).astype(float)
            m, lo, hi = _boot_mca(correct)
            res["by_eps"][str(eps)] = {"n_pairs": int(keep.sum()), "mca": m, "ci": [lo, hi]}
        out["benches"]["foil"] = res
        print("foil", json.dumps(res, ensure_ascii=False), flush=True)

    write_json(RESULTS_DIR / "gate" / "ves_mca.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
