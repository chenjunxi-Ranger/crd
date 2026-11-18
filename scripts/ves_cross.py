"""Cross-benchmark check for the trained VES selector (row-level AUROC).

Applies the SAME selector, the SAME K and the SAME lambda to COCO / Flickr /
MM-IMDb val.  This is the C1 test: if VES only helps FOIL, the single-formula
claim fails.

No parameter is updated here; the selector is frozen.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--selector", default="VES-1")
    ap.add_argument("--chunk", type=int, default=1000)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    sel = Selector().to(enc.device)
    state = torch.load(RESULTS_DIR / "gate" / f"selector_{args.selector}.pt",
                       map_location=enc.device)
    sel.load_state_dict(state)
    sel.eval()

    out = {"selector": args.selector, "k_top": K_TOP, "benches": {}}
    for bench in ("coco", "flickr", "mmimdb"):
        df = load_split(bench, "val", args.limit)
        y = df["label"].to_numpy()
        cos, energy = [], []
        for i in range(0, len(df), args.chunk):
            st = enc.encode(df["path"].iloc[i:i + args.chunk].tolist(),
                            df["caption"].iloc[i:i + args.chunk].tolist(),
                            batch_size=64, num_workers=4)
            cos.append((st.pooled_img * st.pooled_txt).sum(1))
            with torch.no_grad():
                v = torch.as_tensor(st.patch, device=enc.device)
                t = torch.as_tensor(st.pooled_txt, device=enc.device)
                e = sel(v, t)
                energy.append(e.topk(K_TOP, dim=1).values.mean(1).cpu().numpy())
        cos = np.concatenate(cos)
        energy = np.concatenate(energy)
        res = {"n": int(len(df)),
               "auc_cosine": float(roc_auc_score(y, -cos)),
               "auc_energy_only": float(roc_auc_score(y, -energy)),
               "lambda_sweep": {}}
        for lam in (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
            s = -cos - lam * energy
            res["lambda_sweep"][str(lam)] = float(roc_auc_score(y, s))
        out["benches"][bench] = res
        print(bench, json.dumps(res, ensure_ascii=False), flush=True)

    write_json(RESULTS_DIR / "gate" / "ves_cross.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
