"""Layer-2: zero-shot transfer of a FGBench-trained CRD head to SugarCrepe.

Nothing is re-fitted: mu, sigma, the head weights and lambda all come from the
FGBench COCO Entity run. SugarCrepe has no train split, so this is the only
honest protocol for it.

Metric: the benchmark's own within-pair accuracy (a correct caption must score
lower than its hard negative). Row-level AUROC is reported as a secondary number
with the caveat that it mixes cross-image variation, which SugarCrepe's design
does not control for.

Run:  python fgbench/experiments/external/run_sugarcrepe.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fgbench.models.crd_backbone import Off  # noqa: E402

DATA = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\derived\sugarcrepe"
            r"\sugarcrepe__comp__lvl0.50__seed0.csv")
CACHE = ROOT / "results" / "fgbench"
OUT = CACHE / "external"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT = 1000


def encode(df: pd.DataFrame):
    p = OUT / "feat_sugarcrepe.pkl"
    if p.is_file():
        with open(p, "rb") as f:
            return pickle.load(f)
    from fgbench.models.crd_backbone import ClipBB
    bb = ClipBB()
    t0 = time.time()
    U = bb.images(df.path.tolist())
    V = bb.texts(df.sentence.tolist())
    print(f"  encoded {len(U)} rows in {time.time()-t0:.0f}s", flush=True)
    with open(p, "wb") as f:
        pickle.dump((U, V), f)
    return U, V


def main() -> int:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    print(f"{len(df)} rows, {df.image_id.nunique()} images, "
          f"{df.pair_id.nunique()} pairs")
    U, V = encode(df)
    S = np.einsum("ij,ij->i", U, V)

    # FGBench COCO Entity head, unmodified
    prov = json.loads((CACHE / "crdbb_prov_clip_COCO_Entity_both_row_s0.json")
                      .read_text(encoding="utf-8"))
    mu, sd = float(prov["mu"]), float(prov["sd"])
    ck = torch.load(CACHE / "crdbb_clip_COCO_Entity_both_row_s0_head.pt",
                    map_location="cpu", weights_only=False)
    m = Off(d=int(ck["dim"]), parts="both")
    m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad():
        q = m(torch.tensor(U.astype(np.float32)),
              torch.tensor(V.astype(np.float32))).numpy().ravel()
    lam = float(np.load(CACHE / "crdbb_scores_clip_COCO_Entity_both_row_s0.npz",
                        allow_pickle=True)["lam"][0])

    Z = (S - mu) / sd
    crd = Z - lam * q

    df = df.assign(cos=S, z=Z, q=q, crd=crd)
    pos = df[df.is_mislabel == 0].set_index("pair_id")
    neg = df[df.is_mislabel == 1].set_index("pair_id")
    idx = pos.index.intersection(neg.index)

    rng = np.random.default_rng(0)
    rows = []
    # Both scores live in similarity space (higher = cleaner), so a CORRECT
    # caption must score HIGHER than its hard negative. (The convention in
    # derived/BENCHMARKS.md is written for detector scores, where the inequality
    # flips.)
    for name, col in (("cosine", "z"), ("crd_zero_shot", "crd")):
        sp, sn = pos.loc[idx, col].to_numpy(), neg.loc[idx, col].to_numpy()
        acc = float((sp > sn).mean())
        # bootstrap over pairs, clustered by image
        img = pos.loc[idx, "image_id"].to_numpy()
        by = pd.Series(np.arange(len(idx))).groupby(img).apply(
            lambda s: s.to_numpy()).to_dict()
        ids = list(by)
        bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(ids, len(ids), replace=True)
            ii = np.concatenate([by[i] for i in pick])
            bs.append((sp[ii] > sn[ii]).mean())
        rows.append({"scorer": name, "pair_accuracy": round(acc, 4),
                     "ci_lo": round(float(np.percentile(bs, 2.5)), 4),
                     "ci_hi": round(float(np.percentile(bs, 97.5)), 4)})
        print(rows[-1], flush=True)

    # sanity check: is the gain monotone in lambda, or a knife-edge?
    sweep = []
    for L in np.round(np.arange(0, 1.01, 0.1), 2):
        s = Z - L * q
        sp = pd.Series(s, index=df.index)[df.is_mislabel == 0].groupby(
            df.pair_id[df.is_mislabel == 0]).first()
        sn = pd.Series(s, index=df.index)[df.is_mislabel == 1].groupby(
            df.pair_id[df.is_mislabel == 1]).first()
        j = sp.index.intersection(sn.index)
        sweep.append({"lam": float(L),
                      "pair_accuracy": round(float((sp[j] > sn[j]).mean()), 4)})
    print("\n== lambda sweep (no labels used) ==")
    print(pd.DataFrame(sweep).to_string(index=False))

    # per-alteration accuracy, to see where the two disagree
    alt = []
    for a, g in pos.loc[idx].assign(neg=neg.loc[idx, "z"]).groupby("alteration"):
        alt.append({"alteration": a, "n": len(g),
                    "cos": round(float((g.z > g.neg).mean()), 4)})
    alt_df = pd.DataFrame(alt).sort_values("n", ascending=False)

    out = {"lambda": lam, "mu": mu, "sd": sd, "n_pairs": int(len(idx)),
           "results": rows}
    (OUT / "sugarcrepe_zeroshot.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n== summary ==")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n== per alteration (cosine) ==")
    print(alt_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
