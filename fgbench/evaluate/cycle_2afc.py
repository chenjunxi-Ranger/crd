"""2AFC cycle accuracy: does the representation carry the pairing signal at all?

For every reciprocal cycle (A,ca) (B,cb) vs crossed (A,cb) (B,ca):
    acc = P[ s(A,ca)+s(B,cb) > s(A,cb)+s(B,ca) ]
A high value means the signal exists and a low AUROC reflects the scoring function,
not an unidentifiable construction.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH, jdump  # noqa: E402

VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
CACHE = ROOT / "results" / "fgbench"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="VG", choices=["VG", "COCO"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--tag", default="clip")
    args = ap.parse_args()

    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    enc = JointEncoder(args.model_dir or CLIP_DIR)

    d = pd.read_parquet(BENCH / f"{args.split}.parquet")
    d = d[d.dataset == args.dataset].copy()
    print("rows", len(d), "cycles", d.cycle_id.nunique(), flush=True)

    if args.dataset == "VG":
        idx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()
        d["iid"] = d.target_image_id.str.split(":").str[1].astype(int)
        d = d[d.iid.isin(idx)].reset_index(drop=True)
        key = d.iid.astype(str)
        paths = {k: str(VG_ROOT / idx[int(k)]) for k in key.unique()}
    else:
        from _cache import resolve
        key = d.target_image_id.astype(str)
        paths = {k: resolve(f) for k, f in zip(key, d.target_file)}

    cache = CACHE / f"emb_{args.tag}_{args.dataset}_{args.split}.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        emb = {k: v for k, v in zip(z["ids"].astype(str), z["vec"])}
        print("cache hit", len(emb), flush=True)
    else:
        ids = list(paths)
        t0 = time.time()
        vecs, emb = [], {}
        for k in range(0, len(ids), 2000):
            chunk = ids[k:k + 2000]
            v = enc.encode_images([paths[i] for i in chunk], batch_size=64, num_workers=4)
            vecs.append(v)
            for i, e in zip(chunk, v):
                emb[i] = e
            print(f"  encoded {k+len(chunk)}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(cache, ids=np.array(ids), vec=np.concatenate(vecs))

    V = np.stack([emb[k] for k in key])
    T = enc.encode_texts(d.caption.astype(str).tolist())
    d["s"] = (V * T).sum(1)
    d["k"] = key.values

    # one cycle -> two clean rows and two noisy rows sharing two images / two captions
    recs = []
    for cid, g in d.groupby("cycle_id"):
        if len(g) != 4:
            continue
        clean = g[g.pair_label == 0]
        noisy = g[g.pair_label == 1]
        if len(clean) != 2 or len(noisy) != 2:
            continue
        diag = clean.s.sum()
        cross = noisy.s.sum()
        recs.append({"cycle_id": cid, "facet": g.facet.iat[0],
                     "diag": diag, "cross": cross, "margin": diag - cross})
    c = pd.DataFrame(recs)
    print("cycles used", len(c), flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for facet in sorted(c.facet.unique()):
        sub = c[c.facet == facet]
        acc = float((sub.margin > 0).mean())
        vals = []
        idxs = np.arange(len(sub))
        for _ in range(args.boot):
            pick = rng.choice(idxs, len(idxs), replace=True)
            vals.append(float((sub.margin.to_numpy()[pick] > 0).mean()))
        rows.append({"facet": facet, "cycles": int(len(sub)), "acc2afc": acc,
                     "ci_lo": float(np.percentile(vals, 2.5)),
                     "ci_hi": float(np.percentile(vals, 97.5)),
                     "median_margin": float(sub.margin.median())})
        print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    p = CACHE / f"cycle2afc_{args.tag}_{args.dataset}_{args.split}.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    jdump({"dataset": args.dataset, "rows": rows}, p.with_suffix(".json"))
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
