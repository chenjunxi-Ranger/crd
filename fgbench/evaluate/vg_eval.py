"""CLIP cosine / CSLS on the VG facets of FGBench v1 (Attribute/Action/Relation)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import BENCH, jdump  # noqa: E402

VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"


def topk_mean(a, b, k=10):
    out = np.empty(len(a))
    for i in range(0, len(a), 1024):
        s = a[i:i + 1024] @ b.T
        kk = min(k, s.shape[1])
        out[i:i + 1024] = np.partition(s, -kk, axis=1)[:, -kk:].mean(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--bank", type=int, default=8000)
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--tag", default="clip")
    args = ap.parse_args()

    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    enc = JointEncoder(args.model_dir or CLIP_DIR)

    idx = pd.read_parquet(INDEX).set_index("image_id").rel_path.to_dict()
    te = pd.read_parquet(BENCH / f"{args.split}.parquet")
    te = te[te.dataset == "VG"].copy()
    te["iid"] = te.target_image_id.str.split(":").str[1].astype(int)
    te = te[te.iid.isin(idx)].reset_index(drop=True)
    print("rows", len(te), "unique images", te.iid.nunique(), flush=True)

    def paths(ids):
        return [str(VG_ROOT / idx[i]) for i in ids]

    uniq = te.iid.unique()
    t0 = time.time()
    emb = {}
    for k in range(0, len(uniq), 2000):
        chunk = uniq[k:k + 2000]
        v = enc.encode_images(paths(chunk), batch_size=64, num_workers=4)
        for i, e in zip(chunk, v):
            emb[i] = e
        print(f"  encoded {k+len(chunk)}/{len(uniq)} {time.time()-t0:.0f}s", flush=True)
    V = np.stack([emb[i] for i in te.iid])
    T = enc.encode_texts(te.caption.astype(str).tolist())

    tr = pd.read_parquet(BENCH / "train.parquet")
    tr = tr[tr.dataset == "VG"].drop_duplicates("target_image_id").head(args.bank).copy()
    tr["iid"] = tr.target_image_id.str.split(":").str[1].astype(int)
    tr = tr[tr.iid.isin(idx)]
    bi = enc.encode_images(paths(tr.iid.tolist()), batch_size=64, num_workers=4)
    bt = enc.encode_texts(tr.caption.astype(str).tolist())
    print("bank", bi.shape, flush=True)

    cos = (V * T).sum(1)
    ri = topk_mean(V, bt)
    rc = topk_mean(T, bi)

    rows = []
    for facet in ("Attribute", "Action", "Relation"):
        m = (te.facet == facet).to_numpy()
        if m.sum() < 20 or te.pair_label[m].nunique() < 2:
            continue
        y = te.pair_label[m].to_numpy()
        for name, s in (("cosine", -cos), ("csls", -cos + 0.8 * (ri + rc))):
            rows.append({"facet": facet, "detector": name, "n_noisy": int((y == 1).sum()),
                         "auc": float(roc_auc_score(y, s[m])),
                         "auprc": float(average_precision_score(y, s[m]))})
        print(facet, "cos", round(rows[-2]["auc"], 4), "csls", round(rows[-1]["auc"], 4),
              flush=True)

    out = pd.DataFrame(rows)
    p = ROOT / "results" / "fgbench" / f"vg_{args.tag}_{args.split}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False, encoding="utf-8-sig")
    jdump({"rows": int(len(te)), "images": int(te.iid.nunique()),
           "bank": int(len(tr)), "results": rows}, p.with_suffix(".json"))
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
