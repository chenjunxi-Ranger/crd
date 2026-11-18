"""Probe A001: is the token/patch space usable, and which projection is right?

Checks
------
1. Our pooled cosine vs the cached official CLIP embeddings (rank correlation).
2. Per-token support AUROC with ``text_projection`` applied to tokens vs raw
   token states.
3. Basic geometry: patch max/mean similarity, token count.

Analysis only; no labels enter ``src/``.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                 # noqa: E402
from src.features import compute, content_mask        # noqa: E402
from src.protocol import CLIP_DIR, PROJECT, load_split  # noqa: E402

CACHE_DIRS = {
    "coco": PROJECT / "mymodel" / "cache" / "mscoco",
    "flickr": PROJECT / "mymodel" / "cache" / "flickr30k",
    "mmimdb": PROJECT / "mymodel2" / "cache" / "mmimdb",
}


def _table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return dict(zip(z["keys"].astype(str).tolist(), z["emb"].astype(np.float32)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    df = load_split(args.bench, "val", args.limit)
    enc = JointEncoder(CLIP_DIR)
    cache = CACHE_DIRS[args.bench]
    img_tab, txt_tab = _table(cache / "image_emb.npz"), _table(cache / "text_emb.npz")
    ref_cos = np.array([
        float(img_tab[i] @ txt_tab[hashlib.md5(c.strip().encode()).hexdigest()])
        for i, c in zip(df["image_id"], df["caption"])
    ])

    for proj in (True, False):
        st = enc.encode(df["path"].tolist(), df["caption"].tolist(),
                        batch_size=32, num_workers=2, project_text=proj)
        cos = (st.pooled_img * st.pooled_txt).sum(1)
        feats = compute(st, enc.tokenizer, enc.grid)
        y = df["label"].to_numpy()
        print(f"project_text={proj}")
        print("  pooled cos vs cache: spearman=%.6f max|d|=%.2e"
              % (spearmanr(cos, ref_cos).statistic, np.abs(cos - ref_cos).max()))
        for k in ("margin_mean", "support_mean", "coverage_soft", "cycle_soft"):
            v = feats[k]
            print("  %-14s auc(-v)=%.4f  tie_rate=%.3f  mean=%.4f sd=%.4f"
                  % (k, roc_auc_score(y, -v),
                     float(pd.Series(v).duplicated(keep=False).mean()), v.mean(), v.std()))
        mask = content_mask(st, enc.tokenizer)
        print("  content tokens/row: mean=%.1f min=%d max=%d"
              % (mask.sum(1).mean(), mask.sum(1).min(), mask.sum(1).max()))
        s = st.token[0][mask[0]] @ st.patch[0].T
        print("  row0 sim: max=%.3f mean=%.3f p95=%.3f" % (s.max(), s.mean(), np.percentile(s, 95)))
        print("  row0 pooled cos=%.3f" % cos[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
