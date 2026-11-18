"""Population statistics behind Figure 1: swap effects, distributions, AUROC.

Every cycle contributes exactly four raw scores, so swapping the image (text held
fixed) and swapping the text (image held fixed) are both computable per cycle.
These are DESCRIPTIVE score differences; they are not pure modality offsets,
because the match/mismatch relation changes at the same time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fgbench"
FIG = ROOT / "results" / "fig1"


def main() -> int:
    meta = pd.read_parquet(OUT / "crdbb_clip_VG_Attribute_test.meta.parquet")
    z = np.load(OUT / "crdbb_clip_VG_Attribute_test.npz", allow_pickle=True)
    df = meta.copy()
    df["score"] = z["S"]

    # ---- pivot each cycle into a 2x2 table ----
    imgs = df.groupby("cycle_id").iid.apply(lambda s: sorted(s.unique()))
    caps = df.groupby("cycle_id").caption.apply(lambda s: sorted(s.unique()))
    ok = imgs[(imgs.str.len() == 2)] .index
    ok = [c for c in ok if c in caps.index and len(caps[c]) == 2]
    sub = df[df.cycle_id.isin(ok)]

    d_img, d_txt, d_match = [], [], []
    for cid, g in sub.groupby("cycle_id", sort=False):
        s = {(r.iid, r.caption): r.score for r in g.itertuples()}
        l = {(r.iid, r.caption): r.pair_label for r in g.itertuples()}
        i1, i2 = imgs[cid]
        t1, t2 = caps[cid]
        keys = [(i1, t1), (i1, t2), (i2, t1), (i2, t2)]
        if any(k not in s for k in keys):
            continue
        # swap image, text fixed
        d_img += [s[(i1, t1)] - s[(i2, t1)], s[(i1, t2)] - s[(i2, t2)]]
        # swap text, image fixed
        d_txt += [s[(i1, t1)] - s[(i1, t2)], s[(i2, t1)] - s[(i2, t2)]]
        clean = [s[k] for k in keys if l[k] == 0]
        noisy = [s[k] for k in keys if l[k] == 1]
        if len(clean) == 2 and len(noisy) == 2:
            d_match.append(np.mean(clean) - np.mean(noisy))

    d_img, d_txt, d_match = map(np.array, (d_img, d_txt, d_match))
    # detector convention: a low score means "noisy", so the anomaly score is -cos
    auroc = float(roc_auc_score(df.pair_label, -df.score))
    clean_s = df[df.pair_label == 0].score.to_numpy()
    noisy_s = df[df.pair_label == 1].score.to_numpy()

    res = {
        "facet": "VG Attribute (FGBench v1-strict, test)",
        "backbone": "CLIP ViT-B/16, raw cosine",
        "n_rows": int(len(df)), "n_cycles": int(df.cycle_id.nunique()),
        "auroc_raw": round(auroc, 4),
        "clean": {"n": int(len(clean_s)), "mean": round(float(clean_s.mean()), 4),
                  "sd": round(float(clean_s.std()), 4)},
        "noisy": {"n": int(len(noisy_s)), "mean": round(float(noisy_s.mean()), 4),
                  "sd": round(float(noisy_s.std()), 4)},
        "swap_image_abs_mean": round(float(np.abs(d_img).mean()), 4),
        "swap_text_abs_mean": round(float(np.abs(d_txt).mean()), 4),
        "swap_image_abs_p90": round(float(np.percentile(np.abs(d_img), 90)), 4),
        "swap_text_abs_p90": round(float(np.percentile(np.abs(d_txt), 90)), 4),
        "match_gap_mean": round(float(d_match.mean()), 4),
        "match_gap_negative_share": round(float((d_match <= 0).mean()), 4),
        # P(noisy score > clean score) over all clean/noisy combinations
        "p_noisy_above_clean": round(1.0 - auroc, 4),
    }

    (FIG / "fig1_stats.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    np.savez_compressed(FIG / "fig1_dist.npz", clean=clean_s, noisy=noisy_s,
                        d_img=d_img, d_txt=d_txt)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
