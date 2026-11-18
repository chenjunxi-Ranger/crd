"""Figure 1 data extraction: candidate ranking conflicts, a 2x2 cycle, audit.

Scope is fixed: FGBench v1-strict, VG Attribute TEST, CLIP ViT-B/16, RAW cosine.
Everything is joined from one manifest so image_id / caption_id / cycle_id trace back.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fgbench"
FIG = ROOT / "results" / "fig1"
VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"


def load():
    meta = pd.read_parquet(OUT / "crdbb_clip_VG_Attribute_test.meta.parquet")
    z = np.load(OUT / "crdbb_clip_VG_Attribute_test.npz", allow_pickle=True)
    vg = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()
    meta = meta.copy()
    meta["score"] = z["S"]                      # RAW cosine, no standardisation
    meta["image_path"] = [str(VG_ROOT / vg[int(i)]) for i in meta.iid]
    return meta, vg


def candidate_pairs(df: pd.DataFrame, n_pairs: int = 40) -> pd.DataFrame:
    """clean-with-low-score vs noisy-with-high-score, different cycles/images."""
    clean = df[df.pair_label == 0]
    noisy = df[df.pair_label == 1]
    # non-extreme: stay inside the central 60% of each class
    c_lo, c_hi = clean.score.quantile([0.15, 0.85])
    n_lo, n_hi = noisy.score.quantile([0.15, 0.85])
    clean = clean[(clean.score >= c_lo) & (clean.score <= c_hi)]
    noisy = noisy[(noisy.score >= n_lo) & (noisy.score <= n_hi)]

    # candidate noisy rows that beat a reasonable share of clean rows
    rows = []
    clean_sorted = clean.sort_values("score")
    for _, nr in noisy.sample(frac=1.0, random_state=0).iterrows():
        below = clean_sorted[clean_sorted.score < nr.score]
        if len(below) < 200:            # must beat a meaningful fraction of clean
            continue
        cr = below.iloc[int(len(below) * 0.25)]   # a LOW-scoring clean pair
        if cr.cycle_id == nr.cycle_id or cr.iid == nr.iid:
            continue
        if nr.score - cr.score < 0.02:
            continue
        rows.append({
            "clean_pair_id": cr.pair_id, "clean_cycle_id": cr.cycle_id,
            "clean_image_id": cr.target_image_id, "clean_image_path": cr.image_path,
            "clean_caption_id": cr.caption_id, "clean_caption": cr.caption,
            "clean_score": round(float(cr.score), 4),
            "noisy_pair_id": nr.pair_id, "noisy_cycle_id": nr.cycle_id,
            "noisy_image_id": nr.target_image_id, "noisy_image_path": nr.image_path,
            "noisy_caption_id": nr.caption_id, "noisy_caption": nr.caption,
            "noisy_score": round(float(nr.score), 4),
            "gap": round(float(nr.score - cr.score), 4),
            "clean_pct_below": round(float(len(below) / len(clean)), 3),
        })
        if len(rows) >= n_pairs * 1.5:
            break
    out = pd.DataFrame(rows).sort_values("gap")
    # prefer moderate gaps (0.04-0.12) and balanced percentiles
    out["pref"] = (out.gap - 0.07).abs()
    return out.sort_values("pref").head(n_pairs).reset_index(drop=True)


def pick_cycle(df: pd.DataFrame) -> dict:
    """Choose a 2x2 cycle that reads cleanly: clean rows high, noisy low, moderate spread."""
    g = df.groupby("cycle_id")
    stats = g.agg(n=("pair_id", "size"), n_clean=("pair_label", lambda s: (s == 0).sum()),
                  mn=("score", "min"), mx=("score", "max"), sd=("score", "std"),
                  n_img=("target_image_id", "nunique"))
    stats = stats[(stats.n == 4) & (stats.n_clean == 2) & (stats.n_img == 2)]
    # separation between the two clean and the two noisy rows
    rows = []
    for cid, sub in df[df.cycle_id.isin(stats.index)].groupby("cycle_id"):
        c = sub[sub.pair_label == 0].score
        n = sub[sub.pair_label == 1].score
        sep = float(c.mean() - n.mean())
        span = float(sub.score.max() - sub.score.min())
        rows.append({"cycle_id": cid, "sep": sep, "span": span,
                     "clean_mean": float(c.mean()), "noisy_mean": float(n.mean())})
    s = pd.DataFrame(rows)
    # want clear separation but not an extreme outlier, and a visible span
    s = s[(s.sep > 0.05) & (s.span > 0.10)]
    s["pref"] = (s.sep - 0.09).abs() + (s.span - 0.16).abs()
    cid = s.sort_values("pref").iloc[0].cycle_id
    sub = df[df.cycle_id == cid].copy()

    a_img = sub[sub.pair_label == 0].iloc[0]        # image A: clean with text A
    ta = a_img.caption
    # the other image / caption in the same cycle
    b_row = sub[(sub.pair_label == 0) & (sub.caption != ta)].iloc[0]
    tb = b_row.caption
    mat = np.full((2, 2), np.nan)
    lab = np.full((2, 2), -1)
    meta = {}
    caps = [ta, tb]
    imgs = [a_img, b_row]
    for i, im in enumerate(imgs):
        for j, cap in enumerate(caps):
            r = sub[(sub.iid == im.iid) & (sub.caption == cap)]
            if len(r):
                mat[i, j] = float(r.iloc[0].score)
                lab[i, j] = int(r.iloc[0].pair_label)
                meta[(i, j)] = r.iloc[0].pair_id
    return {"cycle_id": cid, "captions": caps,
            "images": [{"image_id": im.target_image_id, "path": im.image_path,
                        "iid": int(im.iid), "pair_id": im.pair_id} for im in imgs],
            "score_matrix": mat.round(4).tolist(),
            "label_matrix": lab.tolist(),
            "pair_ids": {f"{k[0]}{k[1]}": v for k, v in meta.items()}}


def audit(df: pd.DataFrame) -> dict:
    img = df.groupby("target_image_id").agg(n=("pair_id", "size"),
                                            mean=("score", "mean"),
                                            n_clean=("pair_label", lambda s: (s == 0).sum()))
    txt = df.groupby("caption_id").agg(n=("pair_id", "size"),
                                       mean=("score", "mean"),
                                       n_clean=("pair_label", lambda s: (s == 0).sum()))
    img["n_noisy"] = img.n - img.n_clean
    txt["n_noisy"] = txt.n - txt.n_clean
    res = {
        "n_rows": int(len(df)), "n_images": int(df.iid.nunique()),
        "n_captions": int(df.caption_id.nunique()),
        "n_cycles": int(df.cycle_id.nunique()),
        "images_with_ge3": int((img.n >= 3).sum()),
        "captions_with_ge3": int((txt.n >= 3).sum()),
        "image_links_mean": round(float(img.n.mean()), 2),
        "caption_links_mean": round(float(txt.n.mean()), 2),
    }
    if (img.n >= 3).any():
        sub = img[img.n >= 3]
        res["image_mean_std_ge3"] = round(float(sub["mean"].std()), 4)
        res["image_mean_range_ge3"] = [round(float(sub["mean"].min()), 4),
                                       round(float(sub["mean"].max()), 4)]
        res["image_label_balance_ge3"] = round(
            float(sub.n_clean.sum() / sub.n.sum()), 3)
    if (txt.n >= 3).any():
        sub = txt[txt.n >= 3]
        res["text_mean_std_ge3"] = round(float(sub["mean"].std()), 4)
        res["text_mean_range_ge3"] = [round(float(sub["mean"].min()), 4),
                                      round(float(sub["mean"].max()), 4)]
        res["text_label_balance_ge3"] = round(float(sub.n_clean.sum() / sub.n.sum()), 3)
    return res, img, txt


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    df, _ = load()
    print(f"rows {len(df)}  cycles {df.cycle_id.nunique()}  "
          f"score range {df.score.min():.4f}-{df.score.max():.4f}")

    cand = candidate_pairs(df)
    cand.to_csv(FIG / "fig1_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"candidates: {len(cand)}  gap {cand.gap.min():.3f}-{cand.gap.max():.3f}")

    cyc = pick_cycle(df)
    (FIG / "fig1_cycle.json").write_text(json.dumps(cyc, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print("cycle", cyc["cycle_id"], cyc["score_matrix"], cyc["label_matrix"])

    res, img, txt = audit(df)
    (FIG / "fig1_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    img.sort_values("n", ascending=False).to_csv(FIG / "fig1_image_audit.csv",
                                                 encoding="utf-8-sig")
    txt.sort_values("n", ascending=False).to_csv(FIG / "fig1_text_audit.csv",
                                                 encoding="utf-8-sig")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
