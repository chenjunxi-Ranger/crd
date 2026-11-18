"""Figure 1 (COCO Quantity rebuild): candidates, 2x2 cycle search, audit.

Scope: FGBench v1-strict, COCO Quantity TEST, CLIP ViT-B/16, RAW cosine (S).
Every row traces back to the same manifest via pair_id / cycle_id / image_id.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fgbench"
FIG = ROOT / "results" / "fig1_coco"
COCO_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\MS-COCO")
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
       "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def count_of(cap: str) -> int | None:
    m = re.search(r"\b(" + "|".join(NUM) + r"|\d+)\b", str(cap).lower())
    if not m:
        return None
    tok = m.group(1)
    return int(tok) if tok.isdigit() else NUM[tok]


def load() -> pd.DataFrame:
    m = pd.read_parquet(OUT / "crdbb_clip_COCO_Quantity_test.meta.parquet")
    z = np.load(OUT / "crdbb_clip_COCO_Quantity_test.npz", allow_pickle=True)
    df = m.copy()
    df["score"] = z["S"]
    df["image_path"] = [str(COCO_ROOT / f) for f in df.target_file]
    df["count"] = [count_of(c) for c in df.caption]
    return df


def candidates(df: pd.DataFrame, n: int = 36) -> pd.DataFrame:
    clean = df[df.pair_label == 0]
    noisy = df[df.pair_label == 1]
    rng = np.random.default_rng(0)
    rows, used_clean = [], set()
    for _, nr in noisy.sample(frac=1.0, random_state=0).iterrows():
        below = clean[(clean.score < nr.score - 0.02) & (clean.cycle_id != nr.cycle_id)]
        below = below[below["count"].notna() & ~below.pair_id.isin(used_clean)]
        if below.empty:
            continue
        # the sibling clean row of the noisy pair states the image's true count
        sib = df[(df.iid == nr.iid) & (df.pair_label == 0)]
        true_cnt = int(sib["count"].iloc[0]) if len(sib) and pd.notna(sib["count"].iloc[0]) else None
        cand = below
        if true_cnt:
            # prefer clean rows whose stated count also differs (a visible mismatch)
            pref = cand[cand["count"] != true_cnt]
            cand = pref if len(pref) else cand
        cr = cand.iloc[(cand.score - (nr.score - 0.08)).abs().argmin()]
        used_clean.add(cr.pair_id)
        rows.append({
            "noisy_true_count": true_cnt,
            "clean_pair_id": cr.pair_id, "clean_cycle_id": cr.cycle_id,
            "clean_image_id": cr.target_image_id, "clean_image_path": cr.image_path,
            "clean_caption": cr.caption, "clean_count": cr["count"],
            "clean_score": round(float(cr.score), 4),
            "noisy_pair_id": nr.pair_id, "noisy_cycle_id": nr.cycle_id,
            "noisy_image_id": nr.target_image_id, "noisy_image_path": nr.image_path,
            "noisy_caption": nr.caption, "noisy_count": nr["count"],
            "noisy_score": round(float(nr.score), 4),
            "gap": round(float(nr.score - cr.score), 4),
        })
        if len(rows) >= n * 2:
            break
    out = pd.DataFrame(rows)
    out = out[out.gap.between(0.02, 0.20)]
    out["pref"] = (out.gap - 0.07).abs()
    return out.sort_values("pref").head(n).reset_index(drop=True)


def cycle_table(df: pd.DataFrame) -> pd.DataFrame:
    """All 2x2 cycles with the four raw scores and the two swap differences."""
    rows = []
    for cid, g in df.groupby("cycle_id"):
        if len(g) != 4:
            continue
        imgs = g.iid.unique()
        caps = g.caption.unique()
        if len(imgs) != 2 or len(caps) != 2:
            continue
        s = {(r.iid, r.caption): r.score for r in g.itertuples()}
        lab = {(r.iid, r.caption): r.pair_label for r in g.itertuples()}
        # image A = the one whose score with T_A is the clean pair of the cycle
        # (use the first clean row as reference so labels are unambiguous)
        clean_rows = g[g.pair_label == 0]
        if len(clean_rows) != 2:
            continue
        a_iid = clean_rows.iloc[0].iid
        b_iid = [i for i in imgs if i != a_iid][0]
        ta = clean_rows.iloc[0].caption
        tb = [c for c in caps if c != ta][0]
        keys = [(a_iid, ta), (b_iid, ta), (a_iid, tb), (b_iid, tb)]
        if any(k not in s for k in keys):
            continue
        dI_a = s[(a_iid, ta)] - s[(b_iid, ta)]
        dI_b = s[(a_iid, tb)] - s[(b_iid, tb)]
        dT_a = s[(a_iid, ta)] - s[(a_iid, tb)]
        dT_b = s[(b_iid, ta)] - s[(b_iid, tb)]
        rows.append({
            "cycle_id": cid, "a_iid": str(a_iid), "b_iid": str(b_iid),
            "ta": ta, "tb": tb,
            "s_ata": float(s[(a_iid, ta)]), "s_bta": float(s[(b_iid, ta)]),
            "s_atb": float(s[(a_iid, tb)]), "s_btb": float(s[(b_iid, tb)]),
            "lab_ata": int(lab[(a_iid, ta)]), "lab_bta": int(lab[(b_iid, ta)]),
            "lab_atb": int(lab[(a_iid, tb)]), "lab_btb": int(lab[(b_iid, tb)]),
            "dI_ta": dI_a, "dI_tb": dI_b, "dT_ia": dT_a, "dT_ib": dT_b,
            "same_sign_img": (dI_a > 0) == (dI_b > 0),
            "same_sign_txt": (dT_a > 0) == (dT_b > 0),
            "sep": (s[(a_iid, ta)] + s[(b_iid, tb)]) / 2
                   - (s[(b_iid, ta)] + s[(a_iid, tb)]) / 2,
            "span": max(s.values()) - min(s.values()),
            "min_abs": min(abs(dI_a), abs(dI_b), abs(dT_a), abs(dT_b)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"rows {len(df)} cycles {df.cycle_id.nunique()} "
          f"score {df.score.min():.3f}-{df.score.max():.3f}")

    cand = candidates(df)
    cand.to_csv(FIG / "fig1_coco_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"candidates {len(cand)}  gap {cand.gap.min():.3f}-{cand.gap.max():.3f}")

    cyc = cycle_table(df)
    cyc.to_csv(FIG / "fig1_coco_cycles.csv", index=False, encoding="utf-8-sig")
    print(f"cycles usable {len(cyc)}")
    print(f"  same-sign image deltas: {int(cyc.same_sign_img.sum())}/{len(cyc)}")
    print(f"  same-sign text  deltas: {int(cyc.same_sign_txt.sum())}/{len(cyc)}")
    print(f"  both same-sign        : {int((cyc.same_sign_img & cyc.same_sign_txt).sum())}")

    # pick: both clean pairs score above both mismatched ones, readable spread
    # prefer cycles whose two image-swap (and two text-swap) differences share a
    # sign, then maximise readability (clear separation, visible spread)
    base = cyc[(cyc.sep > 0.03) & (cyc.span > 0.09)].copy()
    base["pref"] = ((base.sep - 0.08).abs() + (base.span - 0.16).abs()
                    - 0.05 * (base.same_sign_img.astype(int)
                              + base.same_sign_txt.astype(int)))
    best = base.sort_values("pref").iloc[0]
    g = df[df.cycle_id == best.cycle_id]
    pick = {
        "cycle_id": best.cycle_id,
        "images": [
            {"role": "I_A", "iid": str(best.a_iid),
             "image_id": g[g.iid == best.a_iid].target_image_id.iloc[0],
             "path": g[g.iid == best.a_iid].image_path.iloc[0]},
            {"role": "I_B", "iid": str(best.b_iid),
             "image_id": g[g.iid == best.b_iid].target_image_id.iloc[0],
             "path": g[g.iid == best.b_iid].image_path.iloc[0]},
        ],
        "captions": [best.ta, best.tb],
        "score_matrix": [[round(best.s_ata, 4), round(best.s_atb, 4)],
                         [round(best.s_bta, 4), round(best.s_btb, 4)]],
        "label_matrix": [[int(best.lab_ata), int(best.lab_atb)],
                         [int(best.lab_bta), int(best.lab_btb)]],
        "deltas": {"dI_ta": round(best.dI_ta, 4), "dI_tb": round(best.dI_tb, 4),
                   "dT_ia": round(best.dT_ia, 4), "dT_ib": round(best.dT_ib, 4)},
        "same_sign_img": bool(best.same_sign_img),
        "same_sign_txt": bool(best.same_sign_txt),
        "pair_ids": {r.pair_id: {"label": int(r.pair_label),
                                 "score": round(float(r.score), 4)}
                     for r in g.itertuples()},
    }
    (FIG / "fig1_coco_cycle.json").write_text(
        json.dumps(pick, ensure_ascii=False, indent=2), encoding="utf-8")
    print("picked", best.cycle_id, pick["score_matrix"], pick["label_matrix"])
    print("  deltas", pick["deltas"])

    # distribution data for the standalone analysis figure
    np.savez_compressed(FIG / "fig1_coco_dist.npz",
                        clean=df[df.pair_label == 0].score.to_numpy(),
                        noisy=df[df.pair_label == 1].score.to_numpy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
