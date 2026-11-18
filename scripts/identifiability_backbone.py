"""Same-sample backbone comparison on the annotated subsets.

Scores (all anomaly-oriented, higher = more suspicious):
    CLIP   cosine : -cos                                (seed_audit npz)
    CLIP   CSLS   : -cos + 0.8 (rx + rc)                (seed_audit npz)
    FG     cosine : -cos                                (fg_csls npz)
    FG     CSLS   : -cos + 0.8 (rI + rc)                (fg_csls npz)

Reports AUC(clean vs incompat) and AUC(clean vs compat) per backbone, plus
    Delta_compat, Delta_incompat, Delta_interaction
with image_id-clustered paired bootstrap (the same resample for all four scores).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402


def read_any(p: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"cannot decode {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    d = RESULTS_DIR / "audit" / "formal"
    ann = read_any(d / "batch01_filled.csv")
    key = read_any(d / "batch01_key.csv")
    a = ann.merge(key, on="anon_id", how="left")
    a["match"] = a["match"].fillna("").astype(str).str.strip().str.lower()
    a = a[a["match"].isin(["match", "mismatch"])]

    out = {}
    for bench in ("coco", "flickr"):
        zc = np.load(RESULTS_DIR / "seed_audit" / f"seed_{args.seed}" /
                     f"val_scores_{bench}.npz", allow_pickle=True)
        zf = np.load(RESULTS_DIR / "gate" / f"fg_csls_{bench}.npz", allow_pickle=True)
        assert len(zc["label"]) == len(zf["label"]), "row mismatch"

        clip_cos = -zc["cos"]
        clip_csls = zc["score"]                       # already -cos + 0.8(rx+rc)
        fg_cos = -zf["cos"]
        fg_csls = -zf["cos"] + 0.8 * (zf["ri"] + zf["rc"])
        y = zc["label"]
        gid = zc["image_id"].astype(str)

        sub = a[(a["bench"] == bench) & (a["label"] == 1)]   # noisy rows only
        inc = set(sub.loc[sub["match"] == "mismatch", "sample_id"].astype(int))
        com = set(sub.loc[sub["match"] == "match", "sample_id"].astype(int))
        clean = set(np.where(y == 0)[0].tolist())
        inc = sorted(inc); com = sorted(com); clean = sorted(clean)

        def auc_for(idx_clean, idx_part, sc):
            sel = np.array(idx_clean + idx_part)
            yy = np.r_[np.zeros(len(idx_clean)), np.ones(len(idx_part))]
            return float(roc_auc_score(yy, sc[sel]))

        scores = {"clip_cos": clip_cos, "clip_csls": clip_csls,
                  "fg_cos": fg_cos, "fg_csls": fg_csls}
        rec = {"n_clean": len(clean), "n_incompat": len(inc), "n_compat": len(com)}
        for nm, sc in scores.items():
            rec[f"auc_{nm}_incompat"] = auc_for(clean, inc, sc)
            rec[f"auc_{nm}_compat"] = auc_for(clean, com, sc)

        # paired bootstrap over image clusters
        uniq = np.unique(gid)
        by = {g: np.where(gid == g)[0] for g in uniq}
        rng = np.random.default_rng(0)
        deltas = {"d_compat": [], "d_incompat": [], "d_inter": []}
        for _ in range(args.boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            keep = set(np.concatenate([by[g] for g in pick]).tolist())
            c2 = [i for i in clean if i in keep]
            i2 = [i for i in inc if i in keep]
            m2 = [i for i in com if i in keep]
            if len(c2) < 30 or len(i2) < 8 or len(m2) < 8:
                continue
            a_clip_i = auc_for(c2, i2, clip_csls); a_fg_i = auc_for(c2, i2, fg_csls)
            a_clip_c = auc_for(c2, m2, clip_csls); a_fg_c = auc_for(c2, m2, fg_csls)
            deltas["d_compat"].append(a_fg_c - a_clip_c)
            deltas["d_incompat"].append(a_fg_i - a_clip_i)
            deltas["d_inter"].append((a_fg_c - a_clip_c) - (a_fg_i - a_clip_i))
        for k, v in deltas.items():
            rec[k] = float(np.mean(v)) if v else None
            rec[k + "_ci"] = ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                              if v else None)
        out[bench] = rec
        print(bench, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                      if not k.endswith("_ci")}, flush=True)
    write_json(RESULTS_DIR / "bench" / "identifiability_backbone.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
