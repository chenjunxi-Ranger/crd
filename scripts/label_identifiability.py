"""Label identifiability audit.

Split noisy samples by the single-annotator semantic judgement:
    semantic-incompatible noisy : provenance noisy AND human says mismatch
    semantic-compatible   noisy : provenance noisy AND human says match
and compare whether Cosine / CSLS can separate clean from each subgroup.

Decision rules (pre-registered):
    A  incompatible detectable, compatible ~chance  -> detectors track semantics
    B  compatible still detectable                  -> there is extra provenance signal
    C  neither detectable                           -> benchmark cannot support the goal
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


def boot_auc(score, y, groups, n=1000, seed=0):
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by[g] for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    d = RESULTS_DIR / "audit" / "formal"
    ann = read_any(d / "batch01_filled.csv")
    key = read_any(d / "batch01_key.csv")
    a = ann.merge(key, on="anon_id", how="left")
    a["match"] = a["match"].fillna("").astype(str).str.strip().str.lower()
    a = a[a["match"].isin(["match", "mismatch", "uncertain"])]

    out = {"n_annotated": int(len(a)), "by_bench": {}}
    for bench in ("coco", "flickr"):
        z = np.load(RESULTS_DIR / "seed_audit" / f"seed_{args.seed}" /
                    f"val_scores_{bench}.npz", allow_pickle=True)
        s = pd.DataFrame({"sample_id": z["sample_id"], "image_id": z["image_id"].astype(str),
                          "label": z["label"], "cos": z["cos"],
                          "score": z["score"]})          # score = -cos + 0.8(rx+rc)
        sub = a[a["bench"] == bench].merge(s, on="sample_id", suffixes=("", "_s"))
        if not len(sub):
            continue
        clean = s[s["label"] == 0]
        nois = sub[sub["label"] == 1]
        inc = nois[nois["match"] == "mismatch"]
        com = nois[nois["match"] == "match"]
        unc = nois[nois["match"] == "uncertain"]

        def auc_against_clean(part, col, sign=1):
            # cos: higher = cleaner -> flip.  score (anomaly): higher = noisier -> keep.
            if len(part) < 15:
                return None, None
            sc = np.concatenate([sign * clean[col].to_numpy(),
                                 sign * part[col].to_numpy()])
            yy = np.concatenate([np.zeros(len(clean)), np.ones(len(part))])
            gg = np.concatenate([clean["image_id"].astype(str).to_numpy(),
                                 part["image_id"].astype(str).to_numpy()])
            v = float(roc_auc_score(yy, sc))
            lo, hi = boot_auc(sc, yy, gg)
            return v, [lo, hi]

        rec = {"n_clean": int(len(clean)), "n_noisy_annot": int(len(nois)),
               "n_incompat": int(len(inc)), "n_compat": int(len(com)),
               "n_uncertain": int(len(unc))}
        for name, part in (("incompat", inc), ("compat", com)):
            for col, tag, sg in (("cos", "cos", -1), ("score", "csls", 1)):
                v, ci = auc_against_clean(part, col, sg)
                rec[f"auc_{tag}_{name}"] = v
                rec[f"ci_{tag}_{name}"] = ci
        out["by_bench"][bench] = rec
        print(bench, rec, flush=True)

    # pooled decision rule
    dec = []
    for b, r in out["by_bench"].items():
        a1, a2 = r.get("auc_csls_incompat"), r.get("auc_csls_compat")
        if a1 is None or a2 is None:
            continue
        dec.append((b, a1, a2))
    out["decision_hint"] = dec
    print("decision", dec, flush=True)
    write_json(RESULTS_DIR / "bench" / "label_identifiability.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
