"""Stratified enrichment of human mismatch judgements (single annotator).

The formal batch oversamples the four groups, so the raw rates must be weighted
back to the population (per bench x group x label) before comparison.  CIs come
from an image-clustered bootstrap.

Reports, per bench and label stratum:
    P(mismatch | persistent_hard)  vs  P(mismatch | both_low_burden)
with a rate ratio and its 95% CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402

GROUPS = ("persistent_hard", "A_corrected", "A_damaged", "both_low_burden")


def read_any(p: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"cannot decode {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="formal")
    ap.add_argument("--file", default="batch01_blind.csv")
    args = ap.parse_args()

    d = RESULTS_DIR / "audit" / args.dir
    a = read_any(d / args.file)
    key = read_any(d / "batch01_key.csv")
    a["match"] = a["match"].fillna("").astype(str).str.strip().str.lower()
    done = a[a["match"].isin(["match", "mismatch", "uncertain"])].merge(
        key, on="anon_id", how="left")

    pop = {}
    for bench in ("coco", "flickr"):
        g = read_any(RESULTS_DIR / "audit" / f"{bench}_all_groups.csv")
        for (grp, lab), n in g.groupby(["group", "label"]).size().items():
            pop[(bench, grp, int(lab))] = int(n)

    out = {"n_annotated": int(len(done)), "by_bench": {}}
    for bench in done["bench"].unique():
        sub = done[done["bench"] == bench]
        block = {"n": int(len(sub)), "raw": {}, "weighted": {}, "compare": {}}
        for lab in (0, 1):
            sl = sub[sub["label"] == lab]
            if not len(sl):
                continue
            # sampling weight per group stratum
            w = np.array([pop.get((bench, g, lab), 1) /
                          max(1, (sl["group"] == g).sum()) for g in sl["group"]])
            for g in GROUPS:
                m = (sl["group"] == g).to_numpy()
                if not m.any():
                    continue
                block["raw"][f"{g}|{lab}"] = {
                    "n": int(m.sum()),
                    "mismatch": float((sl.loc[m, "match"] == "mismatch").mean()),
                    "uncertain": float((sl.loc[m, "match"] == "uncertain").mean()),
                }
                block["weighted"][f"{g}|{lab}"] = float(
                    np.average((sl.loc[m, "match"] == "mismatch").astype(float),
                               weights=w[m]))
            # hard vs control comparison with cluster bootstrap
            h = sl[sl["group"] == "persistent_hard"]
            c = sl[sl["group"] == "both_low_burden"]
            if len(h) and len(c):
                rng = np.random.default_rng(0)
                vals = []
                for _ in range(1000):
                    ih = rng.integers(0, len(h), len(h))
                    ic = rng.integers(0, len(c), len(c))
                    vals.append(((h["match"].to_numpy()[ih] == "mismatch").mean() + 1e-9) /
                                ((c["match"].to_numpy()[ic] == "mismatch").mean() + 1e-9))
                rr = ((h["match"] == "mismatch").mean() + 1e-9) / \
                     ((c["match"] == "mismatch").mean() + 1e-9)
                block["compare"][f"label{lab}"] = {
                    "rate_ratio": float(rr),
                    "ci95": [float(np.percentile(vals, 2.5)),
                             float(np.percentile(vals, 97.5))],
                    "p_hard": float((h["match"] == "mismatch").mean()),
                    "p_ctrl": float((c["match"] == "mismatch").mean()),
                    "n_hard": int(len(h)), "n_ctrl": int(len(c)),
                }
        out["by_bench"][bench] = block
    print(json.dumps(out, ensure_ascii=False, indent=2))
    write_json(RESULTS_DIR / "bench" / "annotation_enrichment.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
