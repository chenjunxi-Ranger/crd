"""Merge two annotators, report agreement, list disagreements, compute enrichment.

Usage
  python scripts/merge_annotations.py --bench coco
  python scripts/merge_annotations.py --bench coco --final   (after adjudication)
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

TYPES = ["Entity", "Attribute", "Relation", "Quantity", "OCR", "Context", "Multi"]


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pa, pb = pd.Series(a).value_counts(normalize=True), pd.Series(b).value_counts(normalize=True)
    pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    blind = RESULTS_DIR / "audit" / "blind"
    a = pd.read_csv(blind / f"{args.bench}_form_A.csv", encoding="utf-8-sig")
    b = pd.read_csv(blind / f"{args.bench}_form_B.csv", encoding="utf-8-sig")
    m = a.merge(b, on=["anon_id", "image", "caption"], suffixes=("_A", "_B"))
    key = pd.read_csv(blind / f"{args.bench}_key.csv", encoding="utf-8-sig")
    m = m.merge(key[["anon_id", "group", "label"]], on="anon_id", how="left")

    valid = m["match_A"].isin(["match", "mismatch", "uncertain"]) & \
        m["match_B"].isin(["match", "mismatch", "uncertain"])
    out = {"bench": args.bench, "n": int(valid.sum()),
           "kappa_binary": kappa(m.loc[valid, "match_A"], m.loc[valid, "match_B"]),
           "agree_rate": float((m.loc[valid, "match_A"] == m.loc[valid, "match_B"]).mean())}

    disagree = m.loc[valid & (m["match_A"] != m["match_B"])]
    disagree.to_csv(blind / f"{args.bench}_adjudication.csv", index=False,
                    encoding="utf-8-sig")
    out["n_disagree"] = int(len(disagree))

    # per-type agreement on the rows where both flagged something
    rows = []
    for t in TYPES:
        at = m["err_type_A"].fillna("").str.contains(t, regex=False)
        bt = m["err_type_B"].fillna("").str.contains(t, regex=False)
        rows.append({"type": t, "n_A": int(at.sum()), "n_B": int(bt.sum()),
                     "kappa": kappa(at.astype(int), bt.astype(int))})
    out["per_type"] = rows
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    if args.final:
        # majority vote after adjudication (adjudicated rows overwrite via 'match_final')
        col = "match_final" if "match_final" in m else "match_A"
        m["final"] = m[col]
        enr = {}
        for g in ("persistent_hard", "both_correct"):
            sub = m[m["group"] == g]
            enr[g] = {"n": int(len(sub)),
                      "mismatch_rate": float((sub["final"] == "mismatch").mean()),
                      "types": {t: float(sub["err_type_A"].fillna("").str.contains(
                          t, regex=False).mean()) for t in TYPES}}
        out["enrichment"] = enr
        write_json(RESULTS_DIR / "bench" / f"annotation_{args.bench}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
