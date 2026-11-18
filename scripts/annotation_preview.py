"""Preview a single annotator's sheet: match rates and error types by group.

Handles GBK files saved by Excel. No adjudication needed for a preview.
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


def read_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"cannot decode {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco")
    ap.add_argument("--who", default="A")
    args = ap.parse_args()

    blind = RESULTS_DIR / "audit" / "blind"
    df = read_any(blind / f"{args.bench}_form_{args.who}.csv")
    key = read_any(blind / f"{args.bench}_key.csv")
    df["match"] = df["match"].fillna("").astype(str).str.strip().str.lower()
    done = df[df["match"].isin(["match", "mismatch", "uncertain"])].copy()
    done = done.merge(key[["anon_id", "group", "label"]], on="anon_id", how="left")

    out = {"bench": args.bench, "annotator": args.who,
           "n_filled": int(len(done)), "n_total": int(len(df))}
    out["overall"] = done["match"].value_counts().to_dict()
    out["by_group"] = {}
    for g in ("persistent_hard", "A_damaged", "A_corrected", "both_low_burden"):
        sub = done[done["group"] == g]
        if not len(sub):
            continue
        out["by_group"][g] = {
            "n": int(len(sub)),
            "mismatch_rate": float((sub["match"] == "mismatch").mean()),
            "match_rate": float((sub["match"] == "match").mean()),
            "types": {t: int(sub["err_type"].fillna("").str.contains(t, regex=False).sum())
                      for t in TYPES},
        }
    hard = done[done["group"] == "persistent_hard"]
    ctrl = done[done["group"] == "both_low_burden"]
    if len(hard) and len(ctrl):
        out["enrichment"] = {
            "mismatch_hard": float((hard["match"] == "mismatch").mean()),
            "mismatch_ctrl": float((ctrl["match"] == "mismatch").mean()),
            "ratio": float(((hard["match"] == "mismatch").mean() + 1e-9) /
                           ((ctrl["match"] == "mismatch").mean() + 1e-9)),
            "types": {t: [int(ctrl["err_type"].fillna("").str.contains(t, regex=False).sum()),
                          int(hard["err_type"].fillna("").str.contains(t, regex=False).sum())]
                      for t in TYPES},
        }
    # validity check: human mismatch vs the dataset label
    out["vs_label"] = {
        "mismatch_rate_label_noisy": float((done.loc[done["label"] == 1, "match"] == "mismatch").mean())
        if (done["label"] == 1).any() else None,
        "mismatch_rate_label_clean": float((done.loc[done["label"] == 0, "match"] == "mismatch").mean())
        if (done["label"] == 0).any() else None,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    write_json(RESULTS_DIR / "bench" / f"annotation_preview_{args.bench}_{args.who}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
