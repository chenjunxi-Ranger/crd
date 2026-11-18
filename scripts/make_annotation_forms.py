"""Generate per-annotator blank forms from the blind packet.

Each form has: anon_id, image, caption  (pre-filled, no group/label)
plus empty columns the annotator fills:
    match      : match / mismatch / uncertain     (binary semantic judgement)
    err_type   : comma-separated subset of
                 Entity, Attribute, Relation, Quantity, OCR, Context, Multi, Uncertain
    operation  : Add / Omit / Replace / Swap / unknown   (only if the generation
                 record confirms it, otherwise leave 'unknown')
    note       : free text
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="coco,flickr")
    ap.add_argument("--annotators", default="A,B")
    args = ap.parse_args()

    blind = RESULTS_DIR / "audit" / "blind"
    for bench in args.benches.split(","):
        src = pd.read_csv(blind / f"{bench}_blind.csv", encoding="utf-8-sig")
        for who in args.annotators.split(","):
            form = src.copy()
            form["match"] = ""
            form["err_type"] = ""
            form["operation"] = "unknown"
            form["note"] = ""
            out = blind / f"{bench}_form_{who}.csv"
            form.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"{out.name}: {len(form)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
