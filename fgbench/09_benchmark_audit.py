"""Step 09: benchmark audit — facet counts, text-length shortcut check, donor reuse."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgbench.common import BENCH, jdump  # noqa: E402


def main() -> int:
    d = pd.read_parquet(BENCH / "all.parquet")
    d["n_words"] = d.caption.map(lambda s: len(re.findall(r"[A-Za-z']+", str(s))))
    d["n_chars"] = d.caption.str.len()

    rec = {}
    for s in ("train", "val", "test"):
        sub = d[d.split == s]
        rec[s] = {"rows": int(len(sub))}
        for f in ("Entity", "Quantity"):
            a = sub[(sub.facet == f) & (sub.pair_label == 1)].n_words
            b = sub[(sub.facet == f) & (sub.pair_label == 0)].n_words
            rec[s][f] = {"noisy": int(len(a)), "clean": int(len(b)),
                         "noisy_words_mean": round(float(a.mean()), 2) if len(a) else None,
                         "clean_words_mean": round(float(b.mean()), 2) if len(b) else None,
                         "len_gap": round(float(a.mean() - b.mean()), 2)
                         if len(a) and len(b) else None}

    key = "donor_image_id" if "donor_image_id" in d.columns else "target_image_id"
    rec["reuse_key"] = key
    rec["donor_reuse"] = (d[d.pair_label == 1].groupby(key).size()
                          .describe().round(2).to_dict())
    rec["evidence_completeness"] = float(d[d.pair_label == 1].evidence.map(
        lambda e: isinstance(e, dict) and "fact" in e).mean())
    jdump(rec, BENCH / "audit.json")
    print({k: v for k, v in rec.items() if k != "donor_reuse"}, flush=True)
    print("donor reuse:", rec["donor_reuse"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
