"""Audit 14: verify every Quantity label against the annotations, and break the
test set down by the (target count, donor count) pair."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import BENCH, INTER  # noqa: E402


def main() -> int:
    d = pd.read_parquet(BENCH / "test.parquet")
    q = d[(d.facet == "Quantity") & (d.pair_label == 1)].copy()
    ok_caption_eq_donor = ok_ne_target = ok_target_pos = 0
    pairs = Counter()
    for r in q.itertuples():
        e = r.evidence
        n_cap, n_don, n_tgt = e["caption_number"], e["donor_count"], e["target_count"]
        ok_caption_eq_donor += int(n_cap == n_don)
        ok_ne_target += int(n_cap != n_tgt)
        ok_target_pos += int(n_tgt > 0)
        pairs[(int(n_tgt), int(n_don))] += 1
    n = len(q)
    out = {"n_quantity_test": n,
           "caption_eq_donor": ok_caption_eq_donor / n,
           "caption_ne_target": ok_ne_target / n,
           "target_count_positive": ok_target_pos / n,
           "count_pairs": {f"{a}->{b}": c for (a, b), c in
                           sorted(pairs.items(), key=lambda x: -x[1])[:15]},
           "n_distinct_pairs": len(pairs)}
    (BENCH / "audit_14_quantity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
