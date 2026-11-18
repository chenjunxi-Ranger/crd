"""Formal single-annotator batch: fresh draw, all groups mixed and shuffled.

- excludes every sample used by the pilot sheet (so the two rounds are disjoint);
- when a stratum is exhausted, reuses pilot samples and flags them (`reused=1`);
- assigns opaque ids AN-0001.. and writes a private key in a separate folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, load_split, write_json   # noqa: E402

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
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--out", default="formal")
    args = ap.parse_args()

    pilot_key = read_any(RESULTS_DIR / "audit" / "blind" / "coco_key.csv")
    pilot_ids = {(r["bench"], int(r["sample_id"])) for _, r in pilot_key.iterrows()}

    rows = []
    for bench in ("coco", "flickr"):
        full = read_any(RESULTS_DIR / "audit" / f"{bench}_all_groups.csv")
        df = load_split(bench, "val")
        for g in GROUPS:
            pool = full[full["group"] == g].copy()
            fresh = pool[[(bench, int(s)) not in pilot_ids
                          for s in pool["sample_id"]]]
            used = fresh if len(fresh) >= min(args.per_group, len(pool)) \
                else pool
            take = used.sample(min(args.per_group, len(used)), random_state=11)
            for _, r in take.iterrows():
                sid = int(r["sample_id"])
                rec = df.iloc[sid]
                rows.append({
                    "bench": bench, "sample_id": sid, "image_id": r["image_id"],
                    "label": int(r["label"]), "group": g,
                    "n_hard": int(r["n_hard"]),
                    "mean_b_cos": float(r["mean_b_cos"]),
                    "mean_b_A": float(r["mean_b_A"]),
                    "reused": int((bench, sid) in pilot_ids),
                    "image": rec["path"], "caption": rec["caption"],
                })
    out = pd.DataFrame(rows).sample(frac=1.0, random_state=23).reset_index(drop=True)
    out["anon_id"] = [f"AN-{i:04d}" for i in range(len(out))]

    d = RESULTS_DIR / "audit" / args.out
    d.mkdir(parents=True, exist_ok=True)
    blind = out[["anon_id", "image", "caption"]].copy()
    blind["match"] = ""
    blind["err_type"] = ""
    blind.to_csv(d / "batch01_blind.csv", index=False, encoding="utf-8-sig")
    out.drop(columns=["image", "caption"]).to_csv(d / "batch01_key.csv", index=False,
                                                  encoding="utf-8-sig")
    stats = {"n": int(len(out)), "reused": int(out["reused"].sum()),
             "by_bench_group": out.groupby(["bench", "group"]).size().to_dict()}
    write_json(d / "batch01_meta.json", {k: (v if not isinstance(v, dict)
                                             else {f"{a}|{b}": c for (a, b), c in v.items()})
                                         for k, v in stats.items()})
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
