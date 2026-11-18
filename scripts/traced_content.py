"""What does the elimination trajectory actually delete?

Compares hard noisy vs easy clean on the tokens removed at each step:
proper nouns (identity-bearing) vs genre/generic words vs other content.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.protocol import RESULTS_DIR, load_split, write_json   # noqa: E402
from src.scorer import build_scorer                            # noqa: E402
from traced_repro import STOP, tokens                          # noqa: E402

GENRE = set("""love war family crime comedy thriller adventure mystery drama horror
fantasy romance murder revenge killer friend friends life world story young man woman
city police school father mother son daughter brother sister money power secret""".split())


def classify(tok, orig_tokens, pos):
    low = tok.lower()
    if low in GENRE:
        return "genre_generic"
    if pos > 0 and orig_tokens[pos][:1].isupper():
        return "proper_noun"
    return "other_content"


def trace_tokens(sc, img, cap, steps=3):
    toks = tokens(cap)
    cur = list(toks)
    s_cur = float(sc.score(img, cap))
    record = []
    for _ in range(steps):
        idx = [i for i, w in enumerate(cur) if w.lower() not in STOP and len(w) > 2]
        if not idx:
            break
        cands = [" ".join(cur[:i] + cur[i + 1:]) or cap for i in idx]
        vals = sc.score_many([img] * len(cands), cands)
        best = int(np.argmax(vals))
        gain = float(vals[best] - s_cur)
        if gain <= 1e-6:
            break
        record.append((cur[idx[best]], classify(cur[idx[best]], toks, idx[best]), gain))
        s_cur = float(vals[best])
        cur = tokens(cands[best])
    return record, s_cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="mmimdb")
    ap.add_argument("--n-hard", type=int, default=60)
    ap.add_argument("--n-clean", type=int, default=120)
    args = ap.parse_args()

    sc = build_scorer("clip")
    d = pd.read_csv(RESULTS_DIR / "gate" / f"traced_failure_{args.bench}.csv",
                    encoding="utf-8-sig")
    if "group" not in d:
        d = d.rename(columns={[c for c in d.columns if c.startswith("group")][0]: "group"})
    d["group"] = d["group"].fillna("both_low_burden")
    val = load_split(args.bench, "val", len(d))

    hard = d[(d.group == "persistent_hard") & (d.label == 1)].head(args.n_hard)
    clean = d[(d.group == "both_low_burden") & (d.label == 0)].head(args.n_clean)

    out = {}
    for name, sub in (("hard_noisy", hard), ("easy_clean", clean)):
        cnt = Counter()
        gains, n_steps, deleted = [], [], 0
        for sid in sub["sample_id"]:
            rec, _ = trace_tokens(sc, val["path"].iloc[sid], str(val["caption"].iloc[sid]))
            n_steps.append(len(rec))
            deleted += len(rec)
            for _, cls, g in rec:
                cnt[cls] += 1
                gains.append(g)
        tot = max(sum(cnt.values()), 1)
        out[name] = {"n_samples": int(len(sub)),
                     "mean_steps": float(np.mean(n_steps)) if n_steps else 0,
                     "tokens_deleted": int(deleted),
                     "share": {k: round(v / tot, 3) for k, v in cnt.items()},
                     "mean_gain": float(np.mean(gains)) if gains else 0.0}
        print(name, out[name], flush=True)
    write_json(RESULTS_DIR / "gate" / f"traced_content_{args.bench}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
