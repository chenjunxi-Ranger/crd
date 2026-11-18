"""C-VRR pre-experiments on the frozen FOIL validation split.

Corpus = FOIL official train minus every image in the validation split (the same
image-disjoint rule used everywhere else).  The corpus only proposes candidate
words; the frozen CLIP decides which candidate fits the image better.

Pre-experiment 1  candidate coverage (does the correct word enter the set?)
Pre-experiment 2  repair direction accuracy  R(I,c-) > R(I,c+)
Pre-experiment 3  complementarity with CLIP
Pre-experiment 4  ablations (CLIP / witness / frequency-only / adjacent context /
                  content context / shuffled corpus)

Nothing is trained; no parameter is updated.
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.features import STOPWORDS                        # noqa: E402
from src.protocol import (CLIP_DIR, PROJECT, RESULTS_DIR, load_foil_val,  # noqa: E402
                          write_json)

WORD = re.compile(r"[A-Za-z]+")
MAX_CAND = 12
ALLOWED = {"donor": None}


def words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD.finditer(text)]


def replace_word(text: str, index: int, new: str) -> str:
    spans = [m.span() for m in WORD.finditer(text)]
    s, e = spans[index]
    return text[:s] + new + text[e:]


def build_index(captions: list[str]):
    uni: Counter = Counter()
    nxt: dict[str, Counter] = defaultdict(Counter)   # N(l, u)
    pair: set[tuple[str, str]] = set()               # (u, r) support
    for cap in captions:
        toks = words(cap)
        uni.update(toks)
        for a, b in zip(toks, toks[1:]):
            nxt[a][b] += 1
            pair.add((a, b))
    return {"uni": uni, "nxt": nxt, "pair": pair}


def candidates(toks: list[str], i: int, index, mode: str, cap: int = MAX_CAND) -> list[str]:
    """Candidate words for position i under a context definition."""
    if mode == "adjacent":
        l = toks[i - 1] if i > 0 else "<s>"
        r = toks[i + 1] if i + 1 < len(toks) else "</s>"
    else:  # nearest content word on each side
        j = i - 1
        while j >= 0 and toks[j] in STOPWORDS:
            j -= 1
        l = toks[j] if j >= 0 else "<s>"
        k = i + 1
        while k < len(toks) and toks[k] in STOPWORDS:
            k += 1
        r = toks[k] if k < len(toks) else "</s>"
    cand = [(u, c) for u, c in index["nxt"].get(l, {}).items()
            if (u, r) in index["pair"] and u not in STOPWORDS]
    cand.sort(key=lambda x: -x[1])
    return [u for u, _ in cand[:cap]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--mode", default="both", choices=["adjacent", "content", "both"])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    val = load_foil_val(args.limit)
    train = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                        encoding="utf-8-sig")
    names = val["path"].str.replace("\\", "/").str.split("/").str[-1]
    tr_names = train["path"].str.replace("\\", "/").str.split("/").str[-1]
    corpus = train[~tr_names.isin(set(names))]["positive_caption"].astype(str).tolist()

    index = build_index(corpus)
    # shuffled-corpus ablation: destroy bigram structure, keep unigram frequencies
    rng = np.random.default_rng(0)
    flat = [w for cap in corpus for w in words(cap)]
    shuffled = flat.copy()
    rng.shuffle(shuffled)
    sh_caps = [" ".join(shuffled[i:i + 10]) for i in range(0, len(shuffled), 10)]
    sh_index = build_index(sh_caps)

    import torch
    enc = JointEncoder(CLIP_DIR)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def cos_scores(paths, caps):
        out = []
        for i in range(0, len(paths), args.batch):
            st = enc.encode(paths[i:i + args.batch], caps[i:i + args.batch],
                            batch_size=args.batch, num_workers=0)
            out.append((st.pooled_img * st.pooled_txt).sum(1))
        return np.concatenate(out)

    rows = []
    for r in range(len(val)):
        p, n = val["positive"].iloc[r], val["negative"].iloc[r]
        tp, tn = words(p), words(n)
        sm = difflib.SequenceMatcher(a=tp, b=tn)
        pos_idx = neg_idx = None
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("replace", "insert") and j2 > j1:
                neg_idx = j1
                pos_idx = i1 if tag == "replace" else -1
                break
        if neg_idx is None:
            continue
        rows.append({"row": r, "path": val["path"].iloc[r], "pos": p, "neg": n,
                     "pos_idx": pos_idx, "neg_idx": neg_idx})
    df = pd.DataFrame(rows)
    print(f"pairs with a local edit: {len(df)}/{len(val)}")

    modes = ["adjacent", "content"] if args.mode == "both" else [args.mode]
    result = {"n_pairs": int(len(df)), "n_val": int(len(val)), "modes": {}}

    for mode in modes:
        # candidate sets for the edited position in both captions
        sets_pos, sets_neg = [], []
        for _, row in df.iterrows():
            tp, tn = words(row["pos"]), words(row["neg"])
            sp = candidates(tp, int(row["pos_idx"]), index, mode) if row["pos_idx"] >= 0 else []
            sn = candidates(tn, row["neg_idx"], index, mode)
            sets_pos.append(sp)
            sets_neg.append(sn)

        cov_neg = float(np.mean([len(s) > 0 for s in sets_neg]))
        cov_pos_in_neg = float(np.mean([
            (words(row["pos"])[int(row["pos_idx"])] in s) if row["pos_idx"] >= 0 and s else False
            for (_, row), s in zip(df.iterrows(), sets_neg)]))
        mean_cand = float(np.mean([len(s) for s in sets_neg]))

        # build all variant captions once per side
        def side_caps(col, idx_col, sets):
            paths, caps = [], []
            for (_, row), s in zip(df.iterrows(), sets):
                paths.append(row["path"])
                caps.append(row[col])
                idx = row[idx_col]
                if idx < 0:
                    continue
                for u in s:
                    paths.append(row["path"])
                    caps.append(replace_word(row[col], int(idx), u))
            return paths, caps

        pp, pc = side_caps("pos", "pos_idx", sets_pos)
        npth, nc = side_caps("neg", "neg_idx", sets_neg)
        sc_p = cos_scores(pp, pc)
        sc_n = cos_scores(npth, nc)

        # regroup: first value per pair is the original, then the candidates
        base_p = np.zeros(len(df))
        base_n = np.zeros(len(df))
        rep_p = np.zeros(len(df))
        rep_n = np.zeros(len(df))
        kp = kn = 0
        for i in range(len(df)):
            base_p[i] = sc_p[kp]
            kp += 1
            if len(sets_pos[i]):
                rep_p[i] = max(0.0, sc_p[kp:kp + len(sets_pos[i])].max() - base_p[i])
                kp += len(sets_pos[i])
            base_n[i] = sc_n[kn]
            kn += 1
            if len(sets_neg[i]):
                rep_n[i] = max(0.0, sc_n[kn:kn + len(sets_neg[i])].max() - base_n[i])
                kn += len(sets_neg[i])

        dir_acc = float(np.mean(np.where(rep_n > rep_p, 1.0,
                                         np.where(rep_n < rep_p, 0.0, 0.5))))
        clip_acc = float(np.mean(np.where(base_n < base_p, 1.0,
                                          np.where(base_n > base_p, 0.0, 0.5))))
        pd.DataFrame({"base_pos": base_p, "base_neg": base_n,
                      "rep_pos": rep_p, "rep_neg": rep_n}).to_csv(
            RESULTS_DIR / "gate" / f"cvr_scores_{mode}.csv",
            index=False, encoding="utf-8-sig")

        # lambda sweep: does a weighted combination beat CLIP alone?
        sweep = {}
        for lam in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
            sp_ = -base_p + lam * rep_p
            sn_ = -base_n + lam * rep_n
            sweep[str(lam)] = float(np.mean(np.where(sn_ > sp_, 1.0,
                                                     np.where(sn_ < sp_, 0.0, 0.5))))

        ok_clip = base_n < base_p
        ok_rep = rep_n > rep_p
        complement = {
            "clip_right_repair_wrong": int(np.sum(ok_clip & ~ok_rep)),
            "clip_wrong_repair_right": int(np.sum(~ok_clip & ok_rep)),
            "both_right": int(np.sum(ok_clip & ok_rep)),
            "both_wrong": int(np.sum(~ok_clip & ~ok_rep)),
        }
        both_right_or = int(np.sum(ok_clip | ok_rep))
        result["modes"][mode] = {
            "coverage_neg_has_candidate": cov_neg,
            "coverage_correct_word_in_neg_set": cov_pos_in_neg,
            "mean_candidates": mean_cand,
            "clip_acc": clip_acc,
            "repair_dir_acc": dir_acc,
            "union_acc": float(both_right_or / len(df)),
            "complement": complement,
            "mean_regret_pos": float(rep_p.mean()),
            "mean_regret_neg": float(rep_n.mean()),
            "lambda_sweep": sweep,
        }
        print(mode, json.dumps(result["modes"][mode], ensure_ascii=False))

    write_json(RESULTS_DIR / "gate" / "cvr_pre.json", result)
    print(json.dumps(result["modes"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
