"""Gate probe for the text-side leave-one-word-out (LOO) family.

Analysis only: labels are used to compute AUROC / accuracy, never to build a
score.  Run on a subset first (``--limit 500``) before the full Gate.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.counterfactual import LOO_KEYS, content_words, drop_word, loo_features  # noqa: E402
from src.encoders import JointEncoder                                              # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_foil_val, load_split          # noqa: E402


def _loo_for(captions: list[str], imgs: np.ndarray, enc: JointEncoder,
             batch_size: int = 256) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return LOO features per caption plus the full cosine."""
    texts: list[str] = []
    spans_per_row = []
    for c in captions:
        spans = content_words(c)
        spans_per_row.append(spans)
        texts.append(c)                      # interleaved: full row, then its variants
        texts.extend(drop_word(c, s) for s in spans)
    emb = enc.encode_texts(texts, batch_size=batch_size)
    rows, cursor = [], 0
    cos_full = np.zeros(len(captions))
    for i, spans in enumerate(spans_per_row):
        full = emb[cursor]
        cursor += 1
        drops = emb[cursor:cursor + len(spans)]
        cursor += len(spans)
        cos_full[i] = float(imgs[i] @ full)
        rows.append(loo_features(cos_full[i], drops @ imgs[i]))
    feats = {k: np.asarray([r[k] for r in rows]) for k in LOO_KEYS}
    return feats, cos_full


def _row_mode(args, enc) -> None:
    df = load_split(args.bench, "val", args.limit)
    imgs = enc.encode_images(df["path"].tolist(), batch_size=args.batch_size,
                             num_workers=args.workers)
    feats, cos = _loo_for(df["caption"].tolist(), imgs, enc)
    y = df["label"].to_numpy()
    out = pd.DataFrame(feats)
    out["cosine"] = cos
    out["label"] = y
    path = RESULTS_DIR / "gate" / f"{args.bench}_loo_features.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    rows = [{"feature": "cosine", "auc": roc_auc_score(y, -cos)}]
    for k in LOO_KEYS:
        v = feats[k]
        rows.append({"feature": k, "auc": roc_auc_score(y, -v)})
    print(pd.DataFrame(rows).sort_values("auc", ascending=False).to_string(index=False))
    print(f"saved {path}")


def _foil_mode(args, enc) -> None:
    df = load_foil_val(args.limit)
    imgs = enc.encode_images(df["path"].tolist(), batch_size=args.batch_size,
                             num_workers=args.workers)
    f_pos, cos_pos = _loo_for(df["positive"].tolist(), imgs, enc)
    f_neg, cos_neg = _loo_for(df["negative"].tolist(), imgs, enc)

    # locate the word that differs inside the negative caption (batched)
    variants: list[str] = []
    owner: list[int] = []
    for i, (p, n) in enumerate(zip(df["positive"], df["negative"])):
        pw, nw = p.split(), n.split()
        sm = difflib.SequenceMatcher(a=[w.lower() for w in pw], b=[w.lower() for w in nw])
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("replace", "insert") and j2 > j1:
                target = " ".join(nw[j1:j2]).lower()
                for s in content_words(n):
                    if s.word.lower() in target:
                        variants.append(drop_word(n, s))
                        owner.append(i)
                        break
                break
    diff_delta = np.zeros(len(df))
    if variants:
        emb = enc.encode_texts(variants)
        for idx, i in enumerate(owner):
            diff_delta[i] = cos_neg[i] - float(emb[idx] @ imgs[i])

    rows = []
    for k in LOO_KEYS:
        d = f_neg[k] - f_pos[k]
        acc = float(np.mean(np.where(d < 0, 1.0, np.where(d > 0, 0.0, 0.5))))
        rows.append({"feature": k, "acc_support_drop": acc})
    d = cos_neg - cos_pos
    rows.append({"feature": "cosine", "acc_support_drop":
                 float(np.mean(np.where(d < 0, 1.0, np.where(d > 0, 0.0, 0.5))))})
    rows.append({"feature": "diffword_loo_delta", "acc_support_drop":
                 float(np.mean(np.where(diff_delta < 0, 1.0,
                                        np.where(diff_delta > 0, 0.0, 0.5))))})
    print(pd.DataFrame(rows).sort_values("acc_support_drop", ascending=False).to_string(index=False))
    print(f"n={len(df)} mean|diffword delta|={np.abs(diff_delta).mean():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco", choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    enc = JointEncoder(CLIP_DIR)
    if args.bench == "foil":
        _foil_mode(args, enc)
    else:
        _row_mode(args, enc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
