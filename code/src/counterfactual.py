"""Text-side leave-one-word-out (LOO) attribution — no training, no corpus.

For a pair ``(x, c)`` we compare ``cos(x, c)`` with ``cos(x, c without word i)``
for every content word ``i``.  A word that the image actually supports should
*reduce* the similarity when removed (delta > 0); a word the image does not
support contributes nothing or actively hurts (delta <= 0).

This is a forward-only counterfactual: one extra text-tower pass per content
word, no image passes, no parameters, no external vocabulary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .features import STOPWORDS

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


@dataclass(frozen=True)
class WordSpan:
    word: str
    start: int
    end: int


def content_words(text: str) -> list[WordSpan]:
    spans = []
    for m in WORD_RE.finditer(text):
        w = m.group(0).strip("-'").lower()
        if not w or w in STOPWORDS:
            continue
        spans.append(WordSpan(m.group(0), m.start(), m.end()))
    return spans


def drop_word(text: str, span: WordSpan) -> str:
    out = (text[:span.start] + " " + text[span.end:]).strip()
    return re.sub(r"\s+", " ", out)


def loo_features(cos_full: float, cos_drop: np.ndarray) -> dict[str, float]:
    """Aggregate the per-word deltas of one pair (higher = better support)."""
    d = cos_full - np.asarray(cos_drop, dtype=np.float64)
    if d.size == 0:
        return {k: 0.0 for k in LOO_KEYS}
    return {
        "loo_mean": float(d.mean()),
        "loo_min": float(d.min()),
        "loo_max": float(d.max()),
        "loo_std": float(d.std()),
        "loo_neg_frac": float((d < 0).mean()),
        "loo_neg_sum": float(np.clip(-d, 0, None).sum()),
        "loo_pos_sum": float(np.clip(d, 0, None).sum()),
        "loo_top_gap": float(np.sort(d)[-1] - np.sort(d)[-2]) if d.size >= 2 else 0.0,
        "loo_n_words": float(d.size),
    }


LOO_KEYS = ["loo_mean", "loo_min", "loo_max", "loo_std", "loo_neg_frac",
            "loo_neg_sum", "loo_pos_sum", "loo_top_gap", "loo_n_words"]
