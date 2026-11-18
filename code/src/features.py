"""Label-free features computed from the token x patch alignment matrix.

All statistics are forward-only functions of one (image, caption) pair.  No
reference bank, no labels, no fitted parameters: the constants below are fixed
a priori and are the same for every benchmark (CONSTRAINTS.md C1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .encoders import BatchStates

#: fixed a-priori constants (single global set)
TAU_SUPPORT = 0.20      # cosine support threshold for "covered"
TEMP_COVERAGE = 0.05    # soft-coverage temperature
TEMP_CYCLE = 0.07       # soft assignment temperature for mutual-NN
TOP_PATCHES = 3         # top-k patch set for the binding statistic
GRIDS = (2, 3)          # coarse grids in addition to the native 7x7

STOPWORDS = frozenset("""
a an the and or but if while of to in on at by for with from into onto over under
is are was were be been being am do does did doing have has had having
this that these those there here it its it's they them their we our you your i he she his her
as than then so such not no nor too very can could should would may might must will just
also only about above across after against along among around before behind below beneath
beside between beyond during except inside near off out outside since through throughout
toward towards upon within without who whom whose which what when where why how
s t re ve ll d m o y
""".split())


def content_mask(states: BatchStates, tokenizer) -> np.ndarray:
    """Keep the first subword of each non-stopword word; drop specials/padding."""
    mask = states.token_mask.copy()
    n, length = mask.shape
    for b in range(n):
        groups: dict[int, list[int]] = {}
        for t in range(length):
            if not mask[b, t]:
                continue
            wid = states.word_ids[b][t]
            if wid is None:
                mask[b, t] = False
                continue
            groups.setdefault(int(wid), []).append(t)
        for _, pos in groups.items():
            word = tokenizer.decode(states.input_ids[b, pos], skip_special_tokens=True)
            word = word.strip().strip(".,;:!?\"'()[]{}").lower()
            keep = bool(word) and word not in STOPWORDS and any(c.isalnum() for c in word)
            for t in pos[1:]:
                mask[b, t] = False
            if not keep:
                for t in pos:
                    mask[b, t] = False
    # guarantee at least one token per row (fall back to non-special tokens)
    empty = ~mask.any(axis=1)
    if empty.any():
        mask[empty] = states.token_mask[empty]
    return mask


def _grid_margin(token: np.ndarray, patch: np.ndarray, grid: int, k: int) -> float:
    p = torch.as_tensor(patch.T.reshape(1, -1, grid, grid), dtype=torch.float32)
    pooled = F.adaptive_avg_pool2d(p, k).reshape(-1, patch.shape[1])
    pooled = F.normalize(pooled, dim=-1).numpy()
    sim = token @ pooled.T
    mx = sim.max(axis=1)
    return float((mx - sim.mean(axis=1)).mean())


def pair_features(states: BatchStates, index: int, mask: np.ndarray, grid: int) -> dict[str, float]:
    """Features for one row (higher value = better local support)."""
    tok = states.token[index]
    pat = states.patch[index]
    m = mask[index]
    t = tok[m]
    s = t @ pat.T                                   # (n_content, P)

    mx = s.max(axis=1)
    mean_p = s.mean(axis=1)
    margin = mx - mean_p

    cov = 1.0 / (1.0 + np.exp(-(mx - TAU_SUPPORT) / TEMP_COVERAGE))

    z = s / TEMP_CYCLE
    p_row = np.exp(z - z.max(axis=1, keepdims=True))
    p_row /= p_row.sum(axis=1, keepdims=True)
    p_col = np.exp(z - z.max(axis=0, keepdims=True))
    p_col /= p_col.sum(axis=0, keepdims=True)
    pstar = s.argmax(axis=1)
    cycle = float(p_col[np.arange(s.shape[0]), pstar].mean())

    out = {
        "support_mean": float(mx.mean()),
        "margin_mean": float(margin.mean()),
        "coverage_soft": float(cov.mean()),
        "cycle_soft": cycle,
        "coverage_hard": float((mx >= TAU_SUPPORT).mean()),
        "n_content": float(s.shape[0]),
    }
    for k in GRIDS:
        out[f"grid{k}_margin"] = _grid_margin(t, pat, grid, k)

    if s.shape[0] >= 2:
        sim = t @ t.T
        np.fill_diagonal(sim, -np.inf)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        a_set = set(np.argsort(-s[i])[:TOP_PATCHES].tolist())
        b_set = set(np.argsort(-s[j])[:TOP_PATCHES].tolist())
        out["bind_disp"] = 1.0 - len(a_set & b_set) / max(1, len(a_set | b_set))
    else:
        out["bind_disp"] = 0.0
    return out


FEATURE_KEYS = [
    "support_mean", "margin_mean", "coverage_soft", "cycle_soft",
    "coverage_hard", "grid2_margin", "grid3_margin", "bind_disp", "n_content",
]


def compute(states: BatchStates, tokenizer, grid: int) -> dict[str, np.ndarray]:
    mask = content_mask(states, tokenizer)
    rows = [pair_features(states, i, mask, grid) for i in range(len(states.token))]
    return {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in FEATURE_KEYS}
