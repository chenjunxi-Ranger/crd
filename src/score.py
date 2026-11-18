"""The only scoring entry point: ``score_pairs(paths, captions) -> ndarray``.

s(x, c) = -cos(x, c) + lambda_coverage * (1 - coverage(x, c))

where ``coverage`` is the mean over content tokens of
``sigmoid((max_p cos(token, patch) - tau) / temp)``.

Invariants (CONSTRAINTS C1-C3):
* no dataset/protocol/split argument exists anywhere in this module;
* no label is read;
* no parameter is updated;
* only the frozen CLIP ViT-B/32 checkpoint is loaded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import CONSTANTS
from .encoders import JointEncoder


class FrozenScorer:
    def __init__(self, model_dir: Path | str | None = None, device: str = "auto") -> None:
        from .protocol import CLIP_DIR
        self.encoder = JointEncoder(model_dir or CLIP_DIR, device)
        self.tau = CONSTANTS.tau_support
        self.temp = CONSTANTS.temp_coverage
        self.lam = CONSTANTS.lambda_coverage

    def score(self, paths: Sequence[str], captions: Sequence[str]) -> np.ndarray:
        states = self.encoder.encode(
            paths, captions,
            batch_size=CONSTANTS.image_batch,
            max_length=CONSTANTS.text_max_length,
        )
        cos = (states.pooled_img * states.pooled_txt).sum(axis=1)
        coverage = self._coverage(states)
        return (-cos + self.lam * (1.0 - coverage)).astype(np.float64)

    def _coverage(self, states) -> np.ndarray:
        from .features import content_mask
        mask = content_mask(states, self.encoder.tokenizer)
        out = np.empty(len(states.token), dtype=np.float64)
        for i in range(len(states.token)):
            s = states.token[i][mask[i]] @ states.patch[i].T
            mx = s.max(axis=1)
            out[i] = float((1.0 / (1.0 + np.exp(-(mx - self.tau) / self.temp))).mean())
        return out


def score_pairs(image_paths: Sequence[str], captions: Sequence[str],
                device: str = "auto") -> np.ndarray:
    """Public, dataset-agnostic entry point. Higher = more suspicious."""
    if len(image_paths) != len(captions):
        raise ValueError("image_paths and captions must align")
    return FrozenScorer(device=device).score(image_paths, captions)
