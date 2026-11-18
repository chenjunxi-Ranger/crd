"""Frozen constants for v2.0 (C4 and C7 lifted; C1/C2/C3 still active).

Frozen on 2026-11-04 from ``scripts/gate_final_grid.py`` (one global set chosen by
maximising the minimum margin to the four thresholds).  Nothing here may depend
on a dataset, split, or protocol.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

VERSION = "v2.0.0"


@dataclass(frozen=True)
class ScoreConstantsV2:
    version: str = VERSION
    alpha_bank: float = 6.0     # weight of ((rx+rc)/2 - cos), i.e. CSLS-style term
    beta_lm: float = 0.05       # weight of (-mean GPT-2 token log-prob)
    top_k: int = 10             # neighbours used for rx / rc
    text_max_length: int = 77
    image_batch: int = 64


CONSTANTS = ScoreConstantsV2()


def constants_dict() -> dict:
    return asdict(CONSTANTS)


def constants_hash() -> str:
    payload = json.dumps(constants_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
