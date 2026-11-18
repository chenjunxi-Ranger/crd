"""The single, global constant set (C1).

Frozen on 2026-11-03 from ``scripts/freeze_weights.py``: the grid showed that any
witness weight lifting FOIL above 0.92 drops COCO to ~0.84, so the frozen
compromise is ``lambda_witness = 0`` (documented in the paper).  Nothing in this
file is allowed to depend on a dataset, a split, or a protocol.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

VERSION = "v1.0.0-rc1"


@dataclass(frozen=True)
class ScoreConstants:
    version: str = VERSION
    tau_support: float = 0.20        # cosine support threshold (soft coverage)
    temp_coverage: float = 0.05      # soft-coverage temperature
    lambda_coverage: float = 0.50    # weight of (1 - coverage)
    lambda_witness: float = 0.00     # dataset-internal witness is disabled (see METHOD)
    text_max_length: int = 77
    image_batch: int = 64


CONSTANTS = ScoreConstants()


def constants_dict() -> dict:
    return asdict(CONSTANTS)


def constants_hash() -> str:
    payload = json.dumps(constants_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
