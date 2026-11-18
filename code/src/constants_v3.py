"""Frozen constants for v3.0 (C3 revised: one CLIP backbone + LoRA, C2 training).

Frozen on 2026-11-04 after the LoRA Stage 1/2/3 sweep.  Selection rule:
maximise the number of columns above threshold, then the minimum margin.
Nothing here depends on a dataset id or on the protocol.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

VERSION = "v3.0.0"


@dataclass(frozen=True)
class ScoreConstantsV3:
    version: str = VERSION
    lambda_bank: float = 0.8      # A = -cos + lambda * (r_I + r_c)
    top_k: int = 10
    lora_blocks: tuple = (10, 11)
    lora_rank: int = 4
    lora_alpha: float = 8.0
    rank_margin: float = 0.1
    anchor_weight: float = 5.0
    nce_weight: float = 1.0
    train_steps: int = 300
    train_lr: float = 1e-4


CONSTANTS = ScoreConstantsV3()


def constants_dict() -> dict:
    d = asdict(CONSTANTS)
    d["lora_blocks"] = list(d["lora_blocks"])
    return d


def constants_hash() -> str:
    payload = json.dumps(constants_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


if __name__ == "__main__":
    print(json.dumps(constants_dict(), ensure_ascii=False, indent=2))
    print("hash:", constants_hash())
