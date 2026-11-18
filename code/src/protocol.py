"""Data contract, paths, and statistical helpers.

Everything in this module is label-agnostic except :func:`bootstrap_auc` and
:func:`bootstrap_acc`, which are *evaluation* helpers and are only imported by
``scripts/`` (never by ``src/score.py``).
"""
from __future__ import annotations

import os

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "data"
RESULTS_DIR = PROJECT / "results"
CLIP_DIR = Path(os.environ.get("CLIP_DIR", PROJECT / "weights" / "clip-vit-base-patch32"))

DERIVED = PROJECT / "data" / "derived"
BENCH_CSV = {
    "coco": DERIVED / "mscoco" / "mscoco__cat__lvl0.40__seed0.csv",
    "flickr": DERIVED / "flickr30k" / "flickr30k__noun__lvl0.40__seed0.csv",
    "mmimdb": DERIVED / "mmimdb" / "mmimdb__cat__lvl0.40__seed0.csv",
}
FOIL_VAL = DATA_DIR / "foil_val_10k.csv"
FOIL_TEST = PROJECT / "data" / "FOIL-COCO" / "pairs_test.csv"
FOIL_MANIFEST = DATA_DIR / "foil_val_manifest.json"

THRESHOLDS = {"coco": 0.960, "flickr": 0.960, "mmimdb": 0.880, "foil": 0.920}


@dataclass(frozen=True)
class Row:
    image_id: str
    path: str
    caption: str


def load_split(bench: str, split: str, limit: int | None = None) -> pd.DataFrame:
    """Load one benchmark split as ``image_id / path / caption / label``."""
    if bench not in BENCH_CSV:
        raise KeyError(f"unknown bench {bench!r}; expected one of {sorted(BENCH_CSV)}")
    df = pd.read_csv(
        BENCH_CSV[bench], encoding="utf-8-sig",
        usecols=["split", "image_id", "path", "sentence", "is_mislabel"],
    )
    sub = df[df["split"] == split].copy()
    if sub.empty:
        raise ValueError(f"{bench}/{split} is empty")
    if sub["image_id"].duplicated().any():
        raise ValueError(f"{bench}/{split}: duplicated image_id")
    out = pd.DataFrame({
        "image_id": sub["image_id"].astype(str).to_numpy(),
        "path": sub["path"].astype(str).to_numpy(),
        "caption": sub["sentence"].astype(str).to_numpy(),
        "label": sub["is_mislabel"].astype(int).to_numpy(),
    })
    if limit is not None:
        out = out.head(int(limit)).reset_index(drop=True)
    return out


def load_foil_val(limit: int | None = None) -> pd.DataFrame:
    if not FOIL_VAL.is_file():
        raise FileNotFoundError(
            f"{FOIL_VAL} missing; run scripts/build_foil_split.py first")
    df = pd.read_csv(FOIL_VAL, encoding="utf-8-sig")
    out = pd.DataFrame({
        "pair_id": df["pair_id"].astype(str),
        "image_id": df["image_id"].astype(str),
        "path": df["path"].astype(str),
        "positive": df["positive_caption"].astype(str),
        "negative": df["negative_caption"].astype(str),
    })
    if limit is not None:
        out = out.head(int(limit)).reset_index(drop=True)
    return out


def load_foil_test(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(FOIL_TEST, encoding="utf-8-sig")
    out = pd.DataFrame({
        "pair_id": df["pair_id"].astype(str),
        "image_id": df["image_id"].astype(str),
        "path": df["path"].astype(str),
        "positive": df["positive_caption"].astype(str),
        "negative": df["negative_caption"].astype(str),
    })
    if limit is not None:
        out = out.head(int(limit)).reset_index(drop=True)
    return out


def bootstrap_auc(y: np.ndarray, s: np.ndarray, n_boot: int = 1000,
                  seed: int = 0) -> tuple[float, float, float]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    point = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return point, lo, hi


def bootstrap_acc(correct: np.ndarray, n_boot: int = 1000,
                  seed: int = 0) -> tuple[float, float, float]:
    c = np.asarray(correct, dtype=np.float64)
    point = float(c.mean())
    rng = np.random.default_rng(seed)
    vals = [float(c[rng.integers(0, len(c), len(c))].mean()) for _ in range(n_boot)]
    lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return point, lo, hi


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
