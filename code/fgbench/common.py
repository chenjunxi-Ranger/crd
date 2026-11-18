"""Shared utilities for FGBench manifest construction.

Only public structured annotations are used; no human labelling, no LLM rewriting,
every caption is a *natural original* caption taken from a donor image.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent

# COCO annotation dir: set COCO_ANN if your layout differs
COCO_ANN = Path(os.environ.get("COCO_ANN", PROJ / "data" / "coco" / "annotations"))
OUT = Path(os.environ.get("FGBENCH_OUT", PROJ / "data"))
INTER = OUT / "intermediate"
# v0.1 lives in data/benchmark; v0.2 in data/benchmark_v02 (switch via env)
BENCH = Path(os.environ.get("FGBENCH_BENCH", PROJ / "data" / "manifests"))

NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "single": 1, "pair": 2, "both": 2, "dozen": 12}

# surface form (lemma, lower-case) -> canonical COCO class
ALIASES = {
    "bike": "bicycle", "bikes": "bicycle", "bicycles": "bicycle", "cycle": "bicycle",
    "motorbike": "motorcycle", "motorcycles": "motorcycle",
    "aeroplane": "airplane", "airplanes": "airplane", "plane": "airplane", "planes": "airplane",
    "sofa": "couch", "couches": "couch", "settee": "couch",
    "tv": "tv", "television": "tv", "monitor": "tv", "screen": "tv",
    "cellphone": "cell phone", "phone": "cell phone", "phones": "cell phone",
    "mobile": "cell phone",
    "fridge": "refrigerator", "refrigerators": "refrigerator",
    "plant": "potted plant", "plants": "potted plant",
    "ball": "sports ball", "balls": "sports ball",
    "table": "dining table", "tables": "dining table", "desk": "dining table",
    "glass": "wine glass", "glasses": "wine glass",
    "bag": "handbag", "bags": "handbag", "purse": "handbag",
    "racket": "tennis racket", "racquet": "tennis racket",
    "donuts": "donut", "doughnut": "donut", "doughnuts": "donut",
    "skis": "skis", "ski": "skis",
    "scissors": "scissors", "teddy": "teddy bear",
    "hairdryer": "hair drier", "hair dryer": "hair drier",
    "hydrant": "fire hydrant", "meter": "parking meter",
    "sign": "stop sign", "light": "traffic light", "lights": "traffic light",
    "bat": "baseball bat", "glove": "baseball glove", "gloves": "baseball glove",
    "bear": "teddy bear",
}


def load_categories(inst: dict) -> dict[int, str]:
    return {c["id"]: c["name"].lower() for c in inst["categories"]}


def load_instances(path: Path):
    with open(path, encoding="utf-8") as f:
        ins = json.load(f)
    cat = load_categories(ins)
    counts: dict[int, Counter] = defaultdict(Counter)
    boxes: dict[int, dict] = defaultdict(lambda: defaultdict(list))
    for a in ins["annotations"]:
        if a.get("iscrowd", 0):
            continue                      # crowd regions are not instances
        counts[a["image_id"]][a["category_id"]] += 1
        boxes[a["image_id"]][a["category_id"]].append(
            [round(v, 1) for v in a["bbox"]])
    return counts, boxes, cat, ins["images"]


def load_captions(path: Path):
    with open(path, encoding="utf-8") as f:
        cap = json.load(f)
    caps: dict[int, list[str]] = defaultdict(list)
    for a in cap["annotations"]:
        caps[a["image_id"]].append(a["caption"].strip())
    return caps


def class_map_terms(cat_names: dict[int, str]):
    """surface term -> canonical class name (only COCO's 80 classes)."""
    term2cls = {}
    for name in set(cat_names.values()):
        term2cls[name] = name
        term2cls[name + "s"] = name
        if name.endswith("s"):
            term2cls[name[:-1]] = name
    for surf, canon in ALIASES.items():
        if canon in set(cat_names.values()):
            term2cls[surf] = canon
    return term2cls


def split_of(image_id: int, train=0.6, val=0.2) -> str:
    """Deterministic, image-id based split (isolation is decided BEFORE mining)."""
    h = (image_id * 2654435761) % 10000 / 10000.0
    return "train" if h < train else ("val" if h < train + val else "test")


def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
