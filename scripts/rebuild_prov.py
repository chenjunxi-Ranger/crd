"""Regenerate provenance JSONs from existing artifacts (no retraining).

All provenance fields are read from the artifacts (mu/sd/lam from the head
checkpoint, hashes from the cached tensors and the manifest files), so this is
a faithful re-serialisation, not a recomputation of results.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "fgbench"
BENCH = ROOT / "fgbench" / "data" / "fgbench_v1_strict"
DATA_SEED = 0
CFG = "clip_VG_Attribute"
ARMS = ["image_row", "text_row", "both_row", "both_cycle", "both_shuffled"]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sha_arr(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def sha_text(values) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def main() -> int:
    tr = np.load(OUT / f"crdbb_clip_VG_Attribute_c8000_train.npz", allow_pickle=True)
    tr_meta = pd.read_parquet(OUT / f"crdbb_clip_VG_Attribute_c8000_train"
                                    f".meta.parquet")
    va = np.load(OUT / "crdbb_clip_VG_Attribute_val.npz", allow_pickle=True)
    te = np.load(OUT / "crdbb_clip_VG_Attribute_test.npz", allow_pickle=True)
    man = {s: sha_file(BENCH / f"{s}.parquet") for s in ("train", "val", "test")}

    for arm in ARMS:
        parts, target = arm.split("_", 1)
        for seed in (0, 1, 2):
            head = torch.load(OUT / f"crdbb_{CFG}_{arm}_s{seed}_head.pt",
                              map_location="cpu")
            sc = np.load(OUT / f"crdbb_scores_{CFG}_{arm}_s{seed}.npz",
                         allow_pickle=True)
            prov = {
                "backbone": "clip", "dataset": "VG", "facet": "Attribute",
                "parts": parts, "target": target, "seed": seed,
                "data_seed": DATA_SEED,
                "manifest_sha": man,
                "cache_sha": {"U": sha_arr(tr["U"]), "V": sha_arr(tr["V"]),
                              "S_train": sha_arr(tr["S"]), "S_test": sha_arr(te["S"])},
                "train_order_sha": sha_text(tr_meta.cycle_id.astype(str).values),
                "n_train_rows": int(len(tr_meta)), "n_val_rows": int(len(va["U"])),
                "n_test_rows": int(len(te["U"])),
                "mu": float(head["mu"]), "sd": float(head["sd"]),
                "lam": float(sc["lam"][0]),
                "steps": 1500, "batch": 256, "lr": 1e-3,
                "note": "provenance re-serialised from artifacts; values not recomputed",
            }
            p = OUT / f"crdbb_prov_{CFG}_{arm}_s{seed}.json"
            p.write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
            print("wrote", p.name, "lam", prov["lam"], "mu", round(prov["mu"], 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
