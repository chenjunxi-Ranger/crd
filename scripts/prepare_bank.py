"""Resumable reference-bank builder (official train only, no labels).

Stages
------
``images``   encode missing images in bounded chunks (repeat until done)
``texts``    encode missing captions in one pass
``assemble`` build the row-aligned (image, caption) arrays used by the Gate
``status``   print progress

Bank definition
---------------
* coco / flickr / mmimdb : the official ``train`` split, one caption per row
* foil                   : FOIL ``pairs_train.csv`` minus every image used by
                           the frozen validation split (image-disjoint, C6)

Nothing here reads test data or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                       # noqa: E402
from src.protocol import (CLIP_DIR, DATA_DIR, PROJECT, RESULTS_DIR,  # noqa: E402
                          load_foil_val, load_split)

BANK_DIR = RESULTS_DIR / "gate" / "bank"


def _rows(bench: str, limit: int) -> pd.DataFrame:
    if bench == "foil":
        df = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                         encoding="utf-8-sig")
        val = load_foil_val()
        val_names = set(val["path"].str.replace("\\", "/").str.split("/").str[-1])
        names = df["path"].str.replace("\\", "/").str.split("/").str[-1]
        df = df[~names.isin(val_names)].reset_index(drop=True)
        out = pd.DataFrame({
            "key": df["pair_id"].astype(str),
            "image_key": df["path"].astype(str),
            "path": df["path"].astype(str),
            "caption": df["positive_caption"].astype(str),
        })
    else:
        df = load_split(bench, "train")
        out = pd.DataFrame({
            "key": df["image_id"].astype(str),
            "image_key": df["image_id"].astype(str),
            "path": df["path"].astype(str),
            "caption": df["caption"].astype(str),
        })
    if limit:
        out = out.head(int(limit))
    return out


def _dir(bench: str, limit: int) -> Path:
    d = BANK_DIR / f"{bench}_{limit or 'all'}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_chunks(prefix: Path, keys_wanted: list[str]) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for f in sorted(prefix.parent.glob(prefix.name + "_*.npz")):
        with np.load(f, allow_pickle=False) as z:
            for k, v in zip(z["keys"].astype(str).tolist(), z["emb"]):
                table[k] = v.astype(np.float32)
    if not keys_wanted:
        return table
    return {k: table[k] for k in keys_wanted if k in table}


def _save_chunk(path: Path, keys: list[str], emb: np.ndarray) -> None:
    np.savez_compressed(path, keys=np.asarray(keys, dtype=str), emb=emb.astype(np.float32))


def cmd_status(args) -> int:
    rows = _rows(args.bench, args.limit)
    d = _dir(args.bench, args.limit)
    imgs = _load_chunks(d / "img", sorted(rows["image_key"].unique().tolist()))
    txts = _load_chunks(d / "txt", sorted({_tk(c) for c in rows["caption"]}))
    print(json.dumps({
        "bench": args.bench, "rows": int(len(rows)),
        "unique_images": int(rows["image_key"].nunique()),
        "unique_captions": int(rows["caption"].nunique()),
        "images_done": len(imgs), "texts_done": len(txts),
        "dir": str(d),
    }, ensure_ascii=False, indent=2))
    return 0


def _tk(caption: str) -> str:
    return hashlib.md5(caption.strip().encode("utf-8")).hexdigest()


def cmd_images(args) -> int:
    rows = _rows(args.bench, args.limit)
    d = _dir(args.bench, args.limit)
    uniq = rows.drop_duplicates("image_key")[["image_key", "path"]]
    done = _load_chunks(d / "img", [])
    todo = uniq[~uniq["image_key"].isin(done)].head(args.chunk)
    if todo.empty:
        print("images complete")
        return 0
    enc = JointEncoder(CLIP_DIR)
    emb = enc.encode_images(todo["path"].tolist(), batch_size=args.batch, num_workers=4)
    idx = len(list(d.glob("img_*.npz")))
    _save_chunk(d / f"img_{idx:03d}.npz", todo["image_key"].tolist(), emb)
    print(json.dumps({"encoded": int(len(todo)), "remaining": int(len(uniq) - len(done) - len(todo))}))
    return 0


def cmd_texts(args) -> int:
    rows = _rows(args.bench, args.limit)
    d = _dir(args.bench, args.limit)
    uniq = rows.drop_duplicates("key")[["key", "caption"]]
    done = _load_chunks(d / "txt", [])
    todo = uniq[~uniq["key"].isin(done)]
    if todo.empty:
        print("texts complete")
        return 0
    enc = JointEncoder(CLIP_DIR)
    emb = enc.encode_texts(todo["caption"].tolist(), batch_size=256)
    idx = len(list(d.glob("txt_*.npz")))
    _save_chunk(d / f"txt_{idx:03d}.npz", todo["key"].tolist(), emb)
    print(json.dumps({"encoded": int(len(todo)), "remaining": 0}))
    return 0


def cmd_assemble(args) -> int:
    rows = _rows(args.bench, args.limit)
    d = _dir(args.bench, args.limit)
    img = _load_chunks(d / "img", sorted(rows["image_key"].unique().tolist()))
    txt = _load_chunks(d / "txt", sorted(rows["key"].unique().tolist()))
    if len(img) < rows["image_key"].nunique() or len(txt) < rows["key"].nunique():
        raise SystemExit(f"incomplete: images {len(img)}/{rows['image_key'].nunique()}, "
                         f"texts {len(txt)}/{rows['key'].nunique()}")
    bank_img = np.stack([img[k] for k in rows["image_key"]])
    bank_txt = np.stack([txt[k] for k in rows["key"]])
    out = RESULTS_DIR / "gate" / f"bank_{args.bench}_{args.limit or 'all'}.npz"
    np.savez_compressed(out, img=bank_img, txt=bank_txt,
                        image_key=rows["image_key"].to_numpy(dtype=str))
    manifest = {"bench": args.bench, "rows": int(len(rows)),
                "unique_images": int(rows["image_key"].nunique()),
                "file": str(out), "bytes": int(out.stat().st_size)}
    (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--stage", required=True, choices=["status", "images", "texts", "assemble"])
    ap.add_argument("--limit", type=int, default=0, help="0 = every allowed train row")
    ap.add_argument("--chunk", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    return {"status": cmd_status, "images": cmd_images,
            "texts": cmd_texts, "assemble": cmd_assemble}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
