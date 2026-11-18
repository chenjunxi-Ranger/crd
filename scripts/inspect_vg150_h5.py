"""Inspect the VG150 HDF5 file: what keys does it actually contain?"""
from __future__ import annotations

import sys
from pathlib import Path

P = Path(r"D:\迅雷下载\VG-SGG-with-attri.h5")


def main() -> int:
    if not P.is_file():
        print("NOT FOUND:", P)
        return 1
    print(f"size: {P.stat().st_size/1e9:.2f} GB")
    try:
        import h5py
    except ImportError:
        print("h5py missing")
        return 1
    with h5py.File(P, "r") as f:
        print("keys:", list(f.keys()))
        for k in f.keys():
            d = f[k]
            if isinstance(d, h5py.Dataset):
                print(f"  {k:24s} shape={d.shape} dtype={d.dtype}")
            else:
                print(f"  {k:24s} <group> keys={list(d.keys())[:5]}")
        print("attrs:", dict(f.attrs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
