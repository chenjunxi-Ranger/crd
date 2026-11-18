"""Probe public locations of the VG150 split id lists."""
from __future__ import annotations

import socket
import urllib.request

CAND = [
    "https://raw.githubusercontent.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/master/data/vg150/image_data.json",
    "https://raw.githubusercontent.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/master/data/vg150/train.json",
    "https://raw.githubusercontent.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/master/data/vg150/val.json",
    "https://raw.githubusercontent.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/master/data/vg150/test.json",
    "https://raw.githubusercontent.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/master/data/vg150/VG-SGG-dicts-with-attri.json",
    "https://huggingface.co/datasets/liminghao1630/vg150/resolve/main/train.json",
    "https://huggingface.co/datasets/liminghao1630/vg150/resolve/main/val.json",
]


def main() -> int:
    socket.setdefaulttimeout(20)
    for u in CAND:
        try:
            req = urllib.request.Request(u, method="HEAD")
            with urllib.request.urlopen(req) as r:
                n = int(r.headers.get("Content-Length") or 0)
                print(f"OK   {n/1e6:7.2f} MB  {u}")
        except Exception as e:                    # noqa: BLE001
            print(f"--   {type(e).__name__:16s} {u[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
