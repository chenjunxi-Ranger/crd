"""Probe reachable sources for the Visual Genome image archives."""
from __future__ import annotations

import socket
import urllib.request

CAND = [
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/images.zip",
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/images_part2.zip",
    "https://visualgenome.org/static/data/dataset/images.zip",
    "https://visualgenome.org/static/data/dataset/images_part2.zip",
    "https://hf-mirror.com/datasets/JosephZ/vg150_train_sgg_prompt/resolve/main/README.md",
]


def main() -> int:
    socket.setdefaulttimeout(25)
    for u in CAND:
        try:
            req = urllib.request.Request(u, method="HEAD")
            with urllib.request.urlopen(req) as r:
                n = int(r.headers.get("Content-Length") or 0)
                print(f"OK   {n/1e9:8.2f} GB  {u}")
        except Exception as e:                     # noqa: BLE001
            print(f"--   {type(e).__name__:18s} {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
