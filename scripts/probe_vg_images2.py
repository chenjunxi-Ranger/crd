"""Second round of probes for VG image archives."""
from __future__ import annotations

import socket
import urllib.request

CAND = [
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
    "http://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/VG_100K.tar.gz",
    "https://huggingface.co/datasets/ranjaykrishna/visual_genome/resolve/main/README.md",
]


def main() -> int:
    socket.setdefaulttimeout(25)
    for u in CAND:
        try:
            req = urllib.request.Request(u, method="HEAD",
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r:
                n = int(r.headers.get("Content-Length") or 0)
                print(f"OK   {n/1e9:8.2f} GB  {u}")
        except Exception as e:                     # noqa: BLE001
            print(f"--   {type(e).__name__:18s} {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
