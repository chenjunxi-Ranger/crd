"""Find the correct filenames for the VG 1.2 attribute / image_data archives."""
from __future__ import annotations

import socket
import urllib.request

BASE = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/"
CANDIDATES = [
    "attributes.json.zip",
    "attributes_v1_2.json.zip",
    "attribute.json.zip",
    "attributes_v1.2.json.zip",
    "image_data.json.zip",
    "image_data_v1_2.json.zip",
    "image_data_v1.2.json.zip",
    "objects.json.zip",
    "relationships.json.zip",
    "region_descriptions.json.zip",
]


def main() -> int:
    socket.setdefaulttimeout(20)
    for f in CANDIDATES:
        try:
            req = urllib.request.Request(BASE + f, method="HEAD")
            with urllib.request.urlopen(req) as r:
                n = int(r.headers.get("Content-Length") or 0)
                print(f"{f:32s} OK   {n/1e6:8.1f} MB")
        except Exception as e:                      # noqa: BLE001
            print(f"{f:32s} --   {type(e).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
