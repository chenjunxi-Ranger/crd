"""Check reachability/size of the raw annotations FGBench needs."""
from __future__ import annotations

import socket
import urllib.request

URLS = {
    "VG objects": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects_v1_2.json.zip",
    "VG attributes": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes_v1_2.json.zip",
    "VG relationships": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships_v1_2.json.zip",
    "VG image_data": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data_v1_2.json.zip",
    "F30k entities": "https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/annotations.zip",
    "F30k split": "https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/train.txt",
}


def main() -> int:
    socket.setdefaulttimeout(25)
    for name, url in URLS.items():
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as r:
                n = int(r.headers.get("Content-Length") or 0)
                print(f"{name:20s} OK   {n / 1e6:8.1f} MB")
        except Exception as e:                      # noqa: BLE001
            print(f"{name:20s} FAIL {type(e).__name__:18s} {str(e)[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
