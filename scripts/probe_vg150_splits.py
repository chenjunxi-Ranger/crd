"""Probe small official VG150 split json files."""
from __future__ import annotations

import socket
import urllib.request

URLS = []
for br in ("master", "main"):
    for f in ("train.json", "val.json", "test.json", "train_idx.json",
              "vg150_train.json", "VG-SGG-dicts-with-attri.json",
              "image_data.json"):
        URLS.append("https://raw.githubusercontent.com/KaihuaTang/"
                    f"Scene-Graph-Benchmark.pytorch/{br}/data/vg150/{f}")

URLS += [
    "https://huggingface.co/datasets/JosephZ/vg150_train_sgg_prompt/resolve/main/README.md",
]


def main() -> int:
    socket.setdefaulttimeout(15)
    ok = 0
    for u in URLS:
        try:
            req = urllib.request.Request(u, method="HEAD",
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r:
                n = int(r.headers.get("Content-Length") or 0)
                print(f"OK {n/1e6:8.2f} MB  {u}")
                ok += 1
        except Exception:                          # noqa: BLE001
            pass
    print("total ok:", ok, "of", len(URLS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
