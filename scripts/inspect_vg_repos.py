"""Inspect candidate repos for a compact VG image archive."""
from __future__ import annotations

import json
import socket
import urllib.request

REPOS = ["lsnature/VG_100K", "BoyangZ/VisualGenome_VG_100K_1_and_2",
         "jn12/VG100K4CL", "eternalBlissard/VisualGenomeTrainingSet",
         "xmu-xiaoma666-dataset/detection_VisualGenome"]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req))


def main() -> int:
    socket.setdefaulttimeout(30)
    for rid in REPOS:
        try:
            info = get(f"https://huggingface.co/api/datasets/{rid}")
            fs = [s["rfilename"] for s in info.get("siblings", [])]
            print(f"== {rid}  ({len(fs)} files)")
            for f in fs[:8]:
                try:
                    t = get(f"https://huggingface.co/api/datasets/{rid}/tree/main/"
                            + urllib.request.quote(f))
                    sz = sum(x.get("size", 0) for x in t) if isinstance(t, list) else t.get("size", 0)
                except Exception:                  # noqa: BLE001
                    sz = 0
                print(f"   {sz/1e9:7.3f} GB  {f}")
        except Exception as e:                     # noqa: BLE001
            print(rid, "FAIL", type(e).__name__, str(e)[:60])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
