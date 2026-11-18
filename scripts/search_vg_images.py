"""Search HuggingFace for datasets that actually host Visual Genome images."""
from __future__ import annotations

import json
import socket
import urllib.parse
import urllib.request

QUERIES = ["visual genome", "vg100k", "vg_100k", "visualgenome images",
           "VG relation", "visual genome scene graph"]


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req))


def main() -> int:
    socket.setdefaulttimeout(25)
    seen = {}
    for q in QUERIES:
        try:
            d = get("https://huggingface.co/api/datasets?search="
                    + urllib.parse.quote(q) + "&limit=20&full=false")
        except Exception as e:                     # noqa: BLE001
            print("search fail", q, type(e).__name__)
            continue
        for x in d:
            seen.setdefault(x["id"], 0)
            seen[x["id"]] += 1

    print(f"candidates: {len(seen)}")
    for rid, n in sorted(seen.items(), key=lambda kv: -kv[1])[:25]:
        try:
            info = get("https://huggingface.co/api/datasets/" + rid)
            fs = [s["rfilename"] for s in info.get("siblings", [])]
            has_img = any(f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                          for f in fs)
            has_parquet = any(f.endswith(".parquet") for f in fs)
            tag = "IMG" if has_img else ("PARQUET" if has_parquet else "   ")
            print(f"{tag:8s} {rid:55s} files={len(fs)}")
            if has_img:
                print("        ", [f for f in fs if f.lower().endswith((".jpg", ".png"))][:3])
        except Exception:                          # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
