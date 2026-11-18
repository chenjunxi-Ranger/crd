"""Repair only the corrupted entries of images2.zip via HTTP Range requests."""
from __future__ import annotations

import json
import socket
import struct
import sys
import time
import urllib.request
from pathlib import Path

P = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome\images2.zip")
BAD = Path(r"D:\PyCharm\PythonProject\异常标签检测\mymodel7\results\vg_zip_bad.json")
URL = ("https://huggingface.co/datasets/BoyangZ/VisualGenome_VG_100K_1_and_2/"
       "resolve/main/images2.zip")


def fetch_range(start: int, end: int) -> bytes:
    req = urllib.request.Request(
        URL, headers={"User-Agent": "Mozilla/5.0",
                      "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def main() -> int:
    socket.setdefaulttimeout(120)
    bad = json.load(open(BAD, encoding="utf-8"))
    print("entries to repair:", len(bad), flush=True)
    t0 = time.time()
    ok = 0
    with open(P, "r+b") as f:
        for i, b in enumerate(bad):
            off = int(b["header_offset"])
            f.seek(off)
            head = f.read(30)
            if len(head) < 30:
                print("short header at", off, flush=True)
                continue
            sig, _, _, _, _, _, _, _, _, nlen, elen = struct.unpack(
                "<IHHHHHIIIHH", head)
            if sig != 0x04034B50:
                # local header itself was overwritten -> read it from the remote copy
                remote = fetch_range(off, off + 29)
                sig, _, _, _, _, _, _, _, _, nlen, elen = struct.unpack(
                    "<IHHHHHIIIHH", remote)
                if sig != 0x04034B50:
                    print("remote sig also bad at", off, flush=True)
                    continue
            total = 30 + nlen + elen + int(b["compress_size"])
            for attempt in range(4):
                try:
                    data = fetch_range(off, off + total - 1)
                    if len(data) != total:
                        raise IOError(f"got {len(data)} want {total}")
                    f.seek(off)
                    f.write(data)
                    ok += 1
                    break
                except Exception as e:             # noqa: BLE001
                    if attempt == 3:
                        print("FAIL", b["name"], type(e).__name__, str(e)[:50],
                              flush=True)
                    time.sleep(3)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(bad)} ok={ok} {time.time()-t0:.0f}s", flush=True)
    print(f"repaired {ok}/{len(bad)} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
