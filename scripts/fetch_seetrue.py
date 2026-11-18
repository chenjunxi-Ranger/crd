"""Download SeeTRUE (csv + images.zip) from HuggingFace."""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

MID = "yonatanbitton/SeeTRUE"
DST = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\SeeTRUE")


def get(f: str) -> None:
    dst = DST / f
    url = f"https://huggingface.co/datasets/{MID}/resolve/main/{f}"
    if dst.is_file() and dst.stat().st_size > 0 and f != "images.zip":
        print(f"{f} exists", flush=True)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
        print(f"{f}: {total/1e6:.1f} MB", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        got, t0 = 0, time.time()
        with open(tmp, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if total and got % (50 << 20) < (1 << 20):
                    print(f"  {got/1e6:.0f}/{total/1e6:.0f} MB "
                          f"({got/max(1,time.time()-t0)/1e6:.2f} MB/s)", flush=True)
        tmp.replace(dst)
        print(f"{f}: done {got/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    for f in ("SeeTRUE.csv", "SeeTRUE.py", "images.zip"):
        try:
            get(f)
        except Exception as e:                     # noqa: BLE001
            print(f"{f} FAILED {type(e).__name__} {str(e)[:80]}", flush=True)
    print("DONE", DST, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
