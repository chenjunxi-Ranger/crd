"""Resume the truncated SeeTRUE images.zip with Range requests."""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

URL = "https://huggingface.co/datasets/yonatanbitton/SeeTRUE/resolve/main/images.zip"
DST = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\SeeTRUE\images.zip")
TOTAL = 5_400_400_848


def main() -> int:
    for attempt in range(60):
        have = DST.stat().st_size if DST.is_file() else 0
        if have >= TOTAL:
            break
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0",
                                                   "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r, open(DST, "ab") as out:
                t0, mark = time.time(), have
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                now = DST.stat().st_size
                print(f"attempt {attempt+1}: {have/1e6:.0f} -> {now/1e6:.0f} MB "
                      f"({(now-mark)/1e6:.0f} MB in {time.time()-t0:.0f}s)", flush=True)
        except Exception as e:                     # noqa: BLE001
            print(f"retry {attempt+1}: {type(e).__name__} {str(e)[:60]}", flush=True)
            time.sleep(3)
    print(f"FINAL {DST.stat().st_size} / {TOTAL} full={DST.stat().st_size >= TOTAL}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
