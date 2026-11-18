"""Download the What'sUp repository archive."""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

DST = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\WhatsUp")


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    url = "https://codeload.github.com/amitakamath/whatsup_vlms/zip/refs/heads/main"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    print("zip", round(len(blob) / 1e6, 2), "MB", flush=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(DST)
        names = z.namelist()
    print("entries", len(names), flush=True)
    for n in names[:25]:
        print("  ", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
