"""Resume the truncated SigLIP 2 safetensors download."""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

URL = "https://huggingface.co/google/siglip2-base-patch16-224/resolve/main/model.safetensors"
DST = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights"
           r"\siglip2-base-patch16-224\model.safetensors")
TOTAL = 1_500_800_904


def main() -> int:
    for attempt in range(20):
        have = DST.stat().st_size if DST.is_file() else 0
        if have >= TOTAL:
            break
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0",
                                                   "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(DST, "ab") as out:
                t0 = time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                print(f"attempt {attempt+1}: {have/1e6:.0f} -> "
                      f"{DST.stat().st_size/1e6:.0f} MB ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:                      # noqa: BLE001
            print(f"retry {attempt+1}: {type(e).__name__} {str(e)[:60]}", flush=True)
            time.sleep(3)
    print(f"FINAL {DST.stat().st_size} / {TOTAL} full={DST.stat().st_size >= TOTAL}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
