"""Download facebook/flava-full with resume (direct URLs, no xet)."""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

MID = "facebook/flava-full"
DST = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\flava-full")
FILES = ["config.json", "preprocessor_config.json", "special_tokens_map.json",
         "tokenizer_config.json", "vocab.txt", "merges.txt",
         "model.safetensors", "pytorch_model.bin"]


def size_of(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return int(r.headers.get("Content-Length", 0))


def fetch(f: str) -> None:
    url = f"https://huggingface.co/{MID}/resolve/main/{f}"
    dst = DST / f
    if dst.is_file() and dst.stat().st_size > 1000:
        print(f"{f} exists {dst.stat().st_size/1e6:.1f} MB", flush=True)
        return
    try:
        total = size_of(url)
    except Exception as e:                       # noqa: BLE001
        print(f"{f} HEAD failed {str(e)[:60]}", flush=True)
        return
    print(f"{f}: {total/1e6:.1f} MB", flush=True)
    for attempt in range(40):
        have = dst.stat().st_size if dst.is_file() else 0
        if have >= total:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                   "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r, open(dst, "ab") as out:
                t0, mark = time.time(), have
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            print(f"  {attempt+1}: {have/1e6:.0f} -> {dst.stat().st_size/1e6:.0f} MB "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:                   # noqa: BLE001
            print(f"  retry {attempt+1}: {type(e).__name__}", flush=True)
            time.sleep(3)
    print(f"{f}: final {dst.stat().st_size/1e6:.1f} MB (want {total/1e6:.1f})", flush=True)


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        fetch(f)
    print("DONE", DST, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
