"""Download SigLIP 2 base/16-224 by direct HTTPS (snapshot_download hits Xet and fails)."""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

MID = "google/siglip2-base-patch16-224"
DST = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\siglip2-base-patch16-224")
FILES = ["config.json", "preprocessor_config.json", "tokenizer.json",
         "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
         "model.safetensors"]


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = DST / f
        url = f"https://huggingface.co/{MID}/resolve/main/{f}"
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    total = int(r.headers.get("Content-Length", 0))
                    tmp = dst.with_suffix(dst.suffix + ".part")
                    got = 0
                    t0 = time.time()
                    with open(tmp, "wb") as out:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
                            got += len(chunk)
                    tmp.replace(dst)
                    print(f"{f}: {got/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
                break
            except Exception as e:                      # noqa: BLE001
                print(f"retry {f} ({attempt+1}): {type(e).__name__} {str(e)[:70]}", flush=True)
                time.sleep(3)
    print("DONE", DST, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
