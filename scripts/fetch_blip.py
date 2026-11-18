"""Download Salesforce/blip-itm-base-coco (PyTorch weights only, no TF)."""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

MID = "Salesforce/blip-itm-base-coco"
DST = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\blip-itm-base-coco")
FILES = ["config.json", "preprocessor_config.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt", "pytorch_model.bin"]


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = DST / f
        url = f"https://huggingface.co/{MID}/resolve/main/{f}"
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    total = int(r.headers.get("Content-Length", 0))
                    tmp = dst.with_suffix(dst.suffix + ".part")
                    got, t0 = 0, time.time()
                    with open(tmp, "wb") as out:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
                            got += len(chunk)
                    if total and got < total:
                        raise IOError(f"short {got}/{total}")
                    tmp.replace(dst)
                    print(f"{f}: {got/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"{f}: 404 skip", flush=True)
                    break
                print(f"retry {f} ({attempt+1}): {e.code}", flush=True)
                time.sleep(3)
            except Exception as e:                     # noqa: BLE001
                print(f"retry {f} ({attempt+1}): {type(e).__name__} {str(e)[:60]}", flush=True)
                time.sleep(3)
    print("DONE", DST, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
