"""Download Visual Genome images (2 archives) with resume + progress."""
from __future__ import annotations

import argparse
import shutil
import socket
import time
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/BoyangZ/VisualGenome_VG_100K_1_and_2/resolve/main/"
FILES = {"images.zip": 9.73e9, "images2.zip": 5.47e9}


def human(n: float) -> str:
    return f"{n/1e9:.2f} GB"


def fetch(url: str, dst: Path, expect: float) -> None:
    if dst.is_file() and dst.stat().st_size >= expect * 0.995:
        print(f"skip {dst.name} ({human(dst.stat().st_size)})", flush=True)
        return
    have = dst.stat().st_size if dst.is_file() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"resume {dst.name} at {human(have)}", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(req) as r, open(dst, "ab" if have else "wb") as w:
        done = have
        last = 0.0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            w.write(b)
            done += len(b)
            if done - last > 5e8:
                last = done
                sp = done / max(1e-9, time.time() - t0)
                print(f"  {dst.name} {human(done)} / {human(expect)}  "
                      f"{sp/1e6:.1f} MB/s", flush=True)
    print(f"done {dst.name} ({human(dst.stat().st_size)})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    print(f"free disk at target: {human(free)}", flush=True)
    if free < 20e9:
        print("WARNING: less than 20 GB free", flush=True)
    socket.setdefaulttimeout(60)
    for f, exp in FILES.items():
        dst = out / f
        for attempt in range(40):                  # retry until the size matches
            try:
                fetch(BASE + f, dst, exp)
            except Exception as e:                 # noqa: BLE001
                print(f"  retry {f} ({attempt+1}): {type(e).__name__} {str(e)[:60]}",
                      flush=True)
                time.sleep(5)
            got = dst.stat().st_size if dst.is_file() else 0
            if got >= exp * 0.995:
                break
            print(f"  short {f}: {human(got)} / {human(exp)} -> resume", flush=True)
            time.sleep(3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
