"""Fetch What'sUp controlled images (tar.gz) from Google Drive."""
from __future__ import annotations

import os
import tarfile
from pathlib import Path

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

DATA = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\WhatsUp\whatsup_vlms-main\data")
FILES = {
    "controlled_images.tar.gz": "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H",
}


def main() -> int:
    import gdown
    DATA.mkdir(parents=True, exist_ok=True)
    for name, fid in FILES.items():
        dst = DATA / name
        if not dst.is_file() or dst.stat().st_size < 1_000_000:
            url = f"https://drive.google.com/uc?id={fid}"
            try:
                gdown.download(url, str(dst), quiet=False)
            except Exception as e:                 # noqa: BLE001
                print(f"{name} FAILED {type(e).__name__} {str(e)[:90]}", flush=True)
                continue
        print(f"{name}: {dst.stat().st_size/1e6:.1f} MB", flush=True)
        with tarfile.open(dst) as t:
            t.extractall(DATA)
        imgs = list((DATA / "controlled_images").glob("*"))
        print("extracted", len(imgs), "files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
