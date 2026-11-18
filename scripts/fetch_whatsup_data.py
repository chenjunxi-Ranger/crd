"""Fetch What'sUp annotation JSONs (small) from Google Drive via gdown."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\WhatsUp\whatsup_vlms-main\data")
FILES = {
    "coco_qa_one_obj.json": "1RsMdpE9mmwnK4zzMPpC1-wTU_hNis-dq",
    "coco_qa_two_obj.json": "1TCEoM0mgFmz8T4cF7PQ3XJmO6JjtiQ-s",
    "controlled_images_dataset.json": "1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm",
}


def main() -> int:
    import gdown
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, fid in FILES.items():
        dst = ROOT / name
        if dst.is_file() and dst.stat().st_size > 1000:
            print(f"{name} exists {dst.stat().st_size/1e6:.1f} MB", flush=True)
            continue
        url = f"https://drive.google.com/uc?id={fid}"
        try:
            gdown.download(url, str(dst), quiet=False)
            print(f"{name}: {dst.stat().st_size/1e6:.2f} MB", flush=True)
        except Exception as e:                     # noqa: BLE001
            print(f"{name} FAILED {type(e).__name__} {str(e)[:90]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
