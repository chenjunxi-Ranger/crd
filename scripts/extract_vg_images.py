"""Move + extract the two VG image archives into the VisualGenome folder."""
from __future__ import annotations

import shutil
import sys
import time
import zipfile
from pathlib import Path

VG = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
SRC2 = Path(r"D:\迅雷下载\images2.zip")


def main() -> int:
    dst2 = VG / "images2.zip"
    if not dst2.is_file() or dst2.stat().st_size != 5471658058:
        print("moving images2.zip ...", flush=True)
        shutil.move(str(SRC2), str(dst2))
    print("images2.zip at", dst2, f"{dst2.stat().st_size/1e9:.2f} GB", flush=True)

    for name in ("images.zip", "images2.zip"):
        p = VG / name
        t0 = time.time()
        print(f"extracting {name} ...", flush=True)
        with zipfile.ZipFile(p) as z:
            z.extractall(VG)
        print(f"done {name} in {time.time()-t0:.0f}s", flush=True)

    for d in ("VG_100K", "VG_100K_2"):
        p = VG / d
        n = sum(1 for _ in p.glob("*.jpg")) if p.is_dir() else 0
        print(f"{d}: {n} jpg", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
