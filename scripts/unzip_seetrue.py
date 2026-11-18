"""Extract SeeTRUE images.zip."""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

SC = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\SeeTRUE")


def main() -> int:
    p = SC / "images.zip"
    z = zipfile.ZipFile(p)
    names = z.namelist()
    print("entries", len(names), "first", names[:3], flush=True)
    t0 = time.time()
    z.extractall(SC)
    print("done", round(time.time() - t0), "s", flush=True)
    imgs = list((SC / "images").rglob("*"))
    print("extracted paths", len(imgs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
