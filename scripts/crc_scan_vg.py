"""Scan images2.zip entry CRCs and report which entries are corrupted."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

P = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome\images2.zip")
OUT = Path(r"D:\PyCharm\PythonProject\异常标签检测\mymodel7\results\vg_zip_bad.json")


def main() -> int:
    bad = []
    with zipfile.ZipFile(P) as z:
        infos = z.infolist()
        print("entries", len(infos), flush=True)
        for i, info in enumerate(infos):
            if info.is_dir():
                continue
            try:
                with z.open(info) as f:
                    while f.read(1 << 20):
                        pass
            except Exception as e:                 # noqa: BLE001
                bad.append({"name": info.filename,
                            "header_offset": info.header_offset,
                            "compress_size": info.compress_size,
                            "err": type(e).__name__})
            if (i + 1) % 5000 == 0:
                print(f"  scanned {i+1}, bad {len(bad)}", flush=True)
    print("bad entries:", len(bad), flush=True)
    for b in bad[:10]:
        print("   ", b["name"], b["err"], b["header_offset"], b["compress_size"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bad), encoding="utf-8")
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
