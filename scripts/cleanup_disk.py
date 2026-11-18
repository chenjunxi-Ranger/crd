"""One-off disk cleanup: delete verified duplicates / obsolete caches."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测")

DIRS = [
    "mymodel5/ckpt",
    "mymodel4/cache",
    "baseline_tr/npc_runs",
    "baseline_tr/data/precomp",
    "baseline_tr/third_party/2024-TIP-CREAM/output",
]


def sz(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    t = 0
    for dp, _, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return t


def main(apply: bool) -> int:
    total0 = sz(ROOT)
    plan = []

    for f in (ROOT / "LEMoN+CLIPsim/weights").rglob("pytorch_model.bin"):
        if f.stat().st_size > 200 * 1024 * 1024:
            plan.append(("file", f))

    for d in DIRS:
        p = ROOT / d
        if p.exists():
            plan.append(("dir", p))

    for pat in ("*seed1*.csv", "*seed2*.csv"):
        for f in (ROOT / "数据集/derived").rglob(pat):
            plan.append(("file", f))

    freed = 0
    for kind, p in plan:
        s = sz(p)
        print(f"{kind:4s} {s/1e9:7.2f} GB  {p.relative_to(ROOT)}")
        if apply:
            if kind == "dir":
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink()
                except OSError as e:
                    print("   !!", e)
        freed += s

    print(f"\nplanned freed: {freed/1e9:.2f} GB (apply={apply})")
    if apply:
        print(f"disk now: {sz(ROOT)/1e9:.2f} GB (was {total0/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--apply" in sys.argv))
