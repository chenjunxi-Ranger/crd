"""Report disk usage by directory and detect duplicate large files."""
from __future__ import annotations

import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
MIN_MB = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0


def size_of(p: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def digest(p: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    print(f"ROOT = {ROOT}\n")
    # level-1 sizes
    lvl1 = []
    for d in sorted(ROOT.iterdir()):
        if d.is_dir():
            lvl1.append((size_of(d), d.name))
    lvl1.sort(reverse=True)
    total = 0
    print("== 一级目录 ==")
    for s, name in lvl1:
        total += s
        print(f"{human(s):>10}  {name}")
    print(f"{human(total):>10}  (合计)\n")

    # level-2 for the top 6
    print("== 前 6 个目录的内部构成（>50MB）==")
    for s, name in lvl1[:6]:
        print(f"\n-- {name}  {human(s)}")
        subs = []
        for p in (ROOT / name).iterdir():
            if p.is_dir():
                subs.append((size_of(p), p.name))
            elif p.is_file():
                subs.append((p.stat().st_size, p.name))
        subs.sort(reverse=True)
        for ss, sn in subs[:10]:
            if ss > 50 * 1024 * 1024:
                print(f"   {human(ss):>10}  {sn}")

    # duplicate detection by size then md5
    print("\n== 重复文件检测（>%.0fMB）==" % MIN_MB)
    by_size = defaultdict(list)
    for dirpath, _, files in os.walk(ROOT):
        for f in files:
            fp = Path(dirpath) / f
            try:
                s = fp.stat().st_size
            except OSError:
                continue
            if s >= MIN_MB * 1024 * 1024:
                by_size[s].append(fp)
    dup_groups = []
    for s, files in by_size.items():
        if len(files) < 2:
            continue
        by_hash = defaultdict(list)
        for fp in files:
            try:
                by_hash[digest(fp)].append(fp)
            except OSError:
                pass
        for h, fl in by_hash.items():
            if len(fl) > 1:
                dup_groups.append((s, fl))
    dup_groups.sort(key=lambda x: -x[0] * (len(x[1]) - 1))
    wasted = 0
    for s, fl in dup_groups:
        wasted += s * (len(fl) - 1)
        print(f"\n{human(s)} x {len(fl)}  (可省 {human(s*(len(fl)-1))})")
        for fp in fl:
            print("   ", fp.relative_to(ROOT))
    print(f"\n可回收合计: {human(wasted)}")

    # same-name dataset dirs
    print("\n== 同名目录（可能的重复数据集）==")
    names = defaultdict(list)
    for dirpath, dirnames, _ in os.walk(ROOT):
        depth = len(Path(dirpath).relative_to(ROOT).parts)
        if depth > 2:
            continue
        for d in dirnames:
            names[d].append(Path(dirpath) / d)
    for n, paths in sorted(names.items()):
        if len(paths) > 1 and any(k in n.lower() for k in ("coco", "flickr", "mmimdb", "vg", "visual")):
            print(f"\n{n}:")
            for p in paths:
                print(f"   {human(size_of(p)):>10}  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
