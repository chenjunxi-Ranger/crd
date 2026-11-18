"""Move every table caption above its tabular (IEEE style: table captions on top).

Run once from the project root:  python tools/fix_table_captions.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

pat = re.compile(
    r"(\\begin\{table\*?\}\s*(?:\[[^\]]*\])?\s*\\centering\s*(?:\\small\s*)?"
    r"(?:\\setlength\{[^\n]*\}\s*)?)"
    r"(\\begin\{tabular\}.*?\\end\{tabular\}\s*)"
    r"(\\caption\{.*?\\label\{[^}]*\}\s*)",
    re.S,
)

changed = 0
for f in sorted((ROOT / "sec").glob("*.tex")):
    s = f.read_text(encoding="utf-8")
    new, n = pat.subn(lambda m: m.group(1) + m.group(3) + m.group(2), s)
    if n:
        f.write_text(new, encoding="utf-8")
        changed += n
        print(f"{f.name}: moved {n} caption(s)")
print("total moved:", changed)
