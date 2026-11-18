"""Replace the tab:main float in sec/6_experiments.tex with the regenerated table.

The generated table lives in results/fgbench/p0_baselines/main_table_v2.tex and
already carries the two new feature-debiasing baselines.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sec = ROOT / "sec" / "6_experiments.tex"
gen = Path(r"D:\PyCharm\PythonProject\异常标签检测\mymodel7\results\fgbench\p0_baselines\main_table_v2.tex")

new_block = gen.read_text(encoding="utf-8").strip()
text = sec.read_text(encoding="utf-8")

pat = re.compile(r"\\begin\{table\*\}\[t\].*?\\label\{tab:main\}.*?\\end\{table\*\}", re.S)
if not pat.search(text):
    raise SystemExit("tab:main float not found")
text = pat.sub(lambda m: new_block, text, count=1)
sec.write_text(text, encoding="utf-8")
print("spliced tab:main ->", sec.name)
