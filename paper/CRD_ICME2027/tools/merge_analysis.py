"""Merge sec/7_analysis.tex into sec/6_experiments.tex (ICME has 7 sections only).

The analysis chapter becomes subsections 6.6-6.9 of the Experiments section.
Run once from the project root:  python tools/merge_analysis.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
exp = ROOT / "sec" / "6_experiments.tex"
ana = ROOT / "sec" / "7_analysis.tex"

text_exp = exp.read_text(encoding="utf-8")
text_ana = ana.read_text(encoding="utf-8")

# drop the analysis \section{...}\label{...} header but keep its subsections
text_ana = re.sub(r"\\section\{[^}]*\}\s*\n\\label\{sec:analysis\}\s*\n", "", text_ana)
text_ana = re.sub(r"\\section\{[^}]*\}", "", text_ana, count=1)

header = (
    "\n% ---- merged from the former Analysis chapter (ICME: 7 sections only) ----\n"
    "\\subsection{What the Calibration Changes and Does Not Change}\n"
    "\\label{sec:analysis}\n"
)
# the first subsection title of the analysis file is dropped, we re-add it above
text_ana = re.sub(r"\\subsection\{What the Calibration Does and Does Not Change\}\s*\n", "", text_ana, count=1)

merged = text_exp.rstrip() + "\n" + header + text_ana.lstrip()
exp.write_text(merged, encoding="utf-8")
ana.unlink()
print("merged:", exp.name, "->", len(merged), "chars")
print("removed:", ana.name)
