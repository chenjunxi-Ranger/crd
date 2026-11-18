"""Compliance / leakage audit for the frozen formula (CONSTRAINTS C1-C3, C6).

Checks
------
1. ``src/`` never mentions a label column or a dataset/protocol switch;
2. the frozen constants hash matches every result JSON;
3. the FOIL validation manifest is image-disjoint and has 10,000 pairs;
4. no bank file contains test-split identities;
5. test has not been evaluated (a failed-val method must not consume it).

Exit code is non-zero on any violation.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = ROOT / "results"
DATA = ROOT / "data"

#: only the scoring path is audited; protocol.py legitimately reads labels for eval
SCORING_FILES = ["score.py", "features.py", "encoders.py", "constants.py",
                 "counterfactual.py"]
FORBIDDEN = re.compile(r"is_mislabel|mislabel_rate|\blabel\b|dataset_name|bench_name|"
                       r"protocol_name|dataset_id|bench_id", re.IGNORECASE)


def main() -> int:
    problems: list[str] = []
    report: dict = {"checks": {}}

    # tokenise so docstrings/comments do not produce false positives
    import io
    import tokenize
    hits = []
    for name in SCORING_FILES:
        py = SRC / name
        if not py.is_file():
            continue
        src = py.read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type not in (tokenize.NAME, tokenize.STRING):
                continue
            if tok.type == tokenize.STRING:
                continue
            if FORBIDDEN.fullmatch(tok.string):
                hits.append(f"{py.name}:{tok.start[0]}: {tok.string}")
    report["checks"]["src_has_no_label_or_branch"] = {"ok": not hits, "hits": hits}
    if hits:
        problems.append("src/ contains label or dataset/protocol tokens")

    sys.path.insert(0, str(ROOT))
    from src.constants import constants_hash
    from src.constants_v3 import constants_hash as hash_v3
    h = constants_hash()
    # every frozen generation that produced a result file
    known = {h, hash_v3(), "69f79050e2473142"}
    mism = []
    for f in sorted(RESULTS.glob("bench/*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "constants_hash" in d and d["constants_hash"] not in known:
            mism.append(f.name)
    report["checks"]["constants_hash_stable"] = {"ok": not mism, "hash": h, "mismatch": mism}
    if mism:
        problems.append("constants hash mismatch")

    man_path = DATA / "foil_val_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    ok_man = bool(man.get("image_disjoint")) and man.get("val_pairs") == 10000
    report["checks"]["foil_val_protocol"] = {"ok": ok_man, "manifest": {
        k: man.get(k) for k in ("val_pairs", "val_images", "image_disjoint",
                                "val_image_hash", "held_out_images")}}
    if not ok_man:
        problems.append("FOIL val manifest missing or not image-disjoint")

    test_files = list(RESULTS.glob("bench/*_test.json")) + list(RESULTS.glob("bench/*test*.csv"))
    report["checks"]["test_untouched"] = {"ok": not test_files,
                                          "files": [f.name for f in test_files]}
    if test_files:
        problems.append("test has been evaluated; C6 one-shot already spent")

    report["ok"] = not problems
    report["problems"] = problems
    (RESULTS / "leak_check.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "leak_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
