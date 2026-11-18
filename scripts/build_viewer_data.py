"""Build batch01_data.js for the HTML annotation viewer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR   # noqa: E402


def read_any(p: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"cannot decode {p}")


def main() -> int:
    d = RESULTS_DIR / "audit" / "formal"
    blind = read_any(d / "batch01_blind.csv")
    zh = json.loads((d / "batch01_zh.json").read_text(encoding="utf-8"))

    recs = []
    for _, r in blind.iterrows():
        p = str(r["image"]).replace("\\", "/")
        recs.append({"id": r["anon_id"],
                     "img": "file:///" + p,
                     "en": str(r["caption"]).replace("\n", " ").strip(),
                     "zh": zh.get(r["anon_id"], "")})
    out = "window.BATCH = " + json.dumps(recs, ensure_ascii=False) + ";\n"
    (d / "batch01_data.js").write_text(out, encoding="utf-8")
    print(f"wrote {len(recs)} records -> batch01_data.js")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
