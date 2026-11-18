"""AFC-Donor feasibility on COCO only (no Visual Genome, no human labels).

Counts, for the official COCO val split, how many fully-automatic fact-conflict
donor pairs can be mined from Instances + the original captions:

  Add      : target lacks category c, donor has c, and donor's caption names c
  Quantity : same category present in both, but the count mentioned in donor's
             caption agrees with donor and contradicts target

Everything here is automatic; no LLM, no manual verification.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import PROJECT   # noqa: E402

ANN = PROJECT / "数据集" / "MS-COCO" / "annotations"

NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
       "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def load():
    with open(ANN / "captions_val2014.json", encoding="utf-8") as f:
        cap = json.load(f)
    with open(ANN / "instances_val2014.json", encoding="utf-8") as f:
        ins = json.load(f)
    cat_name = {c["id"]: c["name"].lower() for c in ins["categories"]}
    caps = defaultdict(list)
    for a in cap["annotations"]:
        caps[a["image_id"]].append(a["caption"])
    counts = defaultdict(Counter)
    for a in ins["annotations"]:
        counts[a["image_id"]][a["category_id"]] += 1
    return caps, counts, cat_name


def main() -> int:
    caps, counts, cat_name = load()
    imgs = sorted(caps)
    print(f"val images with captions: {len(imgs)}; "
          f"images with instance anns: {len(counts)}", flush=True)

    # category presence sets
    pres = {i: set(counts[i]) for i in imgs if i in counts}
    inv = defaultdict(list)
    for i, cs in pres.items():
        for c in cs:
            inv[c].append(i)

    # singular/plural tolerant match of a category name inside a caption
    def mentions(cap: str, name: str) -> bool:
        n = name.rstrip("s")
        return re.search(rf"\b{n}(s|es)?\b", cap.lower()) is not None

    add_triples, qty_triples = [], []
    rng = np.random.default_rng(0)
    for t in imgs:
        if t not in pres:
            continue
        ct = pres[t]
        cand = set()
        for c in ct:                      # donors sharing >=1 category (scene similarity)
            for d in inv[c]:
                if d != t:
                    cand.add(d)
        if not cand:
            continue
        cand = list(cand)
        rng.shuffle(cand)
        for d in cand[:40]:
            cd = pres[d]
            extra = cd - ct
            for c in list(extra)[:6]:
                name = cat_name.get(c, "")
                for dc in caps[d]:
                    if name and mentions(dc, name):
                        add_triples.append((t, d, c, dc))
                        break
            # quantity: shared category with different counts, numeral in donor caption
            for c in (ct & cd):
                nt, nd = counts[t][c], counts[d][c]
                if nt == nd:
                    continue
                for dc in caps[d]:
                    m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
                                  dc.lower())
                    if not m:
                        continue
                    v = NUM.get(m.group(1)) or (int(m.group(1)) if m.group(1).isdigit() else None)
                    if v == nd and v != nt:
                        qty_triples.append((t, d, c, dc))
                        break
                break

    add_t = {(a[0], a[1]) for a in add_triples}
    qty_t = {(a[0], a[1]) for a in qty_triples}
    out = {
        "val_images": len(imgs),
        "images_with_instances": len(pres),
        "add_triples": len(add_triples),
        "add_unique_pairs": len(add_t),
        "add_targets": len({a[0] for a in add_triples}),
        "qty_triples": len(qty_triples),
        "qty_unique_pairs": len(qty_t),
        "qty_targets": len({a[0] for a in qty_triples}),
        "add_target_rate": round(len({a[0] for a in add_triples}) / max(len(pres), 1), 3),
        "qty_target_rate": round(len({a[0] for a in qty_triples}) / max(len(pres), 1), 3),
        "sample_add": [(a[0], a[1], cat_name[a[2]], a[3]) for a in add_triples[:5]],
        "sample_qty": [(a[0], a[1], cat_name[a[2]], a[3]) for a in qty_triples[:5]],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    import json as J
    p = ROOT / "results" / "bench" / "afc_feasibility.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(J.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
