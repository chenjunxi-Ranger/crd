"""Step 02: parse every original caption into (mentioned classes, quantity claims).

spaCy handles lemmatisation, noun chunks and the numeral -> head-noun dependency,
so "two dogs" binds 2 to *dog* instead of a naive substring test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgbench.common import (COCO_ANN, INTER, NUMWORDS, class_map_terms,  # noqa: E402
                            jdump, load_captions, load_instances)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="captions_val2014.json")
    ap.add_argument("--instances", default="instances_val2014.json")
    ap.add_argument("--out", default="caption_facts.parquet")
    args = ap.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    _, _, cats, _ = load_instances(COCO_ANN / args.instances)
    term2cls = class_map_terms(cats)
    caps = load_captions(COCO_ANN / args.captions)

    rows = []
    for iid, texts in caps.items():
        for t in texts:
            doc = nlp(t)
            mentioned, qty = set(), {}
            for nc in doc.noun_chunks:
                head = nc.root.lemma_.lower()
                cls = term2cls.get(head) or term2cls.get(nc.root.text.lower())
                if not cls:
                    continue
                mentioned.add(cls)
                nums = []
                for tok in nc:
                    w = tok.lower_
                    if w in NUMWORDS:
                        nums.append(NUMWORDS[w])
                    elif w.isdigit():
                        nums.append(int(w))
                if len(nums) == 1 and nums[0] >= 2:
                    qty[cls] = nums[0]
            # tokens outside noun chunks
            for tok in doc:
                if tok.pos_ in ("NOUN", "PROPN"):
                    cls = term2cls.get(tok.lemma_.lower())
                    if cls:
                        mentioned.add(cls)
            rows.append({"image_id": iid, "caption": t,
                         "mentioned": sorted(mentioned),
                         "quantities": qty,
                         "n_mentioned": len(mentioned)})
    df = pd.DataFrame(rows)
    INTER.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTER / args.out, index=False)
    jdump({"captions": int(len(df)),
           "with_mention": int((df.n_mentioned > 0).sum()),
           "with_quantity": int((df.quantities.map(len) > 0).sum()),
           "mean_mentions": round(float(df.n_mentioned.mean()), 3)},
          INTER / "caption_facts_meta.json")
    print(df.shape, "with_mention", int((df.n_mentioned > 0).mean() * 100), "%",
          "with_qty", int((df.quantities.map(len) > 0).mean() * 100), "%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
