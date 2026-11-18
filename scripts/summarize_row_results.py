"""Summarise every row-target CRD result across backbones and facets."""
from __future__ import annotations

import collections
import glob
import json

rows = []
for f in glob.glob("results/fgbench/crdbb_*.json"):
    d = json.load(open(f, encoding="utf-8"))
    if "backbone" not in d:
        continue
    # NB: crdbb_*.json does not record the supervision target; the files on disk
    # are the LAST run for each (backbone, facet, seed), i.e. the row-target runs.
    rows.append(d)

g = collections.defaultdict(list)
for d in rows:
    g[(d["backbone"], d["facet"])].append(d)

for (b, fac), v in sorted(g.items()):
    raw = v[0]["auc_raw"]
    crd = sum(x["auc_crd"] for x in v) / len(v)
    sd = (sum((x["auc_crd"] - crd) ** 2 for x in v) / (len(v) - 1)) ** 0.5 \
        if len(v) > 1 else 0.0
    ap = sum(x["auprc_crd"] for x in v) / len(v)
    print(f"{b:9s} {fac:10s} n={len(v)} raw={raw:.4f} crd={crd:.4f} sd={sd:.4f} "
          f"d={100*(crd-raw):+.2f} ap={ap:.4f} "
          f"2afc={v[0]['acc2afc_raw']:.6f}/{v[0]['acc2afc_crd']:.6f}")
