"""P0-A: build a 400-pair stratified blind annotation kit.

Sampling: 40 cycles per facet (5 facets), one clean and one noisy pair per cycle
=> 400 pairs, 200 clean / 200 noisy. Ground truth stays in key.csv; the
annotation sheet contains no labels.

Output: annotation_kit/{pairs.csv,key.csv,annotate.html,kappa.py,INSTRUCTIONS.md}
Run:    python .../p0_a_build_kit.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "fgbench" / "data" / "fgbench_v1_strict"
DATA = ROOT.parent / "数据集"
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
OUT = ROOT / "results" / "fgbench" / "p0_baselines" / "annotation_kit"
OUT.mkdir(parents=True, exist_ok=True)
FACETS = ["Entity", "Quantity", "Attribute", "Action", "Relation"]
PER_FACET = 40
RNG = np.random.default_rng(0)


def image_path(row) -> str:
    if row.dataset == "COCO":
        return str(DATA / "MS-COCO" / str(row.target_file))
    vg = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()
    iid = int(str(row.target_image_id).split(":")[1])
    return str(DATA / "VisualGenome" / str(vg[iid]))


def main() -> int:
    te = pd.read_parquet(BENCH / "test.parquet")
    rows, key = [], []
    for facet in FACETS:
        d = te[te.facet == facet]
        cycles = np.array(sorted(d.cycle_id.unique()))
        pick = RNG.choice(cycles, min(PER_FACET, len(cycles)), replace=False)
        n = 0
        for c in pick:
            g = d[d.cycle_id == c]
            for lab in (0, 1):
                s = g[g.pair_label == lab]
                if s.empty:
                    continue
                r = s.iloc[0]
                pid = f"{facet[:3].upper()}-{n:03d}"
                rows.append({
                    "pair_id": pid, "facet": facet,
                    "image_path": image_path(r), "caption": str(r.caption),
                    "label_A": "", "label_B": "", "notes": "",
                })
                key.append({"pair_id": pid, "facet": facet,
                            "true_label": "clean" if lab == 0 else "noisy"})
                n += 1
    df = pd.DataFrame(rows)
    kd = pd.DataFrame(key)
    df.to_csv(OUT / "pairs.csv", index=False, encoding="utf-8-sig")
    kd.to_csv(OUT / "key.csv", index=False, encoding="utf-8-sig")

    # self-contained annotator (no server needed)
    payload = json.dumps([
        {"id": r.pair_id, "facet": r.facet,
         "img": Path(r.image_path).as_uri(), "cap": r.caption}
        for r in df.itertuples()], ensure_ascii=False)
    page = f"""<!doctype html><meta charset="utf-8">
<title>FGBench blind annotation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px}}
img{{max-width:100%;max-height:420px;border:1px solid #ccc;border-radius:6px}}
.cap{{font-size:18px;margin:12px 0;padding:12px;background:#f6f6f6;border-radius:6px}}
button{{font-size:16px;padding:10px 18px;margin:6px 8px 6px 0;cursor:pointer}}
#bar{{height:8px;background:#eee;border-radius:4px;margin:10px 0}}
#fill{{height:8px;background:#3a7;border-radius:4px;width:0}}
small{{color:#666}}
</style>
<h2>FGBench blind annotation</h2>
<p><small>Judge whether the caption correctly describes the image. Do not infer the
dataset label. 1 = clean, 0 = noisy, ? = unsure. Progress is stored in this browser;
use “Export CSV” when done and send the file to the organizer.</small></p>
<div id="bar"><div id="fill"></div></div>
<div id="meta"></div>
<img id="im" alt="pair image">
<div class="cap" id="cap"></div>
<button onclick="mark(1)">Clean (1)</button>
<button onclick="mark(0)">Noisy (0)</button>
<button onclick="mark(-1)">Unsure (?)</button>
<button onclick="prev()">Back</button>
<button onclick="exp()">Export CSV</button>
<script>
const P={payload};
let i=0; const K='fgb_a_'+P.length;
let A=JSON.parse(localStorage.getItem(K)||'{{}}');
function show(){{const p=P[i];document.getElementById('im').src=p.img;
document.getElementById('cap').textContent=p.cap;
document.getElementById('meta').textContent=p.id+' · '+p.facet+' · '+(i+1)+'/'+P.length;
document.getElementById('fill').style.width=(Object.keys(A).length/P.length*100)+'%';}}
function mark(v){{A[P[i].id]=v;localStorage.setItem(K,JSON.stringify(A));
if(i<P.length-1){{i++;show();}}else{{alert('Done. Export CSV.');}}}}
function prev(){{if(i>0){{i--;show();}}}}
function exp(){{let s='pair_id,label\\n';for(const p of P){{if(p.id in A)s+=p.id+','+A[p.id]+'\\n';}}
const b=new Blob([s],{{type:'text/csv'}});const a=document.createElement('a');
a.href=URL.createObjectURL(b);a.download='annotator.csv';a.click();}}
show();
</script>"""
    (OUT / "annotate.html").write_text(page, encoding="utf-8")

    kappa = '''"""Cohen's kappa for the 400-pair blind audit.

Usage:
  python kappa.py annotator_A.csv annotator_B.csv
CSV columns: pair_id,label  (label: 1 clean, 0 noisy, -1 unsure)
"""
import sys
import pandas as pd


def kappa(a, b):
    a, b = pd.Series(a), pd.Series(b)
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    a = pd.read_csv(sys.argv[1]).set_index("pair_id").label
    b = pd.read_csv(sys.argv[2]).set_index("pair_id").label
    key = pd.read_csv("key.csv").set_index("pair_id")
    d = pd.concat([a.rename("A"), b.rename("B")], axis=1).dropna()
    d = d[(d.A != -1) & (d.B != -1)]
    d = d.join(key, how="inner")
    print("n =", len(d))
    print("Cohen kappa (A vs B) =", round(kappa(d.A, d.B), 4))
    print("accuracy A vs key =", round((d.A == (d.true_label == "clean").astype(int)).mean(), 4))
    print("accuracy B vs key =", round((d.B == (d.true_label == "clean").astype(int)).mean(), 4))
    for f, g in d.groupby("facet"):
        print(f, "n=", len(g), "kappa=", round(kappa(g.A, g.B), 4))


if __name__ == "__main__":
    main()
'''
    (OUT / "kappa.py").write_text(kappa, encoding="utf-8")

    instr = f"""# FGBench 400-pair blind annotation

- {len(df)} pairs: 40 cycles per facet x 2 pairs (one clean, one noisy).
- Two annotators work independently. Do not open `key.csv` before both sheets are done.
- Open `annotate.html` in a browser (double-click). Judge only whether the caption
  correctly describes the image. `Unsure` is allowed and excluded from kappa.
- Export CSV at the end; send `annotator.csv` to the organizer.
- Compute agreement: `python kappa.py annotator_A.csv annotator_B.csv`.
- Expected time: 45–75 min per annotator.
"""
    (OUT / "INSTRUCTIONS.md").write_text(instr, encoding="utf-8")
    print(f"wrote {len(df)} pairs to {OUT}")
    print(df.groupby("facet").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
