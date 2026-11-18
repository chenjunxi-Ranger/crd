"""P0-A (blind package): shuffle the 400 pairs, strip labels and keys.

Output: annotation_kit/blind/{pairs.csv, annotate.html, README.md}
        annotation_kit/check_paths.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KIT = (Path(__file__).resolve().parents[3] / "results" / "fgbench" /
       "p0_baselines" / "annotation_kit")
BLIND = KIT / "blind"
BLIND.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(KIT / "pairs.csv", encoding="utf-8-sig")
key = pd.read_csv(KIT / "key.csv", encoding="utf-8-sig")
check = df[["pair_id", "facet", "image_path"]].copy()
check["exists"] = check.image_path.map(lambda p: Path(p).is_file())
check.to_csv(KIT / "check_paths.csv", index=False, encoding="utf-8-sig")

# shuffle so clean/noisy are not alternating
df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
df = df[["pair_id", "facet", "image_path", "caption"]]
df.to_csv(BLIND / "pairs.csv", index=False, encoding="utf-8-sig")

payload = json.dumps([
    {"id": r.pair_id, "facet": r.facet, "img": Path(r.image_path).as_uri(),
     "cap": r.caption} for r in df.itertuples()], ensure_ascii=False)
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
<p><small>Judge whether the caption correctly describes the image. 1 = clean,
0 = noisy, ? = unsure. Progress is stored in this browser. Export CSV when done.
Do not open the parent directory's key.csv.</small></p>
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
let i=0; const K='fgb_blind_'+P.length;
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
(BLIND / "annotate.html").write_text(page, encoding="utf-8")
(BLIND / "README.md").write_text(
    "# Blind package\n\n"
    "Hand this folder (not its parent) to each annotator. It contains only\n"
    "`pairs.csv` and `annotate.html`; the key stays in the parent directory.\n"
    "Run `python ../kappa.py A.csv B.csv` from the parent after both sheets are done.\n",
    encoding="utf-8")
print("wrote", len(df), "shuffled pairs; images exist:",
      int(check.exists.sum()), "/", len(check))
