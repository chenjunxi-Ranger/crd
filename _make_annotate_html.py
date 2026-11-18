import csv
import json
from pathlib import Path
from urllib.parse import quote

KIT = Path("results/fgbench/p0_baselines/annotation_kit")
SRC = KIT / "blind" / "pairs.csv"
OUT = KIT / "blind" / "annotate_v2.html"

rows = []
with SRC.open("r", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        p = Path(r["image_path"].strip().replace("\\", "/"))
        url = "file:///" + quote(str(p))
        rows.append({
            "id": r["pair_id"].strip(),
            "img": url,
            "cap": r["caption"].strip(),
        })

data = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
html = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FGBench blind annotation v2</title>
<style>
:root{font-synthesis:none}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#111;color:#eee}
.wrap{max-width:1150px;margin:0 auto;padding:18px}
h1{font-size:22px;margin:0 0 8px}
.card{background:#1c1c1c;border:1px solid #333;border-radius:12px;padding:16px;margin-top:14px}
.grid{display:grid;grid-template-columns:minmax(380px,1fr) minmax(300px,.8fr);gap:18px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
img{max-width:100%;max-height:62vh;object-fit:contain;border-radius:8px;background:#000;min-height:280px}
.cap{font-size:22px;line-height:1.45;background:#262626;border-radius:10px;padding:18px;margin:0}
.small{color:#aaa;font-size:13px}
button{border:0;border-radius:10px;padding:12px 18px;font-size:17px;font-weight:600;cursor:pointer;color:#fff}
button:hover{filter:brightness(1.15)}
button:disabled{opacity:.45;cursor:not-allowed}
.clean{background:#157347}.noisy{background:#bb2d3b}.unsure{background:#6c757d}
.nav{background:#333}.export{background:#0d6efd}
#bar{height:10px;background:#333;border-radius:6px;overflow:hidden;margin:10px 0}
#fill{height:100%;width:0;background:#20c997;transition:.2s}
kbd{background:#444;border-radius:5px;padding:2px 6px;font-size:12px}
.badge{background:#333;border-radius:999px;padding:4px 10px;font-size:13px}
#msg{min-height:20px;color:#ffc107;font-size:14px}
input,select{font-size:16px;padding:10px;border-radius:8px;border:1px solid #444;background:#262626;color:#eee;width:100%}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.stat{background:#262626;border-radius:10px;padding:10px;margin-top:10px}
</style>
</head>
<body>
<div class="wrap">
  <h1>FGBench Blind Annotation</h1>
  <div class="small">Judge only whether the caption correctly describes the image. Do not open key.csv.</div>

  <div id="start" class="card">
    <label>Annotator name / initials</label>
    <input id="name" placeholder="e.g. A, B, CJ" autocomplete="off">
    <div class="row" style="margin-top:12px">
      <button class="export" onclick="start()">Start / Resume</button>
      <span class="small">Your annotations are saved locally under this name.</span>
    </div>
  </div>

  <div id="work" style="display:none">
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <span class="badge" id="pos">Item 1 / 400</span>
        <span class="badge" id="annot">—</span>
        <span class="badge" id="left">Unfinished: 0</span>
      </div>
      <div id="bar"><div id="fill"></div></div>
      <div class="grid">
        <div><img id="im" alt="image"></div>
        <div>
          <p class="cap" id="cap"></p>
          <div class="row">
            <button class="clean" onclick="mark(1)">Clean · 1</button>
            <button class="noisy" onclick="mark(0)">Noisy · 2</button>
            <button class="unsure" onclick="mark(-1)">Unsure · 3</button>
          </div>
          <div class="row" style="margin-top:10px">
            <button class="nav" onclick="prev()">◀ Prev</button>
            <button class="nav" onclick="next()">Next ▶</button>
            <button class="nav" onclick="nextUnanswered()">Next unanswered</button>
          </div>
          <div class="row" style="margin-top:12px">
            <button class="export" onclick="exp()">Export CSV</button>
            <span class="small">Export file: <span id="fname"></span></span>
          </div>
          <div id="msg"></div>
        </div>
      </div>
      <div class="stat small">
        Shortcuts: <kbd>1</kbd> Clean · <kbd>2</kbd> Noisy · <kbd>3</kbd> Unsure ·
        <kbd>←</kbd> Prev · <kbd>→</kbd> Next · <kbd>U</kbd> next unanswered
      </div>
    </div>
  </div>
</div>

<script>
const P = __DATA__;
let name = "", A = {}, i = 0, K = "";

function start(){
  name = document.getElementById("name").value.trim() || "annotator";
  K = "fgb_annot_v2_" + name + "_" + P.length;
  try{ A = JSON.parse(localStorage.getItem(K) || "{}"); }catch(e){ A = {}; }
  document.getElementById("start").style.display = "none";
  document.getElementById("work").style.display = "block";
  document.getElementById("annot").textContent = name;
  document.getElementById("fname").textContent = "annotator_" + name + ".csv";
  for(let k=0;k<P.length;k++){ if(!(P[k].id in A)){ i=k; break; } }
  show();
}
function save(){ localStorage.setItem(K, JSON.stringify(A)); }
function show(){
  const p=P[i]; document.getElementById("im").src=p.img;
  document.getElementById("cap").textContent=p.cap;
  document.getElementById("pos").textContent=`Item ${i+1} / ${P.length}`;
  const n=Object.keys(A).length;
  document.getElementById("fill").style.width=(n/P.length*100)+"%";
  const un=P.length-n; document.getElementById("left").textContent="Unfinished: "+un;
  document.getElementById("msg").textContent = (p.id in A) ? "Already labelled: "+A[p.id] : "";
}
function mark(v){ A[P[i].id]=v; save();
  if(i<P.length-1){ i++; } else { msg("Last item labelled. Export CSV."); }
  show();
}
function prev(){ if(i>0){ i--; show(); } }
function next(){ if(i<P.length-1){ i++; show(); } }
function nextUnanswered(){
  for(let k=1;k<=P.length;k++){ const j=(i+k)%P.length; if(!(P[j].id in A)){ i=j; show(); return; } }
  msg("All items labelled.");
}
function msg(s){ document.getElementById("msg").textContent=s; }
function exp(){
  const answered=P.filter(p=>p.id in A);
  if(answered.length===0){ msg("Nothing to export."); return; }
  let s="pair_id,label\n";
  for(const p of P){ if(p.id in A) s+=p.id+","+A[p.id]+"\n"; }
  const b=new Blob([s],{type:"text/csv"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(b);
  a.download="annotator_"+name+".csv"; a.click();
  msg(`Exported ${answered.length}/${P.length} answered items.`);
}
document.addEventListener("keydown",e=>{
  if(document.getElementById("work").style.display==="none") return;
  if(e.key==="1") mark(1); else if(e.key==="2") mark(0);
  else if(e.key==="3") mark(-1); else if(e.key==="ArrowLeft") prev();
  else if(e.key==="ArrowRight") next(); else if(e.key.toLowerCase()==="u") nextUnanswered();
});
document.getElementById("name").addEventListener("keydown",e=>{ if(e.key==="Enter") start(); });
</script>
</body>
</html>
"""
OUT.write_text(html.replace("__DATA__", data), encoding="utf-8")
print("wrote", OUT, "items", len(rows))
