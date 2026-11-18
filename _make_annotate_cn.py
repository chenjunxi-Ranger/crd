import base64
import csv
import html
import io
import json
from pathlib import Path

from PIL import Image

KIT = Path("results/fgbench/p0_baselines/annotation_kit")
SRC = KIT / "blind" / "pairs.csv"
OUT = KIT / "blind" / "annotate_v3_cn.html"

ZH = {
"white cords on the desk": "桌子上的白色绳索",
"Three men sitting at a table that has three pizzas on it.": "三名男子坐在放有三份披萨的桌旁。",
"wires on the desk": "桌子上的电线",
"vase with flowers on the window sill": "窗台上有花的花瓶",
"green and brown grass": "绿色和棕色的草地",
"pole on the umbrella.": "雨伞上的杆子。",
"blue coloring in the boat.": "船上的蓝色色彩。",
"a cat sitting on a bed.": "一只猫坐在床上。",
"A cake made to look like a white pimp hat.": "做成白色宽檐帽样子的蛋糕。",
"a person sitting on the ground": "一个人坐在地上",
"crinkled white plastic bag": "起皱的白色塑料袋",
"a shadow casting on toilet": "投射在马桶上的阴影",
"man wearing black shirt": "穿黑色衬衫的男子",
"clear tape and scissors laid out on blue paper": "透明胶带和剪刀摆在蓝色纸上",
"spoon on the bowl.": "碗里的勺子。",
"green street signs with white lettering": "带白色文字的绿色街牌",
"character of person on the sign": "标牌上的人物图案",
"Side by side beds displayed in brightly lit room.": "明亮房间里并排展示的床。",
"black teddy bear placed in front of mirror": "放在镜子前的黑色泰迪熊",
"large evergreen tree": "高大的常绿树",
"Two elephants standing on a grassy field next to a tree.": "两头大象站在草地上的树旁。",
"two brown horses running on the sand": "两匹棕色的马在沙地上奔跑",
"red cans on the counter": "柜台上的红色罐子",
"black and white shirt": "黑白相间的衬衫",
"A little boy sleeping on a couch holding a Wii controller.": "一个小男孩躺在沙发上，手里拿着 Wii 手柄。",
"An 023 airplane flying solo above blue terrain.": "一架 023 飞机独自飞过蓝色地形。",
"rock on near the water": "水边的岩石",
"black writing on the sign": "标牌上的黑色文字",
"trees and mountains behind the house": "房子后面的树木和山脉",
"a body of blue water": "一片蓝色水域",
"two zebras in a field near tall grass": "高草附近田野里的两只斑马",
"a girl with a blue shirt": "穿蓝色衬衫的女孩",
"the cluttered bed the girl is sitting on": "女孩坐着的凌乱床铺",
"A baby giraffe standing next to a wire fence.": "一只长颈鹿幼崽站在铁丝网旁。",
"A man opening the oven with food in one hand.": "一个男人一手拿着食物打开烤箱。",
"Two teddy bears position to look like they are playing instruments.": "两只泰迪熊被摆成演奏乐器的样子。",
"magazine on coffee table": "咖啡桌上的杂志",
"A four door pickup truck parked on a road with hills in the background.": "一辆四门皮卡车停在路上，背景是山丘。",
"man sitting on a baseball field": "坐在棒球场里的男子",
"the sightseeing bus is red.": "观光巴士是红色的。",
"brown chair next to bed": "床边的棕色椅子",
"donuts on a covered plate in a commercial kitchen": "商用厨房里盖着盖子的盘子上的甜甜圈",
"sign on a pole": "杆子上的标牌",
"Four new suitcases are kept one above the other.": "四个新行李箱一个叠一个地放置。",
"blue napkin on the plate.": "盘子上的蓝色餐巾。",
"bear sitting on the ground": "坐在地上的熊",
"people sitting on the beach.": "坐在海滩上的人们。",
"a man sitting on the snow": "一个人坐在雪地上",
"A woman with glasses drinking a glass of wine.": "戴眼镜的女人在喝一杯葡萄酒。",
"a blue luggage cart on wheels": "一辆带轮子的蓝色行李车",
"a black suitcase": "一个黑色行李箱",
"A cat standing on top of a car trunk next to a parked motorcycle.": "一只猫站在汽车后备箱上，旁边停着一辆摩托车。",
"there's a white dog sitting on the sidewalk.": "有一只白狗坐在人行道上。",
"bird sitting on a table": "坐在桌子上的鸟",
"the guy is sitting on the curb": "那个男人坐在路边石上",
"A group of three giraffe standing inside of a cage.": "三只长颈鹿一起站在笼子里。",
"a woman sitting on beach": "坐在海滩上的女人",
"two gray white and brown birds a knife and a red table": "两只灰白色和棕色的鸟、一把刀和一张红色的桌子",
"a bunch of people at a table with food": "一群人在摆着食物的桌旁",
"the shirt is green": "这件衬衫是绿色的",
"boy sitting on the grass": "坐在草地上的男孩",
"wood fence behind the giraffe": "长颈鹿身后的木栅栏",
"person running on the beach": "在海滩上奔跑的人",
"blue baseball cap on man sitting on bleachers": "坐在看台上的男子戴着蓝色棒球帽",
"a man sitting on the ground": "一个人坐在地上",
"a black batter's helmet": "一顶黑色击球手头盔",
"Two birds sitting on the the back of a giraffe.": "两只鸟坐在长颈鹿的背上。",
"rocks on water bank": "水岸边的岩石",
"woman standing behind a chain-link fence": "站在铁丝网后面的女人",
"building behind thin trees": "细树后面的建筑",
"A man pointing at something by people with glasses and wine bottles.": "一个男人指着某样东西，旁边是戴眼镜的人和酒瓶。",
"blue bleachers behind wall": "墙后面的蓝色看台",
"four boats with people  carrying lots of bananas and other foods": "四艘载着许多香蕉和其他食物的船",
"A man sitting in the snow riding a snowboard.": "一个坐在雪地里骑滑雪板的男人。",
"bird sitting on branch": "坐在树枝上的鸟",
"Two horses stand in the desert while small desert hills stand in the background.": "两匹马站在沙漠中，背景是小沙丘。",
"Three elephants kick up dust as they walk through the flat by the bushes.": "三头大象在灌木丛旁的平地上行走时扬起灰尘。",
"several horses are standing together on a beach": "几匹马一起站在海滩上",
"the clock is located high on the tower": "时钟位于塔楼高处",
"black metal pole in ground": "插在地上的黑色金属杆",
"white house behind truck": "卡车后面的白房子",
"Three beds in a white bedroom with two windows.": "一间有两个窗户的白色卧室里有三张床。",
"Two monitors sit on an organized desk area": "两台显示器放在整洁的桌面上",
"woman sitting on a boat": "坐在船上的女人",
"a black tennis racket in the girl's hand": "女孩手里拿着黑色网球拍",
"man sitting on bicycle": "骑在自行车上的男子",
"horses sitting on the sand": "坐在沙地上的马",
"boy running on the sand": "在沙地上奔跑的男孩",
"a person sitting on the chair": "坐在椅子上的人",
"Half a dozen donuts from Krispy Kreme of various different flavors.": "半打多种口味的 Krispy Kreme 甜甜圈。",
"A man carrying a surfboard on top of a wave covered beach.": "一个男人在波浪覆盖的海滩上扛着冲浪板。",
"Two beds are in a hotel room type setting.": "类似酒店房间的环境里有两张床。",
"Apple computer monitor with a keyboard and mouse next to a ipod": "苹果电脑显示器，旁边是键盘、鼠标和 iPod",
"a brown cooked piece of meat": "一块棕色熟肉",
"person sitting on side of mountain": "坐在山边的人",
"Three suitcases on the side of a street.": "街边的三个行李箱。",
"a bush is behind the rocks": "岩石后面有灌木",
"Two sugar doughnuts with sprinkles and caramel topping.": "两个撒有糖屑和焦糖糖霜的糖霜甜甜圈。",
"horses running on grass racetrack": "在草地赛马道上奔跑的马",
"A group of people observing two planes at an air show.": "一群人在航空展上观看两架飞机。",
"a bush behind the fence": "栅栏后面的灌木",
"dog sitting on rocky beach": "坐在多石海滩上的狗",
"man wearing black shoes": "穿黑色鞋子的男子",
"shadow on the bench": "长椅上的阴影",
"the dog has a green frisbee behind it's head": "狗头后面有一个绿色飞盘",
"the red and white tail of the plane": "飞机红白相间的尾部",
"a black and white checkered shirt": "一件黑白格子衬衫",
"THERE ARE TWO ZEBRAS THAT ARE WALKING TOGETHER": "有两只斑马正一起行走",
"baseball player running on baseball field": "棒球运动员在棒球场上奔跑",
"paper is on the tray": "托盘上的纸",
"the woman is sitting on a surfboard.": "那个女人正坐在冲浪板上。",
"an electrical outlet on the mirror": "镜子上的电源插座",
"large group of people sitting on grass": "一大群人坐在草地上",
"A black and white photo of two birds walking in a grassy field.": "一张黑白照片，两只鸟走在草地上。",
"Bikes and cars moving in their respected lanes in a road.": "自行车和汽车在道路上各自的车道内行驶。",
"a silver fork": "一把银色叉子",
"A street that has three cows walking down the street together and people on the sidewalk area.": "一条街道上有三头牛一起走在街上，人行道区域有人。",
"the truck has a barrel in the back of it": "卡车后部有一个桶",
"Three sinks and a mirror in a public restroom.": "公共洗手间里有三个洗手池和一面镜子。",
"Cars stopped at an intersection on a dark night": "黑夜中停在十字路口的汽车",
"brown leaves in the green grass": "绿草中的棕色树叶",
"back wheels on a skateboard": "滑板的后轮",
"A man riding on the back of a brown horse.": "一个人骑在棕马背上。",
"A picture of two giraffes, fairly close to a road, with a bus traveling up it.": "一张两只长颈鹿的照片，它们离道路很近，一辆公交车正驶过。",
"the floor is brown.": "地板是棕色的。",
"farthest building behind closest palm tree": "最远处的建筑在最近的棕榈树后面",
"A person laying on a bathtub with their feet sticking out.": "一个人躺在浴缸里，双脚伸出来。",
"Two mini elephants walking  toward people on a plank": "两头迷你大象沿着木板走向人群",
"white clock on wall": "墙上的白色时钟",
"gray chair behind table": "桌子后面的灰色椅子",
"Three giraffes are sitting on the ground.": "三只长颈鹿坐在地上。",
"People sitting down during a tour boat excursion.": "人们在游船观光时坐下。",
"the top of a blue building": "蓝色建筑的顶部",
"a big red cargo van": "一辆大红色货运面包车",
"sky-blue water": "天蓝色的水",
"there are 6 red lights behind the train": "火车后面有 6 个红灯",
"a boy is laying on a large ball": "一个男孩躺在一张大球上",
"a table behind sofa.": "沙发后面的桌子。",
"a white and green traffic sign": "一个白色和绿色的交通标志",
"person sitting on platform": "坐在站台上的人",
"Two giraffes and another animal in a field.": "田野里的两只长颈鹿和另一只动物。",
"A man in a blue outfit holding a white frisbee.": "一个穿蓝色服装的男人拿着白色飞盘。",
"Three stuffed teddy bears are set on a toy picnic scene.": "三只填充泰迪熊被摆放在玩具野餐场景中。",
"a paw on a bear": "熊的爪子",
"A person showing off the inside of two doughnuts": "一个人展示两个甜甜圈的内部",
"Two dogs laying on a bed with a red comforter.": "两只狗躺在铺有红色床罩的床上。",
"A metal bucket with a bottle of booze sticking out of it.": "一个插着酒瓶的金属桶。",
"black purse with a cat on top.": "上面有一只猫的黑色手提包。",
"Two blurry zebras walking in front of a stone like wall.": "两只模糊的斑马走在石墙前。",
"an orange colored round cracker with brown rice on top.": "一块橙色圆形饼干，上面有糙米。",
"wheels on bottom of plane": "飞机底部的轮子",
"Person riding a bicycle while walking two dogs.": "一个人骑自行车的同时牵着两只狗。",
"A red car parked in front of a parking meter on a street.": "一辆红色汽车停在街道的停车计时器前。",
"Kitchen oven set up with burning fire inside of it.": "厨房烤箱里生着火。",
"Asian man lays back on a black couch holding a Wii controller.": "一名亚洲男子躺在黑色沙发上，手里拿着 Wii 手柄。",
"man sitting on ground wearing sunglasses": "坐在地上戴太阳镜的男子",
"person sitting on the boat": "坐在船上的人",
"a small child sitting on a bench": "坐在长椅上的小孩",
"A four faced clock a top a stone column in  a parking area.": "停车区域石柱顶端的四面时钟。",
"A room with four beds placed in various positions.": "一个房间里四张床摆在不同位置。",
"An assortment of skis and snowboards resting on a ski slope.": "滑雪坡上放置的一批滑雪板和滑雪单板。",
"a large tan building": "一座大型黄褐色建筑",
"A city street with many different cars driving.": "城市街道上有许多不同的汽车在行驶。",
"A majolica vase decorated with a fish and filled with wild flowers.": "一个饰有鱼纹并插满野花的彩釉陶瓶。",
"A girl flies a kite in the air outside.": "一个女孩在外面放风筝。",
"black spot on the crust": "饼皮上的黑点",
"there are three vases made of clay on a table": "桌上有三个陶土花瓶",
"A passenger train on the tracks going under a bridge.": "一列客运火车在桥下的铁轨上行驶。",
"the people are sitting on the grass": "人们坐在草地上",
"pepsi can on sink": "水槽上的百事可乐罐",
"a drawer on a counter": "柜台上的抽屉",
"large silver door": "大型银色门",
"A kitchen counter with a pizza ready for the oven.": "厨房台面上有一份准备放进烤箱的披萨。",
"there are three oranges next to  a paper towel on the counter": "台面上有三个橙子，旁边是纸巾",
"A group of four sheep on a hill.": "山坡上有一群四只羊。",
"eye of giraffe is black": "长颈鹿的眼睛是黑色的",
"steep hill with green trees on it.": "长满绿树的陡坡。",
"orange square carrot up highest on the white fork.": "白色叉子上最高的橙色方形胡萝卜。",
"Two elephants that are standing up in the street.": "两头大象站在街上。",
"Three zebra in the middle of a field with a body of water in the distance.": "田野中间有三只斑马，远处有一片水域。",
"a orange tag on a robot": "机器人上有一个橙色标签",
"Three teddy bears and a stuffed cat on a stone wall.": "石墙上有三只泰迪熊和一只填充猫。",
"a person sitting on the steps": "坐在台阶上的人",
"a couple of elephants are walking in a line": "几头大象排成一列行走",
"green foliage on a tree": "树上的绿色枝叶",
"A train is near a track switching station.": "一列火车靠近铁路道岔站。",
"A picture of some pizza on a table.": "桌上披萨的照片。",
"three zebras sitting on the ground": "三只斑马坐在地上",
"a motorcycle is hanging in the air with plants on it": "一辆摩托车悬在空中，上面有植物",
"a person running on the beach": "在海滩上奔跑的人",
"Three horses on the sand at the ocean.": "海边沙滩上的三匹马。",
"the wall behind the fence has mural paintings": "栅栏后面的墙上有壁画",
"wheels on a skateboard": "滑板上的轮子",
"a man sitting on a boat.": "一个坐在船上的男人。",
"light shines on table": "光线照在桌子上",
"a black dog sitting on a tile floor": "一只黑狗坐在瓷砖地板上",
"giraffes sitting on the ground": "长颈鹿坐在地上",
"man sitting on surfboard": "坐在冲浪板上的男子",
}

items = []
missing = []
with SRC.open("r", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        en = r["caption"].strip()
        zh = ZH.get(en)
        if zh is None:
            missing.append(en)
            zh = en
        p = Path(r["image_path"].strip())
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80, optimize=True, progressive=True)
            uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            missing.append(f"IMAGE FAILED: {p} {e}")
            uri = ""
        items.append({"id": r["pair_id"].strip(), "img": uri,
                      "cap_en": en, "cap_zh": html.escape(zh)})

if missing:
    print("MISSING/FAILED:", len(missing))
    for x in missing[:20]:
        print(" ", x)

data = json.dumps(items, ensure_ascii=False, separators=(",", ":"))

page = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FGBench 盲标 · 中文版</title>
<style>
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#111;color:#eee}
.wrap{max-width:1150px;margin:0 auto;padding:18px}
h1{font-size:22px;margin:0 0 8px}
.card{background:#1c1c1c;border:1px solid #333;border-radius:12px;padding:16px;margin-top:14px}
.grid{display:grid;grid-template-columns:minmax(380px,1fr) minmax(300px,.8fr);gap:18px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
img{max-width:100%;max-height:62vh;object-fit:contain;border-radius:8px;background:#000;min-height:280px}
.cap{font-size:23px;line-height:1.55;background:#262626;border-radius:10px;padding:18px;margin:0}
.orig{font-size:14px;color:#bbb;margin-top:8px;line-height:1.35}
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
input{font-size:16px;padding:10px;border-radius:8px;border:1px solid #444;background:#262626;color:#eee;width:100%}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.stat{background:#262626;border-radius:10px;padding:10px;margin-top:10px}
</style>
</head>
<body>
<div class="wrap">
  <h1>FGBench 盲标</h1>
  <div class="small">只判断：这张图对应的中文说明是否正确描述图像。不要打开 key.csv。</div>

  <div id="start" class="card">
    <label>标注者姓名 / 缩写</label>
    <input id="name" placeholder="例如：A、B、CJ" autocomplete="off">
    <div class="row" style="margin-top:12px">
      <button class="export" onclick="start()">开始 / 继续标注</button>
      <span class="small">进度会按姓名保存在本地浏览器里。</span>
    </div>
  </div>

  <div id="work" style="display:none">
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <span class="badge" id="pos">第 1 / 400 条</span>
        <span class="badge" id="annot">—</span>
        <span class="badge" id="left">未完成：0</span>
      </div>
      <div id="bar"><div id="fill"></div></div>
      <div class="grid">
        <div><img id="im" alt="图片"></div>
        <div>
          <p class="cap" id="cap"></p>
          <div class="orig" id="orig"></div>
          <div class="row">
            <button class="clean" onclick="mark(1)">正确 · 1</button>
            <button class="noisy" onclick="mark(0)">错误 · 2</button>
            <button class="unsure" onclick="mark(-1)">不确定 · 3</button>
          </div>
          <div class="row" style="margin-top:10px">
            <button class="nav" onclick="prev()">◀ 上一条</button>
            <button class="nav" onclick="next()">下一条 ▶</button>
            <button class="nav" onclick="nextUnanswered()">下一个未标注</button>
          </div>
          <div class="row" style="margin-top:12px">
            <button class="export" onclick="exp()">导出 CSV</button>
            <span class="small">导出文件：<span id="fname"></span></span>
          </div>
          <div id="msg"></div>
        </div>
      </div>
      <div class="stat small">
        快捷键：<kbd>1</kbd> 正确 · <kbd>2</kbd> 错误 · <kbd>3</kbd> 不确定 ·
        <kbd>←</kbd> 上一条 · <kbd>→</kbd> 下一条 · <kbd>U</kbd> 下一个未标注
      </div>
    </div>
  </div>
</div>

<script>
const P = __DATA__;
let name = "", A = {}, i = 0, K = "";

function start(){
  name = document.getElementById("name").value.trim() || "annotator";
  K = "fgb_annot_cn_" + name + "_" + P.length;
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
  document.getElementById("cap").textContent=p.cap_zh;
  document.getElementById("orig").textContent="原文：" + p.cap_en;
  document.getElementById("pos").textContent=`第 ${i+1} / ${P.length} 条`;
  const n=Object.keys(A).length;
  document.getElementById("fill").style.width=(n/P.length*100)+"%";
  document.getElementById("left").textContent="未完成："+(P.length-n);
  document.getElementById("msg").textContent = (p.id in A) ? "已标注：" + (A[p.id]===1?"正确":A[p.id]===0?"错误":"不确定") : "";
}
function mark(v){ A[P[i].id]=v; save();
  if(i<P.length-1){ i++; } else { msg("已到最后一条，请导出 CSV。"); }
  show();
}
function prev(){ if(i>0){ i--; show(); } }
function next(){ if(i<P.length-1){ i++; show(); } }
function nextUnanswered(){
  for(let k=1;k<=P.length;k++){ const j=(i+k)%P.length; if(!(P[j].id in A)){ i=j; show(); return; } }
  msg("全部已标注。");
}
function msg(s){ document.getElementById("msg").textContent=s; }
function exp(){
  const answered=P.filter(p=>p.id in A);
  if(answered.length===0){ msg("没有可导出的标注。"); return; }
  let s="pair_id,label\n";
  for(const p of P){ if(p.id in A) s+=p.id+","+A[p.id]+"\n"; }
  const b=new Blob([s],{type:"text/csv"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(b);
  a.download="annotator_"+name+".csv"; a.click();
  msg(`已导出 ${answered.length}/${P.length} 条标注。`);
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
OUT.write_text(page.replace("__DATA__", data), encoding="utf-8")
print("wrote", OUT, "items", len(items), "size_MB", round(OUT.stat().st_size/1024/1024, 1))
