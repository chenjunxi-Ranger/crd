# 图文错标检测项目 · 交接文档

> 最后更新：2026-11-06
> 读者：接手这个课题的同学
> **这份文档不依赖任何一台具体电脑的路径。** 它说明：这个项目在做什么、做到哪一步、数据从哪来、代码跑起来需要改哪几行、以及下一步该干什么。

---

## 0. 一句话说清项目

在一个**冻结的视觉语言模型**（CLIP 或 FG-CLIP）之上，用**它的图文相似度评分**，判断一个图文对是否"错配"——即 caption 不属于这张图（来自别的图），或 caption 的内容与图不符。

四个评测集：**COCO、Flickr30k、MM-IMDb、FOIL-COCO**。

---

## 1. 开工前你要准备的三样东西

### 1.1 原始数据集（全部是公开数据集，需自行下载）

| 名称 | 用途 | 你需要拿到什么 |
|---|---|---|
| **MS-COCO 2014** | COCO 行级任务 | `train2014/`、`val2014/` 图片；`captions_train2014.json`、`captions_val2014.json`；`instances_*.json`（类别标注，用于构造 donor） |
| **Flickr30k** | Flickr 行级任务 | 图片（`flickr30k-images/`）+ 每图 5 条 caption（`results_20130124.token` 或同源 CSV） |
| **MM-IMDb** | 电影海报任务 | 海报图片 + 剧情简介（`plot`）+ 电影 `genre` 多标签 + `imdbId` |
| **FOIL-COCO** | 同图反事实二选一 | `pairs_train.csv` / `pairs_test.csv`：每行 `path, positive_caption, negative_caption`（同一张图，一句真一句假） |

> FOIL 也可以自己构造：取同一条 caption，替换其中一个名词/属性生成反事实版本。但**若要和本项目的数字对齐，必须用官方 FOIL-COCO 的划分**。

### 1.2 模型权重（公开，可用 HuggingFace 直接下）

| 代号 | HuggingFace 仓库 | 约大小 | 说明 |
|---|---|---|---|
| `clip-vit-base-patch32` | `openai/clip-vit-base-patch32` | 600 MB | 主基座 |
| `clip-vit-base-patch16` | `openai/clip-vit-base-patch16` | 600 MB | 对照基座 |
| `clip-vit-large-patch14` | `openai/clip-vit-large-patch14` | 1.71 GB | 对照基座（最强） |
| `fg-clip-base` | FG-CLIP 官方仓库（需 `trust_remote_code`） | 600 MB | 细粒度基座，带自定义 `modeling_fgclip.py` |

**下载注意事项（都是踩过的坑）：**

1. `huggingface_hub.snapshot_download` 在部分网络下会报 `LocalEntryNotFoundError`，但**直链 `https://huggingface.co/<repo>/resolve/main/<file>` 是可用的**。仓库里有 `scripts/fetch_clip.py` 就是干这个的，支持断点续传。
2. 必须下 **`model.safetensors`**。`.bin` 在 torch < 2.6 上会被 transformers 拒绝加载（报 `require torch>=2.6`）。若某仓库只有 `.bin`，可用 `torch.load(..., weights_only=True)` 读进来再 `safetensors.torch.save_file` 转一份。
3. 每个 CLIP 目录至少要包含：`config.json`、`preprocessor_config.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`、`special_tokens_map.json`、`model.safetensors`。

### 1.3 Python 环境

本项目验证过的组合：**Python 3.10 / torch 2.5.1+cu121 / transformers 4.53 / numpy 1.26 / pandas 2.2 / scikit-learn 1.5 / safetensors**。

显存：8 GB 级显卡够跑 ViT-B/32 与 ViT-B/16 的 LoRA 微调；**ViT-L/14 必须把 batch 降到 8**，否则 OOM。

**重要**：一张卡上**不要并行跑两个 GPU 进程**，会互相抢显存导致整体慢十倍以上，务必串行。

---

## 2. 目录布局：代码假设的固定结构

代码里 `src/protocol.py` 顶部写死了几个相对位置，**它假设你的项目根目录（记作 `<ROOT>`）长这样**：

```
<ROOT>/
├── mymodel7/                      ← 本项目代码（本仓库）
│   ├── src/
│   ├── scripts/
│   ├── results/
│   ├── data/                      ← 小文件（FOIL val 清单等）
│   └── HANDOVER.md                ← 本文档
├── <weights_dir>/                 ← 模型权重（名字随意，改常量即可）
│   ├── clip-vit-base-patch32/
│   ├── clip-vit-base-patch16/
│   ├── clip-vit-large-patch14/
│   └── fg-clip-base/
└── <data_dir>/                    ← 派生数据（由原始数据生成）
    └── derived/
        ├── mscoco/
        ├── flickr30k/
        └── mmimdb/
```

### 你要改的 5 个常量（`src/protocol.py` 第 18–34 行附近）

```python
PROJECT   = Path(__file__).resolve().parents[2]   # <ROOT>
DATA_DIR  = MYMODEL7 / "data"
RESULTS_DIR = MYMODEL7 / "results"
CLIP_DIR  = PROJECT / "<weights_dir>" / "clip-vit-base-patch32"   # ← 改成你的权重目录
DERIVED   = PROJECT / "<data_dir>" / "derived"                    # ← 改成你的派生数据目录
BENCH_CSV = {...}       # 三个派生 CSV 的路径
FOIL_VAL  = DATA_DIR / "foil_val_10k.csv"
FOIL_TEST = PROJECT / "<data_dir>" / "FOIL-COCO" / "pairs_test.csv"
THRESHOLDS = {"coco":0.960, "flickr":0.960, "mmimdb":0.880, "foil":0.920}  # 旧门槛，仅留档
```

改完这几行，**所有脚本都不需要再改路径**（其余脚本都从这里 import）。
若要用别的骨干，训练脚本支持 `--model-dir <你的某个权重目录>`，不需要改常量。

---

## 3. 派生数据：本项目自己构造的"错配版"数据集

这是全项目的基石。原始 COCO/Flickr 是全干净的；**噪声是本项目按规则注入的**，不是数据集自带的。

### 3.1 文件命名规则

```
{数据集}__{噪声类型}__lvl{噪声比例}__seed{种子}.csv
例：mscoco__cat__lvl0.40__seed0.csv
    flickr30k__noun__lvl0.40__seed0.csv
    mmimdb__cat__lvl0.40__seed0.csv
```

- **噪声类型**：`cat`＝按类别相似选 donor，`noun`＝按 caption 共享名词选 donor，`random`＝随机 donor；
- **lvl0.40**＝40% 的样本被判为噪声；
- **seed0/1/2**＝三次独立抽样（用于检验构造稳健性）。

**主评测统一用 `seed0`**（`protocol.py` 里配的是 seed0）。注意：捐体（donor）相关的诊断脚本历史上用的是 `seed1`，两者是**不同的噪声实现**，数字不可混比——这是已知的一致性问题，见 §12。

### 3.2 每一行的字段与含义

| 字段 | 含义 |
|---|---|
| `image_id` | 目标图 id |
| `filename` / `path` | 目标图文件路径 |
| `gold_sentence` | 这张图**本来**的 caption（干净版） |
| `sentence` | **实际给模型的 caption**（干净样本＝gold；噪声样本＝donor 的句子） |
| `is_mislabel` | 1＝噪声（来源错配），0＝干净 |
| `donor_image_id` / `donor_filename` | 噪声样本的 caption 来自哪张图 |
| `donor_sentence` | 被移植过来的那句话（工程上等于 `sentence`） |
| `split` | `train` / `val` / `test` |
| `cat_labels`（仅 MM-IMDb） | 电影类型多标签，list 字符串 |

### 3.3 构造规则（拿不到原始生成脚本时，按这个重建）

**COCO / Flickr（跨图 donor）**

1. 对每个 split，按比例 40% 选样本作为噪声；
2. 为目标图选一张 donor 图：
   - `cat`：与目标图有相同 COCO 类别的其他图；
   - `noun`：caption 与目标图 caption 至少共享一个内容名词的其他图；
   - `random`：随机其他图；
3. **donor 必须 ≠ 目标图**（本项目实测同图率 0%）；
4. 用 donor 的一条 caption 替换目标的 caption，写入 `sentence`，并记录 `donor_*`。

**MM-IMDb（跨电影 donor）**

1. 选另一部**至少共享一个 genre** 的电影作为 donor（本项目实测：100% 至少共享一个类型，8.2% 类型完全相同）；
2. 用该电影的 plot summary 替换本片 plot；
3. 记录电影 id 与 genre。

**FOIL**：官方提供同图 true/foil 对，直接用，不做改造。

### 3.4 重建后必须校验的不变量

| 检查 | 期望值 |
|---|---|
| 噪声比例 | ≈0.40 |
| 同图 donor 比例 | 0（COCO/Flickr） |
| MM-IMDb 类型重叠率 | 100% |
| MM-IMDb `sentence == donor_sentence` | 100% |

任一条不符，说明构造和本项目不一致，**数字不可直接与本文档的表格比较**。

---

## 4. `data/foil_val_10k.csv` 是什么

FOIL 的 validation 清单：10,000 组同图对，列名 `path, positive, negative, image_id`（用于组内二选一准确率，以及图像组 bootstrap）。
它由 FOIL-COCO 官方 val 抽样得到，**只含清单不含图片**，图片路径指向你本地的 FOIL-COCO 目录。

若你没有这份清单，可用 `pairs_test.csv` 的前 10,000 行替代，注意保持 `image_id` 唯一性用于聚类 bootstrap。

---

## 5. 评测协议（改之前先读）

| 项 | 规定 |
|---|---|
| 行级指标 | **AUROC**（异常分数越大越可疑） |
| FOIL 指标 | **同图二选一准确率**（true 的异常分数应低于 foil），**不是** pooled AUROC，两者不可混算 |
| 置信区间 | 一律**按 `image_id` 聚类 bootstrap**（同图多 caption 不是独立样本） |
| 主门槛（C5，已修订） | 相对**同一基座余弦**的归一化改善 $G_d=\dfrac{M_d-B_d}{1-B_d}\ge 10\%$，且配对 CI 下界 > 0 |
| 禁止 | 用 test 反选门槛；只报点估计不报 CI |

> **口径变化会让结论翻转。** 旧口径要求绝对 AUROC ≥0.96/0.96/0.88/0.92，那时结论是"几乎全不达标"；改成相对改善后，同一批结果变成"3/4 通过"。所以引用数字时**必须注明口径**。

---

## 6. 当前状态（先看这一节判断值不值得接手）

### 6.1 已经做到

| 项 | 结果 |
|---|---|
| 行级最强（自训练） | **R200-LoRA：0.9723 / 0.9707 / 0.8886**（三种子均值） |
| 行级最强（换基座） | **FG-CLIP + CSLS：0.9804 / 0.9778 / 0.8909** |
| FOIL 最好 | **0.9415**（ViT-L/14 纯余弦） |
| 诊断工具 | TRACED 复现、配对循环、无 donor 恢复、人工盲标 310 条，全部完成 |

### 6.2 没做到（真正的开放问题）

1. **没有配置能同时满足四列**：COCO 需要 CSLS（+2.3 pp），FOIL 必须去掉 CSLS（−4.6 pp），同一骨干上互斥。
2. **FOIL 是唯一短板**：换任何训练目标都卡在 0.86–0.90。
3. **没有提出可称得上方法贡献的新机制**（十余个候选全部被否证，见 §8）。

---

## 7. 主线结论（这些是站稳的）

### 7.1 四个 benchmark 其实是三种任务

| 数据集 | 负样本构造 | 实际检测目标 | 局部视觉可验证性 |
|---|---|---|---|
| COCO / Flickr | 跨图 donor（相似图的高相似描述） | 来源错配 ∩ 语义错配 | 高 |
| FOIL | 同图反事实（true vs foil） | **纯语义矛盾**（极细粒度） | 高但极细 |
| MM-IMDb | 跨电影 donor，**100% 同类型** | **影片身份关联** | **低** |

这是全项目最重要的一条：MM-IMDb 考的是"这段简介是不是这部电影"，不是"海报上能不能看到剧情"。所以换基座只涨 2.25 点、局部 patch 证据是负增量。

### 7.2 密度项 $r_c$ 存在跨协议极性冲突

异常分数 $A=-s+\lambda(r_I+r_c)$，其中 $r_I$ 是图像侧库密度、$r_c$ 是文本侧库密度。

| 分数 | COCO | Flickr | MM-IMDb | FOIL(组内) |
|---|---:|---:|---:|---:|
| 余弦 | 0.9310 | 0.9534 | 0.8174 | **0.9274** |
| 仅图像侧 $r_I$ | 0.9566 | 0.9630 | 0.8334 | 0.9274（恒等抵消） |
| 仅文本侧 $r_c$ | 0.9544 | 0.9622 | 0.8639 | **0.8856** |
| 完整 CSLS | 0.9677 | 0.9669 | 0.8810 | 0.8856 |

- 行级：噪声的 $r_c$ **更高** → 加号有用；
- FOIL：**正确描述的 $r_c$ 更高**（mean Δr_c = +0.017，70% 样本如此）→ 加号等于惩罚正确描述；
- **任何对 $r_c$ 的单调变换都救不了**：增函数保留错误极性，减函数会把行级翻过来。

### 7.3 FOIL 的损失是"边际覆盖"

翻转（余弦对、CSLS 错）的充要条件是

$$0.8\,\Delta r_c > \Delta s$$

83% 的翻转集中在 $|\Delta s|<0.035$ 的最小边际档。不是"极性随边际改变"，是"小边际被覆盖"。

### 7.4 换骨干是唯一稳定有效的手段

| 骨干 | COCO 余弦 | COCO+CSLS | FOIL 余弦 | FOIL+CSLS |
|---|---:|---:|---:|---:|
| ViT-B/32 | 0.9390 | 0.9716 | 0.9025 | 0.8595 |
| ViT-B/16 | 0.9424 | 0.9718 | 0.9220 | 0.8825 |
| ViT-L/14 | **0.9533** | **0.9762** | **0.9415** | 0.8950 |
| FG-CLIP B/16 | 0.9642 | 0.9804 | 0.9131 | 0.8789 |

规律：**换大骨干抬高的是余弦本身**；CSLS 在 FOIL 上始终是 −4~−5 pp（三个骨干一致）。

---

## 8. 已证伪的路线（**不要重做**）

| 机制 | 结论 | 关键数字 |
|---|---|---|
| SCSA-B（把 CSLS 写进训练目标） | 无增益 | ΔFOIL = −0.0004 |
| Image-only（删掉 $r_c$） | MM-IMDb 崩 | −4.8 pp |
| $g_c$（top-k − 随机背景） | 与 $r_c$ 相关 0.86，方向未变 | — |
| 条件极性交互 $w_\phi(s)$ | 分层证据不支持共享门控 | — |
| BCAE（Gibbs 匹配边缘概率） | 相对 CSLS +0.33 / +0.03 pp | 跨基座不一致 |
| BCPR（双向条件残差） | 增量在 1e-4 级 | 与余弦同源 |
| CMPI（跨模态二阶交互 κ） | 五格全负 | −0.0004 ~ −0.0017 |
| LPR（训练阶段轨迹） | 增益来自初始检查点＝模型集成 | — |
| MBPI-Net（门控交互头 + 合成负样本） | 三列全低于 CSLS | 训练损失归零 |
| SEML-LoRA（跨环境元学习） | 参数几乎没动（空操作） | — |
| SCPA（相似度分位匹配） | 消融 Δ ≤ 0.0008 | 机制不成立 |
| ES-LoRA（按文本聚类分层采样） | COCO +0.04 / Flickr +0.03 / MM-IMDb +0.51 pp | 不是统一机制 |
| RC-LoRA（拉格朗日双关系约束） | 行级 0.9628 < 0.965 | 差 0.2 pp |
| FMP-LoRA（函数空间投影） | **约束一次都没触发**，结果与对照逐位相同 | margin 只变 −6e-6 |
| R2R-LoRA（关系回放） | 与对照逐位相同（结论待定，见 §12 缺陷 2） | — |

> **FMP 的教训（重要）**：它想解决"学 FOIL 会破坏来源能力"的冲突，但实测**这个冲突不存在**——从已训练模型出发只微调 FOIL 200 步，来源 margin 只掉 6e-6。
> 所以：**任何"保护/约束"类设计，先测被保护的量到底会不会变。**

---

## 9. 当前最强数字（对比、写论文都用这些）

| 配置 | COCO | Flickr | MM-IMDb | FOIL |
|---|---:|---:|---:|---:|
| 冻结 CLIP 余弦 | 0.9287 | 0.9505 | 0.8340 | 0.8997 |
| CLIP + CSLS | 0.9678 | 0.9698 | 0.8838 | 0.8789 |
| **R200-LoRA + CSLS** | **0.9723** | **0.9707** | **0.8886** | 0.8723 |
| FG-CLIP 余弦 | 0.9642 | 0.9768 | 0.8565 | 0.9131 |
| **FG-CLIP + CSLS(0.8)** | **0.9804** | **0.9778** | **0.8909** | 0.8789 |
| FG-CLIP + 每数据集学到的 $(\lambda_I,\lambda_c)$ | 0.9807 | 0.9801 | 0.8926 | 0.9146 |
| ViT-L/14 余弦 | 0.9533 | 未测 | 未测 | **0.9415** |

**R200-LoRA 的配方**（要复现就照这个）：LoRA 注入最后两个 transformer block 的 Q/V，`r=4, α=8`；训练 200 步、batch 24、AdamW(1e-3)；损失 = clean/noisy 排序 `softplus(0.1 - s_clean + s_noisy)` + 权重 5.0 的教师锚束；**不用 FOIL 反事实负样本、不用 in-batch NCE**。

---

## 10. 未解问题（新同学从这里选一个）

### Q1（最重要）：FOIL 上 CSLS 为什么必然负贡献？

已定位到 $r_c$ 的极性冲突，但**没找到能同时服务行级和 FOIL 的替代统计量**。
可试：把 $r_c$ 换成"与该图其他候选 caption 的相对位置"而非"与全库的密度"——但这已接近 LEMoN 的领域。

### Q2：FOIL 微调为什么只能涨 0.7–0.8 pp？

三种训练方式（仅 FOIL / 加保护 / 双关系回放）都停在 0.9105（余弦）。
**先排除实现问题**：这些微调用的是手写 SGD(η=1e-4)，而 R200 用 AdamW(1e-3)，步长差一个数量级。
**第一件该做的事**：把微调优化器换成 AdamW 重跑，确认"三种机制结果逐位相同"是不是假象。

### Q3：MM-IMDb 的影片身份能否从海报恢复？

donor 100% 同类型，视觉线索被中和。突破口可能是引入电影元数据——但这会破坏"基座无关"的定位。

### Q4：跨基座迁移是否成立？

诊断显示配对信息确实存在于表示中（见 §11），但**没有方法验证过"在 CLIP 上学到的检测器能否零样本用于 FG-CLIP"**。

---

## 11. 诊断型实验的证据（不是方法，是论据）

| 实验 | 做法 | 结论 |
|---|---|---|
| **TRACED 复现** | 按论文机制实现"贪心删词轨迹 + 轨迹特征 + 轻量分类器" | 总体 +2.7~+3.5 pp，但 **62–95% 的增益来自 easy+/easy− 这一格**；hard+/hard− 权重仅 0.4–0.6% |
| **配对循环** | 用真实 donor 构造 2×2：$\Delta_{cycle}=s(I_a,c_a)+s(I_b,c_b)-s(I_a,c_b)-s(I_b,c_a)$ | CycleAcc **98.6%（COCO）/ 99.4%（Flickr）/ 99.8%（FG-CLIP）**，困难组仍有 91.9%–100% → **表示里确实有来源配对信息**，只是单行分数用不上 |
| **无 donor 恢复** | 只给 caption，在 5,000 张无标签库里找回 donor 图 | Recall@1 **30%**（CLIP，随机水平 0.02%）、63.7%（Flickr/1014 库）、45%（FG-CLIP） |
| **人工盲标 310 条** | 单人盲标 match/mismatch，比较"持续困难组"与"低负担对照组" | **无富集**：率比 0.94 [0.56,1.56]（COCO）、1.16 [0.63,2.44]（Flickr） |
| **任务构造审计** | 读派生数据统计 donor 关系 | 四 benchmark = 三种任务；MM-IMDb donor 100% 同类型 |

---

## 12. 已知缺陷与待修（接手第一周该处理的）

1. **输出文件名不含骨干标识** → 换骨干跑会覆盖旧结果（本项目已发生过一次）。跑不同骨干务必加 `--tag`。
2. **微调优化器不一致**：R200 用 AdamW(1e-3)，而 `fmp_lora.py` / `rc_lora.py` 的微调用手写 SGD(1e-4)。这直接导致 M1/M2/R2R 结果**逐位相同**，那组对比目前**不可作为机制结论**。
3. **派生数据 seed 混用**：主评测用 `seed0`，donor 诊断脚本历史上用 `seed1`。两者是不同的噪声实现，跨文档引用数字前要确认来源。
4. **人工标注是单人**，没有 intra-rater κ（重测清单已抽好 50 条，但**尚未执行**）。
5. **FOIL 子集与全量不可混用**：2,000 对子集与 10,000 对全量会差 1–2 个百分点。

---

## 13. 代码地图

### 核心库（`src/`）

| 文件 | 作用 |
|---|---|
| `protocol.py` | 路径常量、`load_split(bench, split, limit)`、`load_foil_val(n)`、JSON 写出 |
| `encoders.py` | `JointEncoder(model_dir)`：图像/文本批量编码 |
| `scorer.py` | **统一打分接口** `score(image, caption) -> float`（带调用计数），两个基座同一契约 |

### 训练类（`scripts/`）

| 脚本 | 作用 | 主要参数 |
|---|---|---|
| `lora_stage1.py` | LoRA 定义与注入（`LoRALinear`、`inject`） | — |
| `seml_lora.py` | **R200 配方**、按文本聚类采样、换骨干、保存 adapter | `--bench --model-dir --save-adapter --no-eval --sample env|global` |
| `rc_lora.py` | 双关系 + 拉格朗日约束（含 `--calib` 标定 τ） | `--mode src|fixed|dual` |
| `fmp_lora.py` | 函数空间投影 / 关系回放 | `--mode foil\|fmp\|r2r --teacher --eps --tol` |

### 评分与诊断（`scripts/`）

| 脚本 | 作用 |
|---|---|
| `component_audit.py` | 四分量（cos / $r_I$ / $r_c$ / full）审计 |
| `foil_decomp.py` | FOIL 四分量与组内分解 |
| `foil_signed_audit.py` | 有符号 $\Delta r_c$ 归因（极性证据） |
| `conditional_polarity.py` | 条件极性分层审计 |
| `ranking_burden.py` / `burden_stable.py` | 样本级排序负担、四组划分、盲标材料导出 |
| `operating_point_audit.py` | 固定 FPR 下 TPR 对比 |
| `traced_repro.py` / `traced_failure.py` | TRACED 复现与失效图谱 |
| `cycle_diag.py` | 真实 donor 配对循环 |
| `sbcra.py` | 无 donor 来源恢复 |
| `backbone_full.py` / `fg_csls.py` | FG-CLIP 全量评测与 CSLS |
| `task_audit.py` | 四个 benchmark 的构造审计 |
| `fetch_clip.py` | 直链下载 HF 权重（支持续传） |

### 人工标注（`scripts/` + `results/audit/`）

| 文件 | 作用 |
|---|---|
| `make_formal_batch.py` | 抽样、打乱、生成盲标表与私有映射表 |
| `annotation_enrichment.py` | 分层加权富集 + 图像组 bootstrap |
| `*_blind.csv` | 盲标材料（只有 id / 图片 / 描述） |
| `*_key.csv` | **私有映射，含分组与标签，不得外发** |

### 约束文件

`CONSTRAINTS.md` —— **动手前必读**。里面是 C1–C8（方法层）与 M1–M5（定位层），历次修订的原因与日期都在。

---

## 14. 接手 checklist

**Day 1：让一切能跑**

```bash
# 1) 数据能读（应打印 5 行）
python -c "import sys;sys.path.insert(0,'.');from src.protocol import load_split;print(load_split('coco','val',5))"

# 2) 两个基座都能打分
python -c "import sys;sys.path.insert(0,'.');from src.scorer import build_scorer;from src.protocol import load_split;d=load_split('coco','val',4);[print(k,build_scorer(k).score_many(d.path.tolist(),d.caption.tolist())) for k in ('clip','fg')]"

# 3) 复现最强基线（约 6 分钟，期望 auc_csls ≈ 0.972）
python scripts/seml_lora.py --bench coco --mode erm --match none --sample env \
  --steps 200 --batch 24 --train 6000 --seed 0
```

**Day 2：读三份文档**
1. `analysis/task_audit_report.md` —— 四个数据集其实是三种任务；
2. `analysis/foil_decomp_report.md` —— FOIL 的损失 100% 来自 $r_c$；
3. `analysis/conditional_polarity_report.md` —— 极性冲突的证据。

**Day 3：复现 FOIL 现象**

```bash
python scripts/foil_decomp.py --limit 2000 --seed 1
# 期望：余弦约 0.927 > 完整 CSLS 约 0.886
```

**Day 4–5：选一个 Q 往下做**
优先 **Q2**：把微调优化器统一成 AdamW，重跑"仅 FOIL"与"关系回放"两组。
- 若仍完全相同 → FOIL 提升已饱和，转 Q1；
- 若关系回放明显更好 → 值得继续。

---

## 15. 三条经验（省你几个月）

1. **先确认前提，再设计机制。** FMP 花了整轮去"保护"一个根本不存在的损害（margin 只变 6e-6）。任何保护/约束类设计，先测被保护的量会不会变。
2. **换骨干是最便宜的有效手段，但它不是方法贡献。** FOIL 从 0.9025→0.9415 全靠 ViT-L/14，而 CSLS 在 FOIL 上始终 −4~−5 pp。
3. **不要在同一个排序损失上反复修补。** 已经试过：加密度项、换密度项、加约束、加回放、加投影、加元学习、加环境采样——**都没用**。要么换监督信号（FOIL 的监督密度不足），要么换接口（允许 batch 或中间层，代价是放弃"基座无关"）。

---

## 16. 产物索引

```
analysis/                     每个实验一份结论文档
  task_audit_report.md        四 benchmark = 三种任务（最重要）
  foil_decomp_report.md       FOIL 归因到 r_c
  conditional_polarity_report.md
  traced_failure_map.md       TRACED 失效图谱
  cycle_diag_report.md
  sbcra_report.md
  annotation_result_report.md
  mbpi_report.md / seml_report.md / scpa_report.md / eslora_report.md

results/
  bench/    跨脚本汇总指标（component_audit / task_audit / annotation_*）
  gate/     各机制的准入结果（json + 逐样本 npz）
  audit/    人工标注的盲标表、私有映射表、分组表
```

> 所有"关闭"决定都有对应的 json 与文档，**不要凭记忆推翻**，先读 `analysis/` 下对应文档。
