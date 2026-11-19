# CRD / FGBench 项目交接文档

> 更新时间：2026-11-19  
> 交接对象：接手代码、实验或论文投稿的同学  
> 一句话：这是一个在冻结 CLIP 上做**全局图文错标配对检测**的项目；核心方法是 CRD，核心数据是 FGBench，当前论文已压到英文 6 页。

---

## 0. 先看这里：10 分钟了解全貌

### 这个项目在做什么

给定一对 `(image, caption)`，判断它们是否**错配**（caption 不属于这张图，或内容与图不符）。

和普通“同一张图内选 caption”的任务不同，这里比较的是**两个互不相关的图文对**：

```text
pair A = (image A, caption A)
pair B = (image B, caption B)
```

模型只能看单对，不能看另一对，也不能依赖参考库。这就是论文里的 **global + pair-local** 设定。

### 核心方法

CRD 不训练 CLIP，只在冻结特征上拟合两个小型加性头：

```text
g(u, v) = a(u) + b(v) + b0
z' = z - λ g(u, v)
```

- `u`、`v` 是冻结 CLIP 的图像/文本特征；
- `z` 是训练集统计量标准化后的余弦分数；
- `g` 回归 `z` 中可从两侧特征预测的加性成分；
- `λ` 在验证集上选，标签只用于选这一个标量；
- 推理只需要单对，不需要 reference bank。

### 当前最重要的文件

```text
README.md                         项目入口
HANDOVER.md                       本文档
paper/CRD_ICME2027/main.pdf       英文 6 页投稿版
paper/CRD_ICME2027/cn/main_cn.pdf 中文对应版
results/fgbench/paper_ready/      论文主表/官方 baseline/joint 消融
fgbench/models/crd_backbone.py    CRD 主实现
fgbench/data/fgbench_v1_strict/   FGBench manifest
```

### 不要做的事

- 不要写“CRD 改善下游数据清洗/检索”：P1 是负结果；
- 不要写“human agreement / Cohen's κ”：P0-A 只标了 89/400，未完成；
- 不要把 `TRACED-style` 说成 official TRACED；
- 不要把 official TRACED 的 BLIP backbone 和主表 CLIP 混在一起；
- 不要把个人 GitHub URL 写进双盲论文，论文里用 `anonymous.4open.science`。

---

## 1. 项目背景与目标

### 1.1 问题背景

CLIP 等双模态模型的余弦相似度是“同一张图内排序”的常用分数，但在**跨对比较**时并不可比：

- 不同图像/文本的分数尺度不同；
- 一对图的分数高低，可能主要由图像侧或文本侧的加性成分决定；
- 这种成分与“是否匹配”无关，却会改变全局排序。

论文把这一现象写成：

```text
z = m(I,T) + c(I,T)
```

其中 `m` 是跨模态匹配项，`c` 是可由 `u`、`v` 预测的加性成分。

### 1.2 方法目标

- 去掉可预测的加性成分；
- 不训练编码器；
- 不用 reference bank；
- 推理只依赖单对；
- 在全局 AUROC 上提升，同时保持 cycle-level 2AFC 不变。

### 1.3 论文贡献

1. **CRD 方法**：两个加性头 + 标准化 + 单标量校准；
2. **2AFC 不变性**：加性校准对 cycle 内两侧求和不改变排序；
3. **FGBench**：88,024 个 reciprocal cycle、五个事实维度、显式划分与审计；
4. **实验**：五维主表、配对 bootstrap + Holm、跨基座、head/target 消融、joint-head 对照、边界分析。

### 1.4 当前论文状态

```text
英文：paper/CRD_ICME2027/main.pdf                 6 页
英文副本：CRD_ICME2027_6pages.pdf                 6 页
中文：paper/CRD_ICME2027/cn/main_cn.pdf           7 页
```

英文是投稿版；中文是内容对应版，排版比英文多约 1 页。

---

## 2. 仓库与代码结构

### 2.1 顶层目录

```text
fgbench/              FGBench 构建、模型、评估、实验
src/                  CLIP encoder 和 scoring 基础层
scripts/              数据准备、分析、审计、历史探索脚本
data/                 小型 FOIL 数据
results/fgbench/      paper-ready CSV 与小型结果
paper/CRD_ICME2027/   LaTeX 源文件、图、最终 PDF
handoff_fgb/          FGBench 数据管线 handoff bundle
analysis/             分析与作图
figures/              论文图
```

### 2.2 核心代码模块

| 路径 | 作用 |
|---|---|
| `fgbench/models/crd_backbone.py` | CRD 主实现：加性头、标准化、训练、λ 选择、评估 |
| `fgbench/models/crd_action.py` | Action 专用：内部 train/val 划分、图像不重叠 |
| `fgbench/models/gmm_filter.py` | GMM 单调对照 |
| `fgbench/models/crd_v2.py` | 早期 CRD 变体，保留作历史参考 |
| `fgbench/evaluate/strict_main_table.py` | 主 AUROC 表；需要冻结特征张量 |
| `fgbench/evaluate/final_stats_strict.py` | cycle-clustered paired bootstrap + Holm |
| `fgbench/evaluate/traced_fgbench.py` | TRACED-style：CLIP + 删除轨迹 + LR |
| `fgbench/evaluate/lemon_fgbench.py` | LEMoN-style：邻居一致性特征 + LR |
| `fgbench/evaluate/gmm_filter.py` | GMM 对照 |
| `fgbench/evaluate/csls_gmm_blip.py` | CSLS/GMM 辅助 |
| `fgbench/experiments/joint_ablation.py` | additive vs joint MLP 同预算消融 |
| `src/encoders.py` | CLIP / SigLIP2 / FLAVA 编码器封装 |
| `src/protocol.py` | 数据路径、数据行、bootstrap 辅助 |
| `scripts/` | 大量历史实验；**不是所有脚本都属于最终论文** |

### 2.3 哪些是最终论文主线

优先看：

```text
fgbench/models/crd_backbone.py
fgbench/models/crd_action.py
fgbench/evaluate/strict_main_table.py
fgbench/evaluate/final_stats_strict.py
fgbench/evaluate/traced_fgbench.py
fgbench/evaluate/lemon_fgbench.py
fgbench/experiments/joint_ablation.py
results/fgbench/paper_ready/
paper/CRD_ICME2027/
```

`scripts/` 里很多是早期路线、失败实验、数据下载/上传、审计工具。不要因为文件名多就全部重跑；先看 `RESULTS.md` 和 `paper_ready/`。

---

## 3. 数据与当前结果

### 3.1 FGBench

```text
full v1-strict:   88,024 cycles / 352,096 pairs
strict test:      17,565 cycles / 70,260 pairs
```

五个 facet：

```text
COCO Entity
COCO Quantity
VG Attribute
VG Action
VG Relation
```

每个 reciprocal cycle 有 4 行：2 个 clean、2 个 noisy。图像和 caption 在 clean/noisy 两侧各出现一次，因此任何单侧分数在两类上的条件分布相同。

### 3.2 主表（seed-mean）

| Facet | CRD | LEMoN official | TRACED-style |
|---|---:|---:|---:|
| COCO Entity | 0.9995 | 0.9990 | 0.9985 |
| COCO Quantity | 0.8837 | 0.8266 | 0.8412 |
| VG Attribute | 0.7187 | 0.6755 | 0.6875 |
| VG Action | 0.6145 | 0.5923 | 0.6102 |
| VG Relation | 0.5530 | 0.5218 | 0.5356 |

Attribute 原始余弦：

```text
0.6806 -> 0.7187（+3.81 points）
```

统计：

```text
24 个 Holm cells 中 21 个显著
扣除 4 个 GMM 单调重复后，17/20 independent
例外：Entity vs CSLS、Quantity vs CSLS、Action vs TRACED
```

### 3.3 官方 baseline 的口径

- **LEMoN**：official adapter，完整 FGBench test split，五维；主表用 official。
- **TRACED**：主表用 `TRACED-style`（CLIP + 删除轨迹 + LR），是 lower bound。
- **official TRACED**：COCO Entity/Quantity 有完整结果；VG 只有 100-pair pilot；backbone 是 BLIP COCO ITM，不能和主表 CLIP 直接比。
- 详细见 `results/fgbench/paper_ready/REPORT_official_baselines.md`。

### 3.4 joint-head 消融

- additive：65,795 参数；
- joint MLP `[u;v]`：65,665 参数；
- 四个共享缓存 facet，3 seeds；
- joint 没有超过 additive；Quantity 最明显（0.8759 vs 0.8866）；
- additive 列是 same-cache re-run，和 Table II 最多差 0.003 AUROC（Quantity；Relation ≤0.001）。

### 3.5 P1 负结果

P1 在 `results/fgbench/paper_ready/p1_negative/`：

- classification probe：CRD 0.9543 < Cosine 0.9560；
- hard P1：CRD 不是非 Oracle 最优；
- full retrieval R@1：CRD 0.1010 < Cosine 0.1090，seed2 CI 排除 0（负方向）。

结论：**不要写 downstream utility claim**。

---

## 4. 当前已完成的工作与进度

| 项 | 状态 |
|---|---|
| FGBench v1-strict 构建与审计 | 完成 |
| 五维主表 + 三 seeds | 完成 |
| 2AFC 不变性证明 | 完成 |
| 配对 bootstrap + Holm | 完成 |
| official LEMoN 五维 | 完成 |
| TRACED-style 五维 | 完成 |
| official TRACED COCO Entity/Quantity | 完成 |
| official TRACED VG | 只有 100-pair pilot |
| joint vs additive 消融 | 完成 |
| 跨基座 Attribute（3 backbones） | 完成 |
| P1 classification / hard / retrieval | 完成，负结果 |
| P0-A 人工标注 | 未完成（89/400） |
| 英文 6 页版 | 完成 |
| 中文对应版 | 完成（7 页） |
| 匿名审稿包 | 在 `anonymous-release` 分支 |
| 4open 匿名链接 | 待生成 |

---

## 5. 已知问题与待办事项

### 5.1 论文/投稿相关

1. **4open 链接未生成**  
   仓库 `main` 是正常代码；匿名包在 `anonymous-release` 分支。  
   论文里不要写 `github.com/chenjunxi-Ranger/crd`，要用 4open 包装 `anonymous-release`。

2. **TRACED 引用缺 arXiv 号**  
   `references.bib` 里 `afriat2025traced` 只有 arXiv preprint，没有真实 arXiv ID；投稿前必须补。

3. **official TRACED 的 CLIP 版没跑**  
   主表只有 `TRACED-style`；不要声称与 official TRACED 完全持平。

4. **official LEMoN 的 train 段是子集**  
   逐行文件 test 完整；Entity/Attribute train 是 32k 行子集，Relation 是 8k 行。主表 test 数值不受影响，但不要把它说成“完整训练库”。

5. **Action 协议偏差**  
   Action 官方 validation 只有 2 cycles；当前用 80-cycle 内部划分，是单次固定划分，标为 exploratory。

### 5.2 数据/复现相关

1. **仓库不含原始图片**  
   COCO/VG 需要从官方源下载；`image_index.parquet` 和 manifest 给出对应关系。

2. **仓库不含特征缓存**  
   `*.npz`、`*.pt`、trajectory dump 没上传；重新训练需要重新提特征或从原工作区拷贝。

3. **`strict_main_table.py` 不能只靠本包单独跑**  
   它需要冻结特征张量（`crdx_b16_test.npz`、`bank_b16_*.npz` 等），这些没有随包。论文数字看 `results/fgbench/paper_ready/main_table_strict.csv`。

4. **代码里仍有原工作区路径**  
   很多脚本按原 `mymodel7/` 目录布局写死；接手后先全局搜索 `D:\`、`/mnt/d/`、`mymodel7`，按本机路径改。

5. **`_cloud_ssh.py` 已删除**  
   原文件含明文服务器地址和密码；不要恢复，也不要提交任何凭据。

### 5.3 仓库/Git 相关

- `main`：正常代码仓库（给同伴 clone）；
- `anonymous-release`：匿名审稿包（给 4open 用）；
- 本地可能还有未推送的清理提交；clone 前先确认 `git log -1` 和远端一致。
- 不要把 `release/CRD_code.zip` 或大特征文件提交到 Git。

---

## 6. 运行与部署方式

### 6.1 环境

建议：

```text
Python 3.12
PyTorch 2.5.x + CUDA
transformers 4.46.x
timm 0.9.x
faiss-cpu
scikit-learn
pandas / pyarrow
```

安装：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 6.2 数据与权重

需要准备：

```text
COCO 2014 train/val 图片与 captions/instances
Visual Genome 图片（VG_100K / VG_100K_2）
CLIP ViT-B/16 权重（主表）
可选：CLIP L/14、SigLIP2、BLIP-ITM（对照）
```

仓库已包含：

```text
fgbench/data/fgbench_v1_strict/*.parquet
fgbench/data/fgbench_v1_strict/*.json
fgbench/data/intermediate/vg/image_index.parquet
```

不包含原始图片和特征缓存。

### 6.3 环境变量

代码里路径不完全统一，建议先设置：

```text
VG_ROOT           Visual Genome 根目录
COCO_ANN          COCO annotations 目录
FGBENCH_BENCH     FGBench manifest 目录
FGBENCH_OUT       派生数据输出目录
CRD_WEIGHTS       CLIP/BLIP 权重根目录
CLIP_DIR          CLIP 权重目录
```

### 6.4 关键命令

单 facet 训练 CRD（示例）：

```bash
python fgbench/models/crd_backbone.py \
  --backbone clip --dataset VG --facet Attribute \
  --cycles 8000 --target row --seed 0
```

Action 专用：

```bash
python fgbench/models/crd_action.py --seed 0
```

joint 消融：

```bash
python fgbench/experiments/joint_ablation.py \
  --facets Entity,Quantity,Attribute,Relation --seeds 3
```

主表/统计（需要原工作区特征张量）：

```bash
python fgbench/evaluate/strict_main_table.py
python fgbench/evaluate/final_stats_strict.py
```

论文编译：

```powershell
cd paper/CRD_ICME2027
powershell -ExecutionPolicy Bypass -File tools/build.ps1
```

### 6.5 复现边界

**能直接核验：**

- `results/fgbench/paper_ready/main_table_strict.csv`
- `official_lemon_stats.csv`
- `joint_vs_additive.csv`
- `fgbench/data/fgbench_v1_strict/*.parquet` 的计数/审计
- `p1_negative/*.csv` 的负结果

**不能一键重训：**

- 从原始图片重提 CLIP 特征；
- 重跑完整 trajectory / official TRACED；
- 重跑 `strict_main_table.py`（缺特征张量）。

交接时先让同伴用 CSV/manifest 对齐，再决定是否重提特征。

---

## 7. 交接给同伴时需要特别说明的事

1. **先读 `README.md` + 本文档 + `RESULTS.md`**，不要先跑 `scripts/` 里的所有历史脚本。
2. **最终论文只看 `paper_ready/` 和 `paper/CRD_ICME2027/`**；`scripts/` 里很多是失败路线。
3. **`TRACED-style` 不是 official TRACED**：它是 CLIP + 删除轨迹 + LR，定位是 lower bound。
4. **official TRACED 的 BLIP backbone 不能和主表 CLIP 混比**；VG official 只有 100-pair pilot。
5. **不要写 P1 下游收益、human κ**；这些是明确的负结果/未完成项。
6. **主表 CRD 的 `auroc_seedmean` 与 `auroc` 不同**：论文正文用 seed-mean，integrated score 在 CSV 里另列。
7. **joint 表的 additive 是 same-cache re-run**，和 Table II 最多差 0.003 AUROC，不要误写成完全相同的实验。
8. **数据/特征很大，不要提交 Git**：只提交代码、manifest、paper-ready CSV、最终 PDF。
9. **凭据不要入仓库**：`_cloud_ssh.py` 已移除；不要恢复。
10. **Git 分支**：`main` 给同伴；`anonymous-release` 给 4open；不要把 `main` 当成匿名包。

---

## 8. 建议接手顺序

### 第 1 天

1. Clone/解压代码；
2. 读 `README.md`、本文档、`RESULTS.md`；
3. 检查 `results/fgbench/paper_ready/main_table_strict.csv`；
4. 安装依赖，确认 `torch.cuda.is_available()`；
5. 下载 COCO/VG，设置环境变量；
6. 只跑一个 facet（建议 Attribute）验证数据路径。

### 第 2 天

1. 重提 CLIP B/16 特征（或从原工作区拷贝缓存）；
2. 重跑 `crd_backbone.py --target row`；
3. 用 `final_stats_strict.py` 对齐主表；
4. 检查 joint 表是否需要重跑。

### 第 3 天

1. 补 official TRACED/LEMoN 的 supplement；
2. 生成 4open 链接；
3. 补 TRACED arXiv 号；
4. 重新编译英文 6 页版。

---

## 9. 关键文件索引

```text
README.md
HANDOVER.md
METHOD.md
RESULTS.md
CONSTRAINTS.md
requirements.txt

fgbench/models/crd_backbone.py
fgbench/models/crd_action.py
fgbench/evaluate/strict_main_table.py
fgbench/evaluate/final_stats_strict.py
fgbench/evaluate/traced_fgbench.py
fgbench/evaluate/lemon_fgbench.py
fgbench/experiments/joint_ablation.py

fgbench/data/fgbench_v1_strict/
fgbench/data/intermediate/vg/image_index.parquet

results/fgbench/paper_ready/main_table_strict.csv
results/fgbench/paper_ready/official_lemon_stats.csv
results/fgbench/paper_ready/joint_vs_additive.csv
results/fgbench/paper_ready/p1_negative/

paper/CRD_ICME2027/main.pdf
paper/CRD_ICME2027/main.tex
paper/CRD_ICME2027/cn/main_cn.pdf
paper/CRD_ICME2027/cn/main_cn.tex
```

---

## 10. 一句话交接

> 先以 `results/fgbench/paper_ready/` 的冻结结果为准；代码用于审计和重训。不要重新打开已经证伪的 P1/utility 路线，不要把 `-style` 写成 official，不要在论文里暴露个人 GitHub URL。
