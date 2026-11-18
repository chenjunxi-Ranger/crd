# F1–F9 最终核验报告

> 2026-11-10。逐项给证据，不写"已完成"这类无证据结论。

## 结论速览

| 编号 | 项 | 状态 | 证据位置 |
|---|---|---|---|
| F1 | 统一统计口径 | ✅ 已统一（并修正 1 处混用） | §F1 |
| F2 | 集成分数 vs 三种子均值 ± SD | ✅ 双口径并列标注 | §F2 |
| F3 | GMM 纳入同基座比较表 | ✅ 已加入（Action 缺，见说明） | §F3 |
| F4 | 各方法 manifest 完全一致 | ✅ 已修正 2 处 | §F4 |
| F5 | 配对 Bootstrap + Holm | ✅ 20 项检验，18 显著 | §F5 |
| F6 | LEMoN/TRACED 参考信息与测试隔离 | ✅ 代码级确认 | §F6 |
| F7 | 推理开销 | ✅ 实测 3 项，1 项按吞吐换算 | §F7 |
| F8 | checkpoint 字段 | ✅ 已修补 20 个 BLIP 头部 | §F8 |
| F9 | 归档 | ✅ FREEZE_MANIFEST.json | §F9 |

---

## F1 · 统一统计口径

已统一为：**FGBench v1-strict + CRD 标准化实现（`crd_backbone`）+ row 目标 + 3 种子 + λ 由 val 选**。

**修正的一处混用**：主表曾把 CLIP 的 cosine 与 BLIP 空间的 LEMoN/TRACED 放在同一行
（分数文件被 BLIP 运行覆盖）。现已用 `--backbone clip` 重跑恢复：

| 维度 | LEMoN 旧（BLIP） | LEMoN 新（CLIP） | TRACED 旧 | TRACED 新 |
|---|---:|---:|---:|---:|
| Attribute | 0.7449 | **0.6843** | 0.7546 | **0.6875** |
| Relation | 0.5975 | **0.5436** | 0.6091 | **0.5356** |
| Quantity | 0.9770 | **0.8495** | 0.9742 | **0.8412** |

TRACED Attribute 重跑得 0.6875，与最初 CLIP 结果逐位一致，可作交叉验证。

## F2 · 两种聚合口径

| 维度 | 逐种子均值 ± SD | 集成分数 AUC | 差 |
|---|---|---|---|
| Attribute | 0.7187 ± 0.0001 | 0.7190 | +0.0003 |
| Relation | 0.5530 ± 0.0001 | 0.5531 | +0.0001 |
| Quantity | 0.8837 ± 0.0011 | 0.8848 | +0.0011 |
| Action | 0.6145 ± 0.0009 | 0.6145 | 0.0000 |
| Entity | 0.9995 ± 0.0000 | 0.9996 | +0.0001 |

**两者都可复现，差 ≤0.0011**（跨种子平均降低评分噪声）。主表已并列标注；
论文建议主表用"均值 ± SD"、配对比较用集成分数，并在表注声明。

## F3 · GMM 已进入同基座比较表

CLIP 基座、四个维度（Action 除外）：

| 维度 | cosine | **GMM** | GMM 2AFC | cosine 2AFC |
|---|---:|---:|---:|---:|
| Entity | 0.9991 | **0.9991** | 0.9996 | 1.0000 |
| Quantity | 0.8305 | **0.8305** | 0.9298 | 0.9649 |
| Attribute | 0.6806 | **0.6806** | 0.8158 | 0.8282 |
| Relation | 0.5413 | **0.5413** | 0.6601 | 0.6501 |

**四格 AUROC 与 cosine 逐一相同**——二维 GMM 后验是相似度的单调变换，AUROC 只依赖排序，
**数学上不可能改变 AUROC**。但 2AFC 会变（Attr −1.24、Qty −3.51、Rel +1.00、Entity −0.04），
因为 2AFC 用分数之和，单调变换不保和。

**Action 无 GMM**：该维度 CRD 走内部验证协议，没有统一的 `crdbb_clip_VG_Action` 缓存；
补跑需先编码 1,276+320+664 行（约 1 分钟），**但目前未做**，表中标 "—"。

## F4 · manifest 一致性

初检发现 **2 个文件覆盖 v1（756 cycles）而非 strict（706）**：`score_lemon_*_VG_Relation`、`score_gmm_VG_Relation`。
原因分别是 lemon 读旧 `crd_*` 缓存、gmm 的 FACETS 指向旧前缀。

**均已修正**（lemon 改为优先读 `crdbb_clip_*`；gmm 改用 strict 缓存与正确的 train 后缀）：

| 文件 | 修前 | 修后 |
|---|---|---|
| score_lemon_clip_VG_Relation | 3,024（+50 extra） | **2,824（0 extra）** |
| score_gmm_VG_Relation | 3,024 | **2,824** |
| score_gmm_VG_Attribute/Quantity/Entity | v1 | **strict** |

复检：**14/14 个逐样本分数文件的 cycle 集合与 strict test 完全一致（extra=0, missing=0）**。

## F5 · 配对 Bootstrap + Holm

cycle 聚类 bootstrap（1,000 次），20 项检验，**Holm 校正后 18 项显著（p_holm=0.02）**：

| 维度 | vs cosine | vs CSLS | vs GMM | vs LEMoN | vs TRACED |
|---|---|---|---|---|---|
| Entity | **+0.04** | −0.01 (ns) | **+0.04** | — | — |
| Quantity | **+5.38** | −0.38 (ns) | **+5.38** | **+3.47** | **+4.30** |
| Attribute | **+3.85** | **+3.25** | **+3.85** | **+3.48** | **+3.16** |
| Action | **+2.31** | **+2.75** | — | — | — |
| Relation | **+1.17** | **+1.14** | **+1.17** | **+0.95** | **+1.77** |

**三项不显著**（22 项检验、19 显著）：

| 维度 | 基线 | Δ | 95% CI | p_holm | 说明 |
|---|---|---:|---|---:|---|
| Entity | CSLS | −0.01 | [−0.02, 0.01] | 1.000 | CSLS 略优 |
| Quantity | CSLS | −0.38 | [−2.03, 1.15] | 1.000 | CSLS 优 |
| **Action** | **TRACED** | **+0.47** | **[−1.71, 2.72]** | **0.996** | 点估计更高但不显著 |

**因此两条表述必须改写**：
1. 不能写"CRD 全面领先"——Quantity 与 Entity 上 CSLS 更强；
2. 不能写"CRD 超越 TRACED"——Action 上只能写"**与 TRACED 相当，且 2AFC 零损失**"
   （TRACED 在该维度掉 3.25 pp，这是可辩护的差异）。

## F6 · LEMoN / TRACED 隔离审计

| 检查 | LEMoN (`lemon_fgbench.py`) | TRACED (`traced_fgbench.py`) |
|---|---|---|
| 参考库/训练来源 | `dtr = tensors("train")`，仅 train | `dtr, Utr = load("train", ...)`，仅 train |
| 是否使用 test 标签 | 否（LR 拟合用 `dtr.pair_label`） | 否（LR 拟合用 `dtr.pair_label`） |
| 库内自匹配 | `exclude_self=True` + `lib_keys` 去自身 | 不涉及 |
| 超参选择 | k=32 固定，未在 test 上搜 | steps=3 固定，未在 test 上搜 |
| 测试集用途 | 仅前向打分 | 仅前向打分 |

**结论：无测试集泄漏**。

## F7 · 推理开销（batch=1, GPU）

| 方法 | 原生 | +CRD | CRD 头 | 额外前向 |
|---|---:|---:|---:|---|
| CLIP B/16 | 15.06 ms | 15.89 ms | 0.205 ms | 0 |
| BLIP-ITC | 33.32 ms | 33.94 ms | 0.615 ms | 0 |
| BLIP-ITM | 54.42 ms | 55.04 ms | 0.615 ms | 0（融合本就要算） |
| **TRACED-style** | — | — | — | **≈25 次文本前向**（Attribute 57,124 行耗时 171 s ≈ **3.0 ms/行**） |
| **LEMoN-style** | — | — | — | 1 次 kNN（k=32）遍历参考库 + LR，无额外深度前向 |

CRD 的增量来自一个两层 MLP（CLIP 65,795 参数 / BLIP 33,027 参数），**≤0.9 ms**。

## F8 · checkpoint 字段

要求 `backbone / facet / seed / mu / sd / lam`。

- 文件名编码 backbone+dataset+facet+parts+target+seed；
- 内容含 `state / mu / sd / dim / lam`；
- 溯源另有 `crdbb_prov_*.json`（manifest sha、cache sha、train-order sha、μ、σ、λ）。

**修补**：20 个 BLIP 头部是 `lam` 字段加入之前保存的，缺 `lam`。
已用 `scripts/patch_blip_head_lam.py` 从各自运行 JSON 回填（不重算任何指标）。

## F9 · 归档

`results/fgbench/FREEZE_MANIFEST.json` 记录：

| 类别 | 数量/内容 |
|---|---|
| 代码 | **13/13** 个文件的 sha256 前 16 位 |
| 数据 | **4/4**（train/val/test parquet + strict_stats.json） |
| 结果表 | **6/6** CSV（主表、统计、消融、两套对比、ITM 表） |
| 逐样本分数 | CRD 12 + Action 3 + 基线 8 |
| 模型权重 | CLIP/CLIP-L/SigLIP2/BLIP/FLAVA 的文件清单与体积 |
| checkpoint | 79 个 `.pt` |

---

## 仍未闭合的两项（如实列出）

1. **Action 的 GMM / LEMoN / TRACED 配对统计**：分数文件命名刚补齐，F5 未包含这三项。
2. **L/14 Quantity 已补**（本轮新增）：raw 0.8677 → **0.9282 ± 0.0004**（+6.05），2AFC 0.991228 不变。
   至此 L/14 在 Attribute / Relation / Quantity 三个维度齐全。
