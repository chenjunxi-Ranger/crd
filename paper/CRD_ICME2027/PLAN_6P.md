# CRD v1 · ICME 6 页结构设计（v2，经三名子 agent 评审后修订）

> 目标：ICME 2027，CRD v1 为唯一主方法，6 页含参考文献。
> 依据：ICME 2026 官方 camera-ready「total of 6 pages」；2027 的 author kit 尚未发布。
> v1 的页预算算术不成立（评审 R2 指出），v2 改为**词预算**并加入内容层必改项。
> 本文件只做结构，不改正文。

---

## 0 · 一句话论证

> 在全局图文错标检测中，我们指出冻结相似度**跨对不可比**（Attribute 上 31.9% 的干净/噪声对排序反了），并用一个**单对推理、加性、低标签**的校准把全局排序拉回一部分；证据是 88,024 个 reciprocal cycle 的 FGBench、五个冻结骨干与 22 项配对检验；边界是它不修复关系/饱和 facet、在 Quantity 上不敌 CSLS、且标签为银标签。

论文类型：**methods**。贡献顺序：方法 → 基准 → 分析/边界。

**审稿人应带走的三件事**（v2 新增第 4 条）：

1. 前提是错的，而且可量化（31.9%；L/14 不变）。
2. 修法极简且可证（65,795 参数、0.2 ms、无参考库；加性 ⇒ cycle-level 2AFC 不变）。
3. 边界写得比增益更清楚（Relation 弱、Quantity 输 CSLS、探针更高、银标签）。
4. **它能把 +3.8 pp AUROC 翻译成清洗收益**（top-10% precision，待补；见 §8）。

---

## 1 · 术语账本（锁定）

| 规范形式 | 首次定义 | 禁止变体 |
|---|---|---|
| CRD | cross-modal score calibration | “CRD v1”“our module” |
| FGBench | FGBench v1-strict | “our dataset”“FG-ITM benchmark” |
| reciprocal cycle | 两图两句构成的 2×2 四行 | “quadruple”“pair group” |
| facet | Entity / Quantity / Attribute / Action / Relation | “dimension”“category” |
| additive component | `g_θ = a(u)+b(v)+b₀` | “bias”“offset term” |
| 2AFC | cycle-level two-alternative forced choice | “pair accuracy” |
| silver label | rule-derived label | “weak/pseudo label” |
| CSLS / LEMoN-style / TRACED-style | 同协议复现 | 不写“official implementation” |

---

## 2 · 词预算（v2 修正）

**不再用页分数相加**——v1 表格逐行相加为 6.02 + 0.75 = 6.77，与表头 6.00 矛盾。
v2 用「词预算 + 浮动体页数」核算：

- IEEEtran 10pt / US Letter 纯文本 ≈ 950 词/页。
- 浮动体：Fig.1 单栏 0.22、Fig.2 通栏 0.32、Table I 0.12、Table II 0.42、
  Table III-mini 0.15、Table IV 0.18、External-mini 0.18 → **1.59 页**。
- References：实际被引约 15 条 → **0.25 页**（v1 写 0.75 页，高估）。
- 可用正文 = 6 − 1.59 − 0.25 = **4.16 页 ≈ 3,950 词**。
- 当前正文 ≈ 6,300 词 → **需砍 2,350 词（−37%）**。

| 部分 | 词数 | 浮动体 |
|---|---:|---|
| Abstract | 185 | — |
| 1 Introduction（3 段） | 500 | — |
| 2 Related Work（2 段） | 240 | — |
| 3 Problem & Motivation（3 小节） | 400 | — |
| 4 Method（4 小节） | 740 | Fig.2（0.32） |
| 5 FGBench（2 小节） | 400 | Table I（0.12） |
| 6 Experiments（5 小节） | 1,100 | Table II/III-mini/IV/Ext（0.93） |
| 6.6 Limitations | 160 | — |
| 7 Conclusion | 110 | — |
| **正文合计** | **3,835** | **1.37 + Fig1 0.22 = 1.59** |
| References | — | 0.25 |
| **总计** | 3,835/950=4.04 | +1.59+0.25 = **5.88 页** |

余量 0.12 页。若编译超 6 页：先删 Related 第二段（−0.1 页），再把 Method protocol 移补充（−0.1）。

---

## 3 · 逐段映射（v2 修订，一段一个 job）

### Abstract（185 词，6 个动作）
1. 阈值清洗默认分数跨对可比。
2. 缺口：Attribute 上 31.9% 反序。
3. 方法：冻结编码器 + 两个加性头 + 行级回归 + 单对推理。
4. 结果：0.6806→0.7187，19/22，五骨干 15/15。
5. 机制：加性 ⇒ cycle-level 2AFC 不变。
6. 边界：Relation 弱、Quantity 不敌 CSLS、**探针在 Attribute/Quantity/Action 分别高 0.4/1.5/7.6（消耗全量标签）**、银标签。

> v1 摘要只写“probe +0.4 on Attribute”，与正文三处不一致，评审 R2/R3 均列为选择性报告；v2 必须写全。

### 1 Introduction（3 段，删掉与 §VI-C 重复的容量段）
- **I-P1（context + gap，170 词）**：清洗=阈值；阈值=可比性假设；Fig.1；31.9%。不写方法。
- **I-P2（prior attempts + capacity null，180 词）**：CSLS 改分数、ITM 换匹配器、LEMoN/TRACED 依赖上下文；L/14 2.9× 参数但 Attribute 0.6779 vs 0.6806。**此段吸收原 intro ¶3，删除 §VI-C 的重复叙述。**
- **I-P3（approach + contributions，150 词）**：加性分解 → 校准 → FGBench；三条贡献；只给一个头条数字。

### 2 Related Work（2 段，240 词）
- **R-P1（130 词）**：三条检测器线；都以“相对上下文”定义错。
- **R-P2（110 词）**：hubness/anisotropy、单调变换；FOIL/SugarCrepe 是同图；本文跨对、单对推理、无库。

### 3 Problem Formulation（3 小节，400 词）
- **3.1（130 词）**：`q(I,T)` 全局排序；global / pair-local 两性质；AUROC 与 cycle-level 2AFC 分开。
- **3.2（150 词）**：Fig.1 + 31.9% + 同一 cycle 三个分数。只观察。
- **3.3（120 词）**：`z=m+c`；两个可检验后果；明确“不是可识别性断言”。

### 4 Method（4 小节，740 词）
- **4.1（150 词）**：Fig.2；冻结双塔、两个头、一次减法、无库。
- **4.2（220 词）**：`z=(s−μ)/σ`、`g=a(u)+b(v)+b₀`、MLP 宽 64；μ/σ 只来自 train。
- **4.3（180 词）**：`Huber(g,z)`；为什么不用 cycle 均值（组内常数）；“标签只进 λ，因此不是无监督”——**此句是可信度锚点，不可删**。
- **4.4（190 词）**：`z'=z−λg`；λ 网格含 0；AdamW/1500/3 seeds；0.2 ms；**一句 release（代码、头、逐样本分数、manifest）**。

### 5 FGBench（2 小节，400 词）
- **5.1（230 词）**：2×2 四行、50/50、每图每句各一次；五 facet 与来源；挡频率捷径。
- **5.2（170 词）**：88,024 cycles / 352,096 行；test 17,565 / 70,260；银标签、strict tier、证据完整率、TF-IDF、零泄漏——**数字引用 `fgbench/data/benchmark/final_audit.json`，不要只写“audited”**。

### 6 Experiments（5 小节，1,100 词）
- **E-P1 Setup（140 词）**：单骨干 B/16；训练协议；指标；cycle bootstrap + Holm；Action 内部 val 一句；**CSLS 的 λ 与 CRD 在同一 val 预算下重选**（评审 R3 要求）。
- **E-P2 主结果（280 词）**：Table II；Attribute +3.81 种子均值 / +3.85 集成（**只选一个口径写 Δ**）；19/22；三个例外。
- **E-P3 两个 null 与一个上界（200 词）**：centred/ABTT 10/10（注明 Relation vs ABTT CI 触零）；探针 +0.4/+1.5/+7.6 与标签预算。
- **E-P4 跨骨干（70 词）**：Table III-mini（B/16、L/14、SigLIP2 × Attribute）；15 格进补充；L/14 参数 2.9× 但不变。
- **E-P5 消融（170 词）**：Table IV 4 行；超可加 0.93；row>cycle 一句；placebo 归零。
- **E-P6 2AFC（90 词）**：循环内加性项抵消；**Action 0.7831、Entity 1.0000**；GMM 单调变换 AUROC 不变但 2AFC 变。
- **E-P7 外部（110 词）**：Entity 源；**4/5 正向，但 2 个 same-image accuracy + 3 个 row-level AUROC，不可混比**；SugarCrepe 的增益主要由 text-side head 承担；MM-IMDb −0.35（显著负）。
- **E-P8 效率（40 词）**：15.06→15.89 ms；头 0.205 ms。

### 6.6 Limitations（160 词）
银标签；Relation 0.553；Quantity/Entity 输 CSLS；Action 与 TRACED 平；探针更高；λ 用 val 标签；外部非同协议。

### 7 Conclusion（110 词）
贡献 → 证据 → 含义 → 边界。不引入新数字。

---

## 4 · 图表分配（v2 修订）

| 编号 | 内容 | 形式 | 页 | 去留 |
|---|---|---|---:|---|
| Fig. 1 | 0.252 vs 0.331 反例 | **单栏** | 0.22 | 保留 |
| Fig. 2 | 训练/推理总览 | **通栏但压高**（~1.6in） | 0.32 | 保留；单栏不可读 |
| Table I | FGBench 五 facet 统计 | 单栏 | 0.12 | 保留 |
| Table II | 主表 8 行（含探针） | 通栏 | 0.42 | 保留 |
| Table III-mini | B/16、L/14、SigLIP2 × Attribute（Raw/CRD/Δ） | 单栏 | 0.15 | **新增**；完整 15 格进补充 |
| Table IV | 消融 4 行（base/image/text/additive） | 单栏 | 0.18 | cycle/placebo 进正文 |
| Table V-ext | 外部 5 行（target/n/metric/CRD/Δ） | 单栏 | 0.18 | **新增**；替代 3 句散文 |

> v1 说“外部表移出省 0.40 页”是记账错误：当前稿本就没有外部表，外部是散文；v2 改为**保留 5 行小表**，因为它同时防止“选择性报告”（评审 R1）。

---

## 5 · Claim–Evidence map（v2 修订）

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | 跨对不可比真实且可量化 | 31.9% = 1−0.6806（写出“等于 1−AUROC”）；Fig.1 | supported |
| C2 | 加性校准改善全局排序 | +3.81 种子均值；+3.85 集成；19/22 | supported，**统一口径** |
| C3 | 增益不是特征白化 | vs centred +4.00、vs ABTT +4.26（10/10；Relation CI 触零注明） | supported |
| C4 | 两侧互补 | 1.38+1.50=2.88 < 3.81（缺口 0.93） | supported |
| C5 | cycle-level 2AFC 不变 | 代数恒等式（不是“定理”）；0.8282/0.9649/1.0000/0.7831/0.6501 | supported，**修正 Action** |
| C6 | 零样本迁移 | Entity 源 4/5；2 个 same-image + 3 个 row-level；MM-IMDb −0.35 | supported, mixed |
| C7 | 并非全面更优 | Quantity/Entity 输 CSLS；Action 与 TRACED 平；探针更高 | supported |
| C8 | 银标签 | strict tier、证据完整率、TF-IDF（引用 `final_audit.json`） | supported with caveat |
| C9 | 清洗收益 | top-10% precision（待补） | needs evidence |

**不允许出现**：SOTA、first、universal、outperforms all、robust to any noise。

---

## 6 · 内容层必改项（先于排版；三名评审共同指出）

| # | 位置 | 现状 | 必改为 |
|---|---|---|---|
| 1 | `6_experiments.tex:274` | “1.0000 on Entity and Action” | **Action 是 0.7831**；Entity 才是 1.0000 |
| 2 | `0_abstract.tex:25-26` | probe 只写 Attribute +0.4 | 写全 +0.4/+1.5/+7.6，并注明消耗全量标签 |
| 3 | `6_experiments.tex:71-73, 94-102` | 表题说集成、表体是种子均值；同段四种口径；Quantity 出现 5.32/5.38/5.43 | 表注写明两口径；正文 Δ 只用一种；Quantity 统一 |
| 4 | `6_experiments.tex:49-55` | “neither uses a label” 但 k 在 val 上选 | 改为“no gradient training；k/λ 在同一 val 上选” |
| 5 | `6_experiments.tex:98-100` | Action “+0.43” 配 CI[−1.71,2.72]（该 CI 属 +0.47） | 统一为 +0.47，或注明口径 |
| 6 | `6_experiments.tex:302-316` | 外部 4/5 平铺；与摘要“same-image 不变”冲突 | 分开 metric；写 MM-IMDb 显著负；`same-image` 限定为 cycle-level 2AFC |
| 7 | Setup | CSLS 的 λ 未说明 | 加一句“同一 val 预算重选” |
| 8 | `5_fgbench.tex` | “audited”无数值锚点 | 引用 `final_audit.json` 的 100% / TF-IDF 0.5000 |
| 9 | Baseline | LEMoN/TRACED 未说明用了 K=32 邻居库 | 加一句“信息更多，方向对 CRD 不利” |

---

## 7 · 与 v1 设计的差异（评审后）

| 项 | v1 | v2 |
|---|---|---|
| 页预算 | 页分数相加（6.02+0.75 与 6.00 矛盾） | 词预算 + 浮动体（5.88 页） |
| References | 0.75 页 / 28 条 | 0.25 页 / ~15 条 |
| Intro | 3 段（漏掉容量段） | 3 段，容量段并入 P2，删 §VI-C 重复 |
| 跨骨干 | 全表移出 | **保留 3 行 mini 表**，15 格进补充 |
| 外部 | 3 句散文 | **5 行小表**，防选择性报告 |
| Fig.2 | 改单栏 | 通栏压高（单栏不可读） |
| 摘要 probe | 只写 Attribute | 三个 facet 全写 |
| 2AFC | 称为“定理” | 降级为“代数恒等式”，并修正 Action |
| Utility | 无 | **top-10% precision（新增）** |

---

## 8 · 三个可选加分项（按性价比）

1. **top-10% precision**（最高性价比，零训练）：用现有逐样本分数，报“最可疑 10% 中的噪声比例”，把 +3.81 pp 翻译成清洗收益。评审 R1 认为这是翻 borderline 的关键。
2. **CSLS λ 网格复核**：外部报告已证明网格截断会反转结论；在 FGBench 上用同一 val 预算重选 λ，确认 Quantity/Entity 的领先不是假象。
3. **200 对盲标一致率**：银标签的唯一硬证据；0.5–1 天。

---

## 9 · 风险与应急

| 风险 | 触发 | 应急 |
|---|---|---|
| 6 页仍放不下 | 编译 >6 | 删 Related 第二段、Method protocol 移补充 |
| 2027 允许 7–8 页 | CFP 发布 | 恢复完整 Table III + 外部表 |
| 审稿人要下游实验 | rebuttal | 已有 protocol（16–24 GPU-h） |
| 银标签被攻击 | 审稿意见 | 200 对盲标 + 一致率 |
| “为什么不用探针” | 审稿意见 | 主表探针行 + 标签预算句 |
