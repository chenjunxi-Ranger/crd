# FGBench v1 · 数据卡

> 冻结日期：2026-11-07 · 划分：COCO = Karpathy，VG = 官方 VG150，Flickr = 官方三分
> 构造规则：双向互惠 2×2；**只有明确互斥的事实才产生 noisy 标签**

---

## 1. 范围

| 数据源 | 维度 | 文本粒度 | 主任务 |
|---|---|---|---|
| MS COCO 2014 | Entity、Quantity | 整图 caption | Clean/Noisy 二分类（AUROC） |
| Visual Genome 1.2 | Attribute、Action、Relation | 区域描述 | Clean/Noisy 二分类（AUROC） |
| Flickr30k Entities | Entity Grounding | 整图 caption + 实体短语 | Recall@K / IoU（不做二分类） |

三种文本粒度**分开报告**，不合并成一个总分。

---

## 2. 构造：双向互惠 2×2

一个 cycle = 两张图 (A,B) + 两条原文 (pa,pb)：

| | pa | pb |
|---|---|---|
| A | clean | noisy |
| B | noisy | clean |

不变量：每图 / 每条 caption 恰好 1 次 clean + 1 次 noisy；同一文本在两类下逐字相同；两图必须同 split。

### 冲突规则（关键）

| 维度 | 互斥依据 |
|---|---|
| Entity | A 有类别 c、B 无；B 有 A 无的类别 c′（COCO 80 类封闭体系） |
| Quantity | 同类 c，两图计数不同，两条 caption 各断言与自身图像一致的数量 |
| Attribute | **同一属性槽位**且取值不同：color（red/blue/…）、state（open/closed）、size（large/small）。**跨槽位不算冲突**（如 wooden↔brown） |
| Action | **4 组动作对，语义裁决**（见 §2.2）：sitting on↔standing on、walking on↔running on、eating↔drinking、riding↔walking |
| Relation | **6 组空间对，分级**（见 §2.1）：Tier A = on↔under、behind↔in front of；Tier B = above↔below、over↔under、left of↔right of；未验证 = inside↔outside |

**"谓词不同"本身不是冲突**——holding 与 wearing 可以同时成立，因此不进入核心 noisy 集。
**"没有标注"也不是冲突**——只保留两侧都有正向事实、且可验证互斥的样本。

VG 事实必须被同图的一条区域描述同时提到"实体/谓词 + 宾语"才进入（自动对齐）；
无法对齐的一律丢弃，不用 VLM 生成补齐。

---

### 2.1 关系冲突规则的全量发现审计（train-only）

我们没有预设"6 组空间对"，而是先在 **train 划分**上做一次语料级挖掘，再验证：

| 步骤 | 规模 |
|---|---:|
| 扫描 relationships（train） | **1,566,098** |
| 不同谓词 | **26,936** |
| 有双方框、可做几何验证 | 1,317,705 |
| 共享 `(主体类, 客体类)` 上下文的候选**四元组** | **2,714** |
| 折叠后的不同**谓词对** | **478** |

判据（全部记录在 `final_conflict_rules.json`）：

```
垂直轴：above ⇔ center_y(主体) < center_y(客体)      （图像坐标，y 向下）
水平轴：left  ⇔ center_x(主体) < center_x(客体)
纯度  ：|P(侧) − 0.5| + 0.5，逐谓词计算
对立度：两侧多数方向相反时 = min(纯度_A, 纯度_B)
阈值  ：纯度 ≥ 0.60、类上下文支持 ≥ 20、实例共现率 ≤ 0.05
排除轴：containment（包含度受物体尺度混淆，不予采用）
```

**分级结果：**

| 等级 | 谓词对 | 证据 |
|---|---|---|
| **Tier A** | `on ↔ under`（对立度 0.616，8 个上下文）、`behind ↔ in front of`（0.724，3 个上下文） | 几何对立 + 语义互斥 + **可构造** |
| **Tier B** | `above ↔ below`（0.918）、`over ↔ under`（0.842）、`to left of ↔ to right of`（0.879） | 几何对立，但**共享类上下文支持 < 20**（above↔below 有 206 个上下文、最大支持 21；over↔under 298 个、最大 11；left of↔right of 46 个、最大 2）→ 无规则级确认 |
| 未验证 | `inside ↔ outside` | 依赖 containment 轴，该轴已被排除 |
| Tier C（拒绝） | `carrying↔riding`、`holding↔on`、`has↔in`、`wearing↔wears` | 语义兼容或仅词形变体 |

**关键结论：可构造的互斥谓词对极其稀疏。** 2,714 个候选四元组中，同时满足几何对立、
语义互斥、且能形成互惠循环的只有 2 对。限制来自 VG 的类上下文稀疏性，而非人工挑选。

### 2.2 为什么 Action 不做几何验证

`sitting on` 与 `standing on` 的几何签名**完全相同**（主体均在客体上方，P=0.813 / 0.875）。
姿态类冲突无法由二维框裁定，因此 Action 采用**语义裁决**，并单独报告，不声称是几何挖掘的产物。

**Action 的等级是语义等级，不是几何置信等级**（不要与 Relation 的 Tier A/B 混用）：

| 等级 | 规则 | cycles | 判定 |
|---|---|---:|---|
| **Strict**（主评测） | `sitting on ↔ standing on` | **504** | 同一槽位下坐/站互斥 |
| **Strict**（主评测） | `running on ↔ walking on` | **63** | 同一槽位下奔跑/行走为不同运动状态 |
| **Conditional** | `riding ↔ walking` | 38 | 可能同时成立（牵车步行），单独报告 |
| **Excluded** | `drinking ↔ eating` | 1 | 并非互斥，样本仅 1 条，不纳入 |

Strict 合计 **567** 个 cycle（Action 总数 606 的 93.6%）。

### 2.4 谓词别名归一化审计（只审计，不改 manifest）

左侧/右侧方向谓词在 VG 里有多种写法，原始统计会低估其可用上下文：

| 归一化后 | 原始变体 | 合并后词频 |
|---|---|---:|
| `left of` | left of / to left of / on left of / on left side of / left side of / on left | 946 |
| `right of` | right of / to right of / on right of / on right side of / right side of / on right | 873 |

**裸 `left` / `right` 不合并**（多为形容词或名词用法，如 "left shoe"，合并会引入噪声）。

归一化前后对共享类上下文的影响：

| 谓词对 | 归一化前 | **归一化后** | 支持≥20 | 支持≥5 | 最大支持 |
|---|---:|---:|---:|---:|---:|
| left of ↔ right of | 46 | **252** | 0 | 6 | 6 |
| above ↔ below | 206 | 206 | 1 | 7 | 21 |
| over ↔ under | 298 | 298 | 0 | 5 | 11 |
| on ↔ under | 2,176 | 2,176 | **14** | 108 | 95 |
| behind ↔ in front of | 1,483 | 1,483 | **7** | 96 | 105 |

**结论：归一化让左右方向对的可用上下文扩大到 5.5 倍（46 → 252），但支持度仍全部 < 20，
因此维持 Tier B 判定。** 这也说明它此前被判为"无上下文"确实只是口径问题。

**归一化不写入冻结 manifest**：FGBench v1 保存的是原始谓词字符串，
别名只用于统计审计；如需使用，必须在重建时显式应用。

### 2.3 discovered_rules 与 used_rules（两个必须区分概念）

| 概念 | 含义 | 载体 |
|---|---|---|
| **discovered_rules** | train 全量扫描得到的候选及其验证等级（A/B/C/未验证） | `results/rule_mining/final_conflict_rules.json` |
| **used_rules** | 实际生成冻结 FGBench v1 的规则 | 同上 `used_rules` 字段 |

**当前冻结数据用的是全部 6 组空间对 + 4 组动作对，并非只用 Tier A。**
因此 §3 的 Relation 样本中混杂了 Tier B / 未验证规则产生的循环，必须按规则分别报告。

---

## 3. 规模（test 为最终测试集）

**统一定义（务必先读）**：一个互惠 cycle 恒产生 **4 行 = 2 clean + 2 noisy**。
下表每格是 **noisy 行数**（clean 行数恒等于它）。

| 数据源 | 维度 | train | val | test |
|---|---|---:|---:|---:|
| COCO | Entity | 56,000 | 4,582 | 4,596 |
| COCO | Quantity | 5,414 | 238 | 228 |
| VG | Attribute | 65,354 | 4,168 | **28,562** |
| VG | Action | 846 | **4** | 362 |
| VG | Relation | 4,562 | 172 | 1,512 |
| Flickr | Grounding | 410,260 短语 | 13,848 | 13,970 |

换算关系：

```
noisy 行数（上表合计） = 176,600
cycle 数 = 176,600 / 2 = 88,300           # 每个 cycle 贡献 2 条 noisy
总行数   = 176,600 × 2 = 353,200           # 2 clean + 2 noisy
```

### 3.1 逐规则 cycle 数（manifest 实测，非估算）

| 维度 | 规则 | cycles | 占比 |
|---|---|---:|---:|
| COCO Entity | easy / hard | 24,587 / 8,002 | — |
| COCO Quantity | 同类不同计数 | 2,940 | — |
| VG Attribute | 同槽位属性冲突 | 49,042 | — |
| VG Action | sitting on ↔ standing on | **504** | 83.2% |
| | walking on ↔ running on | 63 | 10.4% |
| | riding ↔ walking | 38 | 6.3% |
| | eating ↔ drinking | 1 | 0.2% |
| VG Relation | **Tier A**：on ↔ under | **1,682** | 53.9% |
| | **Tier A**：behind ↔ in front of | **1,204** | 38.6% |
| | Tier B：above ↔ below | 105 | 3.4% |
| | Tier B：over ↔ under | 66 | 2.1% |
| | 未验证：inside ↔ outside | 60 | 1.9% |
| | Tier B：left of ↔ right of | 6 | 0.2% |

**Relation 中 Tier A 占 92.4%（2,886 / 3,123），Tier B 177 个、未验证 60 个。**
任何按维度报告的结果都必须注明这一点，否则会把 Tier B 的样本算进"经数据验证的规则"。

### 3.2 严格子集上的指标（test，复用已有预测，不重训）

| 维度 | 子集 | 规则 | cycles(noisy) | 余弦 | CSLS | CRD | Δ(CRD−余弦) | 2AFC |
|---|---|---|---:|---:|---:|---:|---|---:|
| Relation | **Tier A（严格）** | on↔under、behind↔in front of | 706（1,412） | 0.5413 | 0.5417 | **0.5531** | **+1.18 pp** [0.88, 1.49] | 0.6501 |
| Relation | Tier B（探索） | above↔below、over↔under、left↔right | 40（80） | 0.5680 | 0.5795 | 0.5861 | +1.80 pp [0.12, 3.47] | 0.6750 |
| Relation | 未验证 | inside↔outside | 10（20） | 0.5600 | 0.5775 | 0.5925 | +2.73 pp [−2.00, 8.01] | 0.8000 |
| Relation | 全部 | — | 756（1,512） | 0.5429 | 0.5442 | 0.5549 | +1.19 pp [0.91, 1.49] | 0.6534 |
| Action | **Strict（严格）** | sitting↔standing、running↔walking | 166（332） | 0.5913 | — | 未训练 | — | 0.7831 |
| Action | Conditional | riding↔walking | 15（30） | 0.6733 | — | 未训练 | — | 0.7333 |

**关键：Relation 在严格子集上的 Δ 为 +1.18 pp，与全量的 +1.19 pp 几乎相同**——
说明 CRD 的提升**不是由 Tier B 的 177 个样本撑起来的**。
CRD 从未在 Action 上训练，因此 Action 只报告余弦/CSLS。

即：**上表数字既不是 cycle 数，也不是总行数，而是"噪声侧行数"**，恰好是 cycle 数的 2 倍、
总行数的一半。读者最容易在这里误读，故单列说明。

Flickr 的 438,078 条短语 / 31,780 张图属独立 grounding 任务，**不计入**上述二分类统计。

---

## 4. 与"宽松规则"版本的对比（诚实记录）

| 维度 | 宽松（任意不同谓词） | **严格（仅互斥对）** | 保留率 |
|---|---:|---:|---:|
| Action cycles | 4,669 | **606** | 13% |
| Relation cycles | 21,583 | **3,123** | 14% |
| Attribute cycles | 48,699 | 49,042 | 101% |

也就是说：**Action / Relation 的可用样本在最严格的证据标准下只剩约 1/7**。
这是"不用缺失标注冒充负证据"的直接代价，**不是数据缺陷**，但必须在论文中如实报告。

---

## 5. 审计

| 检查 | 结果 |
|---|---|
| 每图 / 每 caption 正负各一次 | ✅ 0 不平衡 |
| 每 cycle 4 行 / 2+2 / 2 图 2 句 / 同 split | ✅ 88,300/88,300 |
| 跨 split 泄漏 | ✅ 0 |
| 证据完整率 | ✅ 100% |
| **TF-IDF 文本捷径** | **0.5000**（五个 dataset×facet 全部） |

---

## 6. 划分（已核对）

| 数据源 | 协议 | train | val | test |
|---|---|---:|---:|---:|
| COCO | Karpathy（train+restval） | 113,287 | 5,000 | 5,000 |
| VG | **官方 VG150** | **68,538** | **5,000** | **31,876** |
| Flickr30k | 官方 train/val/test | 29,783 图 | 1,000 | 1,000 |

VG150 恢复方式：h5 只存行索引 → 按 `image_data.json` 原始顺序去掉 4 张损坏图
{1592, 1722, 4616, 4617} 得到 108,073 行映射；`split==2` 且有效（≥1 框）为 test，
`split==0` 有效池前 5,000 为 val，其余为 train。**复现值与标准 68,538/5,000/31,876 完全一致。**

---

## 7. 统一字段

```
pair_id, cycle_id, caption_id, dataset(COCO|VG), target_image_id("COCO:123"/"VG:456"),
caption, pair_label(0=clean,1=noisy), facet, fact, evidence{}, split,
text_granularity(image_caption|region_description), region_bbox, target_file, difficulty
```

推理只用 `(image, caption)`；`evidence / region_bbox / cycle_id` 仅用于构造、监督与分析。

---

## 8. 已知边界

1. **Action val 只有 4 条**（2 个循环）→ 该维度无法在 val 上做模型选择，只能报告 test 点估计（362 条）。
2. **Relation val 172 条**，test 1,512 条，统计功效有限。
3. Quantity val/test 238 / 228，CI 较宽。
4. 全部为规则生成的银标签，未经人工核验。
5. Flickr grounding 未与二分类混合，单独评测。

---

## 9. 复现

```bash
python scripts/build_vg150_split_ids.py     # 官方 VG150 → fgbench/splits/
python fgbench/v03/build_facts.py && python fgbench/v03/parse_captions.py
python fgbench/v03/build_cycles_v03.py      # COCO
python fgbench/v04/download_sources.py
python fgbench/v04/build_vg_facts.py
python fgbench/v04/vg_split.py
python fgbench/v04/attach_split.py          # 事实表附 split
python fgbench/v04/build_vg_attribute.py    # 互斥属性槽位
python fgbench/v04/build_vg_rel.py          # 互斥谓词对
python fgbench/v04/build_flickr_grounding.py
python fgbench/v04/merge_fgbench.py

# 规则发现审计（只读 train，不改变数据）
python fgbench/v04/mine_conflict_rules.py       # 频率 / 候选 / 几何验证
python fgbench/v04/audit_buildable_conflicts.py # 可构造 ∩ 对立
python fgbench/v04/generate_rules_json.py       # 规则文件（统计由 manifest 填充）
```

---

## 10. 论文中关于规则发现的表述（可直接引用）

> Rather than assuming that all distinct relationship predicates indicate semantic
> contradictions, we perform a train-only, corpus-wide mining procedure over Visual
> Genome relationship annotations. Candidate predicate pairs are identified through
> shared subject–object category contexts and subsequently examined using
> axis-specific geometric evidence and semantic compatibility constraints. The
> candidates are organized into three validation tiers according to their geometric
> opposition and constructibility. Action conflicts are treated separately through
> semantic rule adjudication because their incompatibility is not reliably
> determined by bounding-box geometry. Only sufficiently supported rules are
> retained for reciprocal-cycle construction.

中文对应：

> 我们不假定"谓词不同即语义矛盾"，而是在训练划分上对 Visual Genome 的关系标注做
> 一次语料级挖掘：先按共享的主体–客体类别上下文筛出候选谓词对，再用分轴的几何证据
> 与语义兼容性约束加以检验，并按几何对立度与可构造性划分为三个等级。动作类冲突
> 单独处理，因为姿态差异无法由边界框几何可靠判定。最终只保留证据充分的规则用于
> 互惠循环构造。

---

## 11. 规则体系（冻结）

| 维度 | 主规则（严格） | 扩展规则（探索） | 排除 |
|---|---|---|---|
| **Action** | sitting on ↔ standing on（504）<br>running on ↔ walking on（63） | riding ↔ walking（38） | drinking ↔ eating（1） |
| **Relation** | **Tier A**：on ↔ under（1,682）<br>behind ↔ in front of（1,204） | **Tier B**：above ↔ below（105）<br>over ↔ under（66）<br>left of ↔ right of（6） | inside ↔ outside（60，未验证） |

四条必须写进论文的说明：

1. **Action 的等级来自语义互斥判定**，不是几何置信度——它的两组 Strict 规则几何签名相同。
2. **Relation 的等级来自全量扫描 + 几何验证 + 语义验证**，Tier B 是"支持度不足"而非"无共享上下文"。
3. **严格子集上的结论**：Relation Tier A 的 Δ(CRD−余弦) = **+1.18 pp** [0.88, 1.49]，
   与全量的 +1.19 pp 一致 → 提升不由 Tier B 驱动。
   但 0.5531 仍接近随机，**只能说明规则子集稳健性，不能说明空间关系错配已被解决**。
4. **FGBench v1 原始版本保留 88,300 cycles / 353,200 rows**，不因子集分析而覆盖；
   严格子集是**质量审计口径**，不是新 benchmark。

**Action 不报告 CRD 结果**：其 val 仅 2 个 cycle，无法在该维度上按统一协议选择 λ。
