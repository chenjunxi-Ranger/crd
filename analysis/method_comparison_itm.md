# 同类方法对比 · BLIP-ITM（跨模态融合分数）版

> 2026-11-09。基座 = `blip-itm-base-coco` 的 ITM 分支（`logit_match − logit_mismatch`）。
> 同一批 strict 测试行；λ 只由 val 选；test 只评估一次。

## 1. 哪些基线在这里是"可定义的"

| 方法 | ITM 空间是否可定义 | 说明 |
|---|---|---|
| ITM raw | ✅ | 融合分数本身 |
| GMM | ✅ | 只依赖标量分数；后验是单调变换，**AUROC 必然等于 raw** |
| LEMoN-style | ⚠️ 混合 | 邻居一致性特征来自 ITC 双塔，把 ITM 分数作为附加特征（已在脚本中标注） |
| **CSLS** | ❌ | 需要点积嵌入空间做 top-k 邻域；融合分数没有对应嵌入 |
| **TRACED** | ❌（成本不可行） | 每样本需 ≈25 次**融合**前向；Attribute 57,124 行 ≈ 1.4M 次融合，约 20 GPU·小时 |

因此 ITM 表只能给出三类方法；CSLS / TRACED 仍以 ITC 表为准。

## 2. ITM 主表（AUROC）

| 方法 | Quantity | Attribute | Relation |
|---|---:|---:|---:|
| BLIP-ITM raw | 0.9859 | 0.8207 | 0.6467 |
| GMM detection | 0.9859 | 0.8207 | 0.6454\* |
| LEMoN-style（混合） | 0.9867 | 0.8263 | 0.6488 |
| **CRD（row）** | **0.9885** | **0.8482** | **0.6586** |

\* GMM 与 raw 的 0.0013 差异是后验浮点并列造成的，理论上二者排序完全相同。

AUPRC：raw 0.9882 / 0.8124 / 0.6382；GMM 同；LEMoN 0.9889 / 0.8173 / 0.6367；
**CRD 0.9897 / 0.8441 / 0.6520**。

## 3. CRD 增益（1,000 次 cycle 聚类 bootstrap）

| 维度 | vs raw ITM | vs LEMoN-style |
|---|---|---|
| **Attribute** | **+2.75** [2.65, 2.84] | **+2.18** [2.08, 2.27] |
| **Relation** | **+1.20** [0.89, 1.55] | **+0.99** [0.63, 1.41] |
| **Quantity** | +0.26 [−0.30, 0.84] | +0.17 [−0.38, 0.72] |

Attribute / Relation 显著；Quantity 不显著（该维度 raw 已达 0.986，无空间）。

## 4. ITC vs ITM 直接对比

| 维度 | ITC raw → CRD | ITM raw → CRD | ITM 领先（CRD 后） |
|---|---|---|---:|
| Attribute | 0.7365 → 0.7808 | 0.8207 → **0.8482** | **+6.74** |
| Relation | 0.5898 → 0.6166 | 0.6467 → **0.6586** | **+4.20** |
| Quantity | 0.9766 → 0.9822 | 0.9859 → **0.9885** | +0.63 |

**ITM 在三个维度上都更好**，但 **CRD 的增益更小**（Attribute +2.75 vs +4.43；
Relation +1.20 vs +2.68）。原因与 R² 一致：ITM 分数中可加成分更少
（Relation R² 0.27 vs ITC 0.60），可扣除的偏置也就更少。

## 5. 2AFC（逐位不变性依然成立）

| 方法 | Attribute | Relation | Quantity |
|---|---:|---:|---:|
| ITM raw | 0.936839 | 0.766289 | 0.991228 |
| **CRD** | **0.936839** | **0.766289** | **0.991228** |
| GMM | 0.928857 | 0.742210 | 0.982456 |
| LEMoN-style | 0.934178 | 0.763456 | 0.991228 |

CRD 在融合分数上同样严格不变；GMM 掉 0.80（Attr）/ 2.41（Relation）/ 0.88（Quantity）。

## 6. 口径提醒

1. BLIP 是 **COCO 检索微调**权重，Quantity 属**域内**结果，不能与 CLIP/SigLIP2 零样本并列。
2. ITM 单对延迟 54.42 ms（ITC 33.32 ms），慢 63%。
3. LEMoN-style 在 ITM 表中是**混合**方法（ITC 特征 + ITM 分数），不应与 ITC 表中的纯 LEMoN 混用。
