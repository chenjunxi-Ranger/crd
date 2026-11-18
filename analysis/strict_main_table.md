# FGBench v1-strict · 主表

> 2026-11-08。只保留 Action Strict（2 组）与 Relation Tier A（2 组）作为正式冲突协议。
> 原始 v1（88,300 cycles）作为历史审计版本保留，未被覆盖。

## 0. 两个版本

| 版本 | cycles | rows | 冲突规则 |
|---|---:|---:|---|
| v1-original | 88,300 | 353,200 | Action 4 组 + Relation 6 组 |
| **v1-strict（正式）** | **88,024** | **352,096** | **Action 2 组 + Relation 2 组** |
| 被移除 | 276 | 1,104 | riding|walking 38、drinking|eating 1、above|below 105、over|under 66、left|right 6、inside|outside 60 |

**被移除占比 0.31%（276 / 88,300）**，其余 99.69% 的样本完全不变。

## 1. 严格版规模

| 维度 | cycles | train | val | test |
|---|---:|---:|---:|---:|
| COCO Entity | 32,589 | 28,000 | 2,291 | 2,298 |
| COCO Quantity | 2,940 | 2,707 | 119 | 114 |
| VG Attribute | 49,042 | 32,677 | 2,084 | 14,281 |
| VG Action (Strict) | **567** | 399 | 2 | 166 |
| VG Relation (Tier A) | **2,886** | 2,099 | 81 | 706 |
| **合计** | **88,024** | 65,882 | 4,577 | **17,565** |

## 2. 主表（test，CLIP ViT-B/16）

| 数据源 | 维度 | 余弦 | CSLS | LEMoN-style | TRACED-style | **CRD v1** |
|---|---|---:|---:|---:|---:|---:|
| COCO | Entity | 0.9991 | — | — | — | — |
| COCO | Quantity | 0.8305 | **0.8882** | — | — | 0.8852 |
| VG | Attribute | 0.6806 | 0.6866 | 0.6843 | 0.6875 | **0.7104** |
| VG | Action (Strict) | 0.5913 | — | — | — | 未训练 |
| VG | Relation (Tier A) | 0.5413 | 0.5417 | 0.5436 | 0.5363 | **0.5529** |

## 3. 2AFC（互惠循环内相对判别）

| 维度 | 余弦 | CSLS | LEMoN | TRACED | CRD |
|---|---:|---:|---:|---:|---:|
| Attribute | 0.8282 | 0.8282 | 0.8268 | **0.8186** | **0.8282** |
| Relation | 0.6501 | 0.6501 | 0.6487 | **0.6246** | **0.6501** |
| Quantity | 0.9649 | 0.9649 | — | — | 0.9649 |

## 4. 与全量版的对照

| 维度 | 全量 v1 | 严格 v1 | 变化 |
|---|---:|---:|---:|
| Relation 余弦 | 0.5429 | 0.5413 | −0.16 pp |
| Relation CSLS | 0.5442 | 0.5417 | −0.25 pp |
| Relation CRD | 0.5549 | 0.5529 | −0.20 pp |
| Attribute CRD | 0.7096 | 0.7104 | +0.08 pp |

**去掉 7.6% 的低置信规则样本后，各方法在 Relation 上的绝对分数几乎没有变化
（−0.16 ~ −0.25 pp），CRD 相对余弦的优势保持 +1.18 pp。**
说明原来的 Relation 结果不是由 Tier B 样本撑起来的。

## 5. 仍然缺失

| 项 | 原因 |
|---|---|
| Action 的 CRD | val 仅 2 个 cycle，无法按统一协议选 λ，故不报告 |
| Entity 的 CSLS/LEMoN/TRACED | Entity 已饱和（0.9991），无区分意义 |
| Quantity 的 LEMoN/TRACED | 未运行（可补，约 10 分钟） |

## 6. 审计状态（严格版）

| 检查 | 结果 |
|---|---|
| 每 cycle 4 行 / 2 clean + 2 noisy | ✅ 88,024 / 88,024 |
| 每图 / 每 caption 正负各一次 | ✅ 0 不平衡 |
| 跨 split 泄漏 | ✅ 每张图 / 每条 caption 的跨 split 归属数 = **1**（即无泄漏） |
| TF-IDF 文本捷径 | ✅ **0.5000**（五个 `dataset×facet`） |
| 证据完整率 | ✅ 100% |
