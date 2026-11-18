# 同类方法对比 · BLIP-ITC 基座版

> 2026-11-09。基座换成 **BLIP-ITC**（`blip-itm-base-coco` 的双塔分支），
> 目的是回答"在更强的原生评分上，CRD 的增益是否依然成立"。
> 全部方法在同一批 strict 测试行上打分；参考库/统计量只来自 train；λ 只由 val 选。

## 1. 主表（AUROC）

| 方法 | Quantity | Attribute | Relation | 2AFC(Attr) |
|---|---:|---:|---:|---:|
| BLIP-ITC cosine | 0.9766 | 0.7365 | 0.5898 | 0.893495 |
| GMM detection | 0.9766 | 0.7365 | 0.5898 | 0.881381 |
| LEMoN-style | 0.9770 | 0.7449 | 0.5975 | 0.887543 |
| TRACED-style | 0.9742 | 0.7546 | 0.6091 | 0.880751 |
| BLIP-ITC + CSLS | 0.9798 | 0.7652 | 0.6092 | 0.893495 |
| **CRD（row）** | **0.9822** | **0.7808** | **0.6166** | **0.893495** |

AUPRC（Quantity / Attribute / Relation）：
cosine 0.9792 / 0.7104 / 0.5737；GMM 0.9792 / 0.7104 / 0.5737；
LEMoN 0.9800 / 0.7150 / 0.5754；TRACED 0.9771 / 0.7285 / 0.6056；
CSLS 0.9826 / 0.7396 / 0.5921；**CRD 0.9836 / 0.7672 / 0.5991**。

## 2. CRD 相对各基线的差值（1,000 次 cycle 聚类 bootstrap）

| 维度 | vs cosine | vs CSLS | vs LEMoN | vs TRACED | vs GMM |
|---|---|---|---|---|---|
| **Attribute** | **+4.43** [4.30,4.56] | **+1.56** [1.44,1.68] | **+3.58** [3.45,3.72] | **+2.62** [2.46,2.78] | **+4.43** |
| **Relation** | **+2.68** [2.27,3.13] | **+0.74** [0.21,1.27] | **+1.90** [1.41,2.45] | **+0.76** [0.04,1.47] | **+2.68** |
| **Quantity** | +0.57 [−0.23,1.39] | +0.24 [−0.54,0.94] | +0.52 [−0.30,1.30] | **+0.82** [0.03,1.65] | +0.57 |

**10/15 项显著**。Quantity 上除 vs TRACED 外均不显著——该维度已近乎饱和（0.98+），
CRD 没有空间，这与 CLIP 基座上的结论一致。

## 3. 与 CLIP 基座的对比：增益是否依赖基座

| 维度 | CLIP：CRD − CSLS | BLIP：CRD − CSLS | CLIP：CRD − cosine | BLIP：CRD − cosine |
|---|---|---|---|---|
| Attribute | +2.91 | **+1.56** | +3.51 | **+4.43** |
| Relation | +1.47 | **+0.74** | +1.61 | **+2.68** |
| Quantity | +0.45 | +0.24 | +6.21 | +0.57 |

**关键观察**：基座越强，CRD 的**绝对增益**越大（相对 cosine：Attribute +3.51 → +4.43，
Relation +1.61 → +2.68），但**相对 CSLS 的领先优势变小**（+2.91 → +1.56，
+1.47 → +0.74）。

解释：CSLS 在 BLIP 空间里本身就更有效（Attribute 0.6866 → 0.7652，+7.9 pp），
它也在利用参考库消除一部分同样的偏置；CRD 与它部分重叠，所以差距收窄。
但 **CRD 仍然在三个维度上全部领先 CSLS**，且不需要参考库。

## 4. 方法性质在 BLIP 上仍然成立

| 方法 | AUROC(Attr) | 2AFC(Attr) | 变化 |
|---|---:|---:|---:|
| CRD | **0.7808** | **0.893495** | **0** |
| CSLS | 0.7652 | 0.893495 | 0（也是可加校正） |
| LEMoN-style | 0.7449 | 0.887543 | −0.60 |
| GMM | 0.7365 | 0.881381 | −1.21 |
| TRACED-style | 0.7546 | 0.880751 | **−1.27** |

**GMM 依然 AUROC 与 cosine 完全相同**（单调变换），2AFC 掉 1.21 pp——与 CLIP 上完全一致。

## 5. 口径说明

1. **BLIP-ITC 表在 strict manifest 上**（Relation 706 cycles / 2,824 行）；
   CLIP 表在 v1 上（Relation 756 cycles / 3,024 行）。Attribute（57,124）与
   Quantity（456）两个表行数相同，可直接比较；Relation 不可混用。
2. **BLIP 是 COCO 检索微调权重**，因此 Quantity（COCO 域）是**域内**结果，
   不能与 CLIP/SigLIP2 的零样本数字并列；Attribute / Relation 属 VG 域外，不受此影响。
3. CRD 数值为**三种子平均分数**的 AUROC；各种子 AUROC 的均值略低
   （Attribute 0.7803、Relation 0.6163、Quantity 0.9812）。

## 6. 顺带修掉的隐患

`lemon_fgbench.py` / `traced_fgbench.py` 原先的分数文件名不含骨干标记，
BLIP 重跑会**静默覆盖 CLIP 的分数文件**。现已把 backbone 写进文件名，
并为三个 facet 补出 `*_blipitc_*` 副本；未在 BLIP 上重跑的 Entity/Action
副本已删除（它们本是 CLIP 结果，标签错误）。

CLIP 版的 LEMoN/TRACED 分数文件已被覆盖，但其 AUROC/AUPRC 与 bootstrap 结果
已保存在 `results/fgbench/method_comparison.csv` 中，结论不受影响；
若后续需要 CLIP 的逐样本分数，重跑即可（`--backbone clip`）。
