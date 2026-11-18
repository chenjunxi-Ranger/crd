# FGBench-COCO v0.1 · 有效性审计报告

> 2026-11-06。数据冻结于提交 `fgbench/data/benchmark/{train,val,test}.parquet`。
> 全部审计脚本在 `fgbench/audit/`，汇总结果 `benchmark/final_audit.json`。

## 0. 结论先行

| 审计 | 结果 | 判定 |
|---|---|---|
| 1. 同图配对 | Entity 0.9715 / Quantity 0.9250，可独立计算 | ✅ |
| 2. 文本捷径 | TF-IDF **0.8434（Entity）/ 0.9721（Quantity）** | ❌ **未通过** |
| 2b. 图像单模 | 0.5115 / 0.5040（随机水平） | ✅ 无图像泄漏 |
| 2c. 打乱图像 | 0.5158 / 0.4997（真值 0.9531 / 0.8624） | ✅ 分数确实依赖图文对应 |
| 3. 数量标签 | 三条关系 100% 成立 | ✅ |
| 4. 聚类 bootstrap | 见 §4 | ✅ |

**一句话：benchmark 本身"用图像"是成立的，但它允许一个只看文本的作弊器（尤其 Quantity）刷出高分。**

---

## 1. 同图配对测试（Audit 10）

对每条 noisy 样本，找同一张目标图的 clean 描述，比较 `s(I,c_clean) > s(I,c_noisy)`：

| facet | 配对数 | 配对准确率 | 75%–95% CI |
|---|---:|---:|---|
| Entity | 2,000 | **0.9715** | [0.964, 0.979] |
| Quantity | 599 | **0.9250** | [0.902, 0.946] |

**关键读法**：配对准确率明显高于跨图 AUROC（0.9531 / 0.8624），
说明跨图 AUROC 中被"图像难度差异"拉低了一部分。Quantity 的差距尤其大（0.925 vs 0.862）。

---

## 2. 构造捷径检查（Audit 11–13）

| 对照 | Entity | Quantity | 全量 |
|---|---:|---:|---:|
| CLIP 余弦（参考） | 0.9531 | 0.8624 | 0.9322 |
| **TF-IDF + LR（仅文本）** | **0.8434** | **0.9721** | 0.8494 |
| CLIP 文本向量 + LR | 0.8188 | 0.9449 | 0.8235 |
| 图像向量 + LR | 0.5115 | 0.5040 | 0.5113 |
| 打乱目标图像后的 CLIP | 0.5158 | 0.4997 | 0.5108 |

**三条判读：**

1. **图像侧无泄漏**（0.51≈随机），且打乱图像后 CLIP 分数崩到 0.51 → CLIP 本身确实在用图文对应。
2. **文本侧有严重捷径**：Quantity 上 TF-IDF 单独就能到 **0.9721**，比 CLIP 还高 11 个点。
3. Entity 上 TF-IDF 也有 0.8434，虽低于 CLIP 0.9531，但远高于随机。

### 2.1 根本原因（构造不对称）

| 维度 | noisy caption 的必然属性 | clean caption |
|---|---|---|
| Quantity | **必然含 ≥2 的数词**（构造时要求 caption 数词 = donor 实例数 ≥2） | 从其它图随机取，**多数没有数词** |
| Entity | 必然提及一个目标图没有的类别 | 随机取，词汇分布不同 |

所以模型只需学到"**这句里有数字 → 噪声**"就能在 Quantity 上拿 0.97。这不是数量理解，是词表偏置。

### 2.2 补充：数量差异分布极不均衡

| target→donor 计数 | 样本数 |
|---|---:|
| **1 → 2** | **411 / 599（69%）** |
| 1 → 3 | 32 |
| 3 → 2 | 31 |
| 4 → 2 | 22 |
| 1 → 4 | 20 |
| 其它（共 35 种） | 83 |

即：**"检测到复数"就足以覆盖约七成样本**。

---

## 3. 数量标签正确性（Audit 14）

| 检查 | 通过率 |
|---|---|
| caption 数词 == donor 图实例数 | **100%** |
| caption 数词 ≠ target 图实例数 | **100%** |
| target 图对应类别计数 > 0 | **100%** |

标签本身没有问题，问题在**clean 对照的抽样方式**（见 §2）。

---

## 4. 聚类 bootstrap（Audit 15，1,000 次，按 target_image_id）

| facet | AUROC | 95% CI | 配对准确率 | 95% CI |
|---|---:|---|---:|---|
| Overall | 0.9322 | [0.9250, 0.9392] | — | — |
| Entity | 0.9531 | [0.9467, 0.9589] | 0.9715 | [0.9644, 0.9786] |
| Quantity | 0.8624 | [0.8395, 0.8846] | 0.9250 | [0.9015, 0.9463] |

CI 都很窄（±1 个点），统计上足够支撑"Entity 与 Quantity 存在真实差距"这一结论。

---

## 5. 必须做的修复（在接入 Visual Genome 之前）

**核心修复：clean 对照必须做文本特征匹配（matched controls）。**

| 维度 | 当前 clean 采样 | 应改为 |
|---|---|---|
| Quantity | 随机取任意 clean caption | 取**同样含数词**的 clean caption，且数词与**目标图真实计数一致**、词数与提及数匹配 |
| Entity | 随机取任意 clean caption | 匹配 `n_mentioned` 与长度；优先取"提及类别数相同且全部被图支持"的句子 |

若某条 noisy 样本找不到匹配的 clean，**丢弃该 noisy**，不用不对等的对照凑数。

**次要修复：**

1. Quantity 按计数对（1→2 / 2→3 / …）分层抽样，避免 1→2 占七成；
2. Entity 增加"类别存在但被替换"的构造（entity replace），减少"类别完全缺失"这一类捷径样本；
3. 报告时同时给出 **matched-clean 版**的 TF-IDF 基线，作为捷径上界。

---

## 6. 当前定位（建议措辞）

> FGBench-COCO v0.1：一个由原始 caption 与结构化实例标注自动构造的图文错配基准，
> 覆盖 Entity 与 Quantity 两个维度。标签为规则生成的 silver label，未经人工核验。
> 首版审计发现 clean 对照存在文本特征不匹配问题，Quantity 维度在建（见 §5）。

**在 §5 修复完成前，不应把 Quantity 的 AUROC 解释为"数量理解能力"。**

---

## 7. 复现

```bash
python fgbench/audit/10_paired_test.py
python fgbench/audit/11_text_only.py
python fgbench/audit/12_image_only.py
python fgbench/audit/13_shuffled_image.py
python fgbench/audit/14_quantity_audit.py
python fgbench/audit/15_bootstrap_ci.py
```
