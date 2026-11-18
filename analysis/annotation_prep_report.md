# 盲标准备：跨种子稳定性修正 + 对照组补齐

> 2026-11-05。实现：`scripts/burden_stable.py`；产物 `results/audit/`、`results/audit/blind/`。

## 1. 修正：均值筛选 ≠ 跨种子稳定性

按你的定义重算。对每个种子 $r$、每个方法 $m\in\{\text{cos},A\}$、每个标签 $y$：

$$
q_{r,m,y}=Q_{0.9}\{b_i^{(r,m)}:y_i=y\},\qquad
H_i^{(r,m)}=\mathbf 1[b_i^{(r,m)}\ge q_{r,m,y_i}]
$$

$$
\text{persistent\_hard}: \sum_{r=1}^{3}H_i^{(r,\text{cos})}H_i^{(r,A)}\ge 2
$$

同法定义 `A_corrected`（$\sum_r H^{\cos}(1-H^{A})\ge2$）与 `A_damaged`（$\sum_r (1-H^{\cos})H^{A}\ge2$）。

| bench | persistent_hard | A_corrected | A_damaged | both_correct |
|---|---:|---:|---:|---:|
| COCO (5,000) | 259 | 232 | 224 | 4,285 |
| Flickr (1,014) | 72 | 36 | 34 | 872 |

（旧值 267/233/233/4,267 与 66/35/36/877 是均值筛选，量级相近但定义不同，已重命名。）

**真实稳定性**：抽样到的 40 个 persistent_hard 中，COCO 有 **29/40** 在三个种子里全部高负担，**11/40** 为 2/3；Flickr 为 **30/40** 与 **10/40**。即绝大多数不是单种子偶然。

## 2. 补齐对照组

新增 `{bench}_both_correct.csv`。它的作用是提供基线占比——只有当一个语义错误类型在困难组显著富集、而在 both_correct 组不富集时，才说明它是 A 失败的成因。

## 3. 盲标材料（已生成，无泄漏）

| 文件 | 内容 |
|---|---|
| `blind/{bench}_blind.csv` | `anon_id, image, caption`（**无 group / 无 label / 无分数**） |
| `blind/{bench}_key.csv` | 私有映射：`anon_id → sample_id, image_id, label, group, n_hard, mean_b_cos, mean_b_A` |

自检结果（COCO 160 / Flickr 150）：

- 盲标表列名仅 3 个，确认不含分组与标签；
- 无重复图像（同一图只出一个对）、无重复 anon_id；
- 各组标签配比约 60/40（clean/noisy），未出现单一标签独占。

抽样规则：组内按标签分层 → 按 `image_id` 去重 → 随机抽取（随机种子固定），未挑选极端案例。

## 4. ΔTPR 表述定稿（按你的修正）

- A 在 **COCO、MM-IMDb** 的三个测试 FPR（1/5/10%）上均取得显著 TPR 增益；
- **Flickr** 仅在 10% 显著（+4.2 [2.3, 7.0]），1% 与 5% 证据不足（5% 下界四舍五入为 −0.0，按未舍入值仍跨零）；
- 「三种子 × 两方向 = 6 次配对评估」不得表述为 6 个独立训练种子；训练随机性仍以三种子各自的点估计与 SD 报告。

## 5. 下一阶段

标注是唯一需要人力的环节。标注完成后按预注册规则比较
$P(\text{semantic type}\mid\text{persistent\_hard})$ 与 $P(\text{semantic type}\mid\text{both\_correct})$，
只有显著富集且可机器计算、控制 $(s,r_I,r_c)$ 后仍有独立增量的语义信号，才进入方法设计。
