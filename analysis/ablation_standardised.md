# 消融实验（标准化口径，`crd_backbone` 正式实现）

> 2026-11-09。CLIP ViT-B/16，VG Attribute，FGBench v1-strict，8,000 cycles / 1,500 steps，
> 每格 3 个训练种子。标准化、训练目标、推理公式**未做任何修改**（见 `impl_audit.md`）。

## 1. 协议冻结（第一阶段）

| 项 | 修复内容 |
|---|---|
| 缓存与训练种子解耦 | `load()` 的 train 子集抽样改用常量 `DATA_SEED = 0`，不再用 `args.seed` |
| 固定数据与顺序 | 子集由 `DATA_SEED` 决定；行序沿用 manifest 顺序；三种子只改变初始化与 batch 采样 |
| 溯源文件 | 每次运行写 `crdbb_prov_*.json`：manifest sha、cache sha、train order sha、backbone/facet/parts/target/seed、mu、sd、λ、行数、超参 |
| `crd_blip` 缺 λ | checkpoint 现保存 `lam`（原来只有 `state/mu/sd/dim`，外部推理会回退成 1.0） |
| 公式 | **未改**：`z=(s−μ)/σ`、`m̂=a(u)+b(v)+b0`、`z'=z−λm̂` |

## 2. 消融主表（三种子，指标从逐样本分数重算）

| 配置 | AUROC | SD | AUPRC | 2AFC | Δ vs 基座 |
|---|---:|---:|---:|---:|---:|
| Base（CLIP cosine） | 0.6806 | — | 0.6680 | 0.828163 | — |
| Image-only `a(u)` | 0.6944 | 0.0001 | 0.6807 | 0.828163 | **+1.38** |
| Text-only `b(v)` | 0.6956 | 0.0008 | 0.6904 | 0.828163 | **+1.50** |
| **Additive（row，官方）** | **0.7187** | **0.0001** | **0.7177** | **0.828163** | **+3.81** |
| Cycle-level | 0.7125 | 0.0001 | 0.7077 | 0.828163 | +3.19 |
| Shuffled placebo | 0.6806 | 0.0000 | 0.6680 | 0.828163 | **0.00** |

**超可加性**：1.38 + 1.50 = 2.88 < 3.81，缺口 **+0.93 pp**（旧口径为 +0.77）。
两侧信息互补，不可互相替代。

**2AFC 在全部五个配置下逐位相同（0.828163）** —— 包括安慰剂，因为它同样是可加校正。

## 3. 配对对照（cycle 聚类 bootstrap，500 次 × 3 种子）

| 对比 | Δ | 逐种子 |
|---|---:|---|
| Additive − Image-only | **+2.43 pp** | 2.42 / 2.45 / 2.43 |
| Additive − Text-only | **+2.31 pp** | 2.22 / 2.37 / 2.35 |
| **Additive − Cycle-level** | **+0.62 pp** | 0.62 / 0.64 / 0.60 |
| Additive − Shuffled | **+3.81 pp** | 3.82 / 3.82 / 3.80 |

三种子方向完全一致，极差 ≤0.15 pp。

## 4. 核验结果（第三阶段）

| 检查 | 结果 |
|---|---|
| 1. 三种子 manifest / cache / train-order 哈希 | **全部相同**（5 臂 × 3 种子） |
| 2. μ、σ 只来自 train | 用缓存复算 `S_train.mean()/std()`，与 prov 记录 **差 0.00e+00** |
| 3. λ 是否为 val AUROC 的 argmax | 13/13 个可反解运行 **OK**；2 个 λ=0 的运行无法反解（见下） |
| 4. AUROC / AUPRC / 2AFC | 全部由保存的逐样本分数重算，与运行日志一致 |
| 5. 配对 bootstrap | 见 §3 |

**关于两个 λ=0 的安慰剂运行**：val 上最优 λ 就是 0，即"不校正"。
这本身就是安慰剂成立的直接证据——**val 集自己就判定该头部无校准价值**，
因此 test AUROC 等于基座（0.6806）不是偶然，而是 val 选参的结果。

## 5. 与旧口径（`crd_offset`，未标准化）的差异

| 配置 | 旧（未标准化） | 新（标准化） | 差 |
|---|---:|---:|---:|
| Image-only | 0.6934 | 0.6944 | +0.10 |
| Text-only | 0.6944 | 0.6956 | +0.12 |
| Additive | 0.7149 | 0.7187 | +0.38 |
| Cycle-level | 0.7096 | 0.7125 | +0.29 |
| Shuffled | 0.6796 | 0.6806 | +0.10 |

**结论方向完全不变**（超可加、row > cycle、安慰剂归零），只是绝对值整体上移。
旧结果保留、不进最终主表。

## 6. 产物

| 文件 | 内容 |
|---|---|
| `results/fgbench/crdbb_prov_clip_VG_Attribute_*_s{0,1,2}.json` | 15 份溯源 |
| `results/fgbench/crdbb_scores_clip_VG_Attribute_*_s{0,1,2}.npz` | 15 份逐样本分数（含 val） |
| `results/fgbench/crdbb_clip_VG_Attribute_*_s{0,1,2}_head.pt` | 15 个 checkpoint（含 mu/sd/λ） |
| `results/fgbench/ablation_standardised.csv` | 上表 §2 |
| `fgbench/evaluate/ablation_report.py` | 五项核验 + bootstrap |
| `scripts/rebuild_prov.py` | 从产物重建溯源（哈希内容计入，非重算指标） |
