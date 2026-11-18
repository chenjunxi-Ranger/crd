# METHOD · 单公式冻结（Stage 3）

> 冻结时间：2026-11-03。常数哈希见 `src/constants.py::constants_hash()`。
> 本文件同时记录**冻结依据**与**为什么没有冻结出一个达标公式**——后者是 Gate 的结论，不是省略。

## 1. 唯一公式

对一对 $(x, c)$：

$$
s(x,c) \;=\; -\cos(z_x, z_c) \;+\; \lambda \cdot \bigl(1-\mathrm{cov}(x,c)\bigr),
\qquad \lambda = 0.5
$$

$$
\mathrm{cov}(x,c) \;=\; \frac{1}{|T_c|}\sum_{i\in T_c}
\sigma\!\left(\frac{\max_{p}\, \cos(t_i, v_p) - \tau}{\kappa}\right),
\qquad \tau=0.20,\ \kappa=0.05
$$

- $z_x, z_c$：冻结 CLIP ViT-B/32 的图像池化嵌入与文本池化嵌入（L2 归一化）；
- $t_i$：第 $i$ 个内容词元的文本塔状态（经 `text_projection` 投到联合空间）；
- $v_p$：第 $p$ 个图块状态（经 `visual_projection`）；
- $T_c$：内容词元集合（去停用词、每词只取首个 subword）；
- 分数越高越可疑。行级 AUROC 与 FOIL 成对比较**共用这一个分数**，无任何分支。

常数只有 4 个（$\lambda,\tau,\kappa$ 与截断长度 77），全局唯一，不含任何 dataset/protocol 键。

## 2. 为什么 $\lambda_{\text{witness}}=0$

`scripts/freeze_weights.py` 在 val 上穷举
$(\lambda_{\text{cov}},\lambda_{\text{wit}})\in\{0,0.5,1,2\}^2$，同一套常数同时评 COCO 与 FOIL：

| $\lambda_{cov}$ | $\lambda_{wit}$ | FOIL acc | COCO AUROC |
|---:|---:|---:|---:|
| 0.5 | 0.0 | 0.8900 | **0.9242** |
| 0.0 | 0.0 | 0.8835 | 0.9295 |
| 0.5 | 0.5 | **0.9475** | 0.8415 |
| 1.0 | 0.5 | 0.9450 | 0.8378 |
| 1.0 | 1.0 | 0.9405 | 0.7359 |

任何把 FOIL 推过 0.92 的见证权重都会把 COCO 打到 0.84 附近。**单常数集无法同时满足两列**，
因此冻结折中解 $\lambda_{wit}=0$（这同时避免 C4/C7 争议）。

## 3. 信息源与归约检验

| 项 | 来源 | 与已知算子的关系 | 结论 |
|---|---|---|---|
| $-\cos$ | CLIPScore 的单调变换 | 已知算子 | 作基线项，不声称创新 |
| $\mathrm{cov}$ | token×patch 对齐矩阵的软覆盖率 | 与 CSLS/CMC/LEMoN 不同（不用库、不用边际均值）；与 CLIP4DM 的词级归因不同（无梯度、无最大聚合到词） | 允许，但**不足以补门槛** |
| 见证项 | 数据集 train 描述的 bigram 频率 | 与 VisualGPTScore 的语言先验同族（虽非预训练语料） | 已冻结为 0，不进公式 |

归约检验逐条结论见 `analysis/reduction_table.md`。

## 4. 失败判据（写死，避免事后解释）

1. 四列任一低于门槛 → 本轮未通过，不得以“分析型/负结果型”替代；
2. 若某列达标但需要换常数或换分支 → 视为未达标；
3. 若 val 达标而 test 不达标 → 以 test 为准，且不得重跑 test。

## 5. 已知边界（Gate 已量化）

| 列 | 公式在 val 的表现 | 门槛 | 缺口 |
|---|---:|---:|---:|
| COCO | 0.9242 | 0.960 | −0.0358 |
| FOIL | 0.8900 | 0.920 | −0.0300 |

Flickr / MM-IMDb 的实测值由 `scripts/run_bench.py` 落盘，见 `RESULTS.md`。

## 6. v2.0（C4 与 C7 解除后，第四轮迭代）

$$
s(x,c) = -\cos + \alpha\left(\frac{r_x + r_c}{2} - \cos\right) + \beta\,(-\ell_{\text{GPT-2}}(c)),
\qquad \alpha = 6,\ \beta = 0.05
$$

- $r_x$：图像对参考库描述的 top-10 平均相似度；$r_c$：描述对参考库图像的 top-10 平均；
- $\ell$：GPT-2 平均 token 对数似然（C4 已解除）；
- 只有一个全局常数对 $(\alpha,\beta)$，行级与成对协议共用，无分支。

常数只在 val 上选一次（`scripts/gate_final_grid.py`，网格 $\alpha\in\{2,4,6\}$、$\gamma\in\{0,0.5,1,2\}$、$\beta\in\{0,0.02,0.05\}$），
选择规则为「最大化 min(四列相对门槛的余量)」，未按 bench 分别选。
**冻结哈希：`69f79050e2473142`**（`src/constants_v2.py`）。

## 7. 复现

```powershell
$py = "D:\PyCharm\PythonProject\异常标签检测\LEMoN\LEMoN-main\.venv\Scripts\python.exe"
& $py scripts/run_bench.py --bench coco  --split val
& $py scripts/run_bench.py --bench coco  --split test
& $py scripts/leak_check.py
```
