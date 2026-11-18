# CONSTRAINTS · 可执行验收条款（C1–C8）

> 本文件把用户的研究方案约束转成**可执行、可被脚本判定的条款**。任何一条不满足，本轮即判为未通过；
> 不允许通过修改条款、改门槛或把论文降级为"分析型/负结果型"来通关。
> 建立时间：2026-11-03。冻结后本文件只允许追加"验证记录"，不允许改动判据文字。

## 0. 一句话目标

> **2026-11-05 已改为基座无关定位，本节仅留档。现行目标见 §2。**

原目标：在单一冻结 CLIP ViT-B/32 双塔上，用一个评分公式 + 一套全局常数、零参数更新，四项硬门槛全部达标。

现行目标：提出一种 **backbone-agnostic 的图文错配检测机制**，只依赖基座提供的标量相关性评分
$s_b(I,c)$，在**同一基座内部**相对其 Cosine 取得可观的独立增益，并在 CLIP 与 FG-CLIP 上同时成立。

## 1. 条款

| 编号 | 条款 | 判定方式（可执行） | 违反示例（直接淘汰） |
|---|---|---|---|
| **C1** | **统一模型（2026-11-05 修订）**：模型结构、前向公式、输入输出与检测机制完全相同；允许每个数据集在**自己的官方 train** 上学习该模型的权重与超参数。$A_d=F_{\mathcal G}(I,c;\Theta_d,\lambda_d)$，$\mathcal G,F$ 固定 | 同一份代码路径服务四列；只允许参数/超参数不同，不允许按数据集切换结构或公式 | **按数据集切换检测机制**（COCO 用 CSLS、FOIL 用 Cosine）、按名称选分支、多检测器拼装后按数据集挑选 —— 这些仍是违规 |
| **C2** | **允许训练共享模块（2026-11-05 修订）**：可训练轻量检测头/适配参数，但只能使用各数据集官方 train；仍禁止引入外部预训练语料做监督 | 训练脚本必须只用官方 train；推理公式与结构四列一致 | 用 test 训练或调参；为某列单独训练专用模块 |
| **C3** | **单骨干 + 有限更新（2026-11-05 修订）**：只允许一个视觉语言基座（CLIP 或 FG-CLIP，二选一作主实验，另一个作泛化验证）；允许 LoRA/部分微调；不得引入第二编码器或 BLIP | 代码中不出现第二种联合编码器；LoRA 配置四列共享 | 用 BLIP-ITM 分数、拼接两个基座、为某列换骨干 |
| **C4** | ~~不以预训练语料池为主线~~ **已于 2026-11-03 由用户解除** | ~~最终分数不得依赖外部语料频次、语言模型困惑度、语料共现统计~~；解除后允许 GPT-2 等语料先验进入公式（仍需过 C1/C7） | 原禁止项已失效；但"按 bench 切换先验权重"仍违反 C1 |
| **C5** | **相对改善门槛（2026-11-05 修订）**：以同一基座的 Cosine 为参照，$G_d=(M_d-B_d)/(1-B_d)\ge 10\%$，且配对置信区间下界 > 0；行级用 AUROC，FOIL 用组内二选一准确率 | 每列各自与其基座余弦比较，不再要求绝对 0.96/0.92 | 用 test 结果反选门槛；只报点估计不报 CI |
| **C6** | FOIL 协议 | val = 官方 train（197,788 对 / 65,697 图）中按**图像互斥**抽 10,000 对（固定 seed，落 manifest + 哈希）；test = 99,480 对**只评一次**；常数哈希与 manifest 不符则脚本拒绝运行 | 用 test 调参、重复评测后择优、val/test 图像重叠 |
| **C7** | 创新性 | 每个候选必须过"归约检验"：若可写成 CSLS、LEMoN `d1+βsn+γsm`、CMC 边际、Deep k-NN 同簇比例、词块最大相似度、ECDF 融合语言先验等的单调变换，即淘汰；结论落 `analysis/reduction_table.md` | 把 `−cos + λ(rx+rc)` 换个名字当新算子 |
| **C8** | 定位 | 交付物必须是"一个达标的方法 + 主表"，不得以"负结果/分析型"作为最终结论 | 用三条不可能性替代达标结果 |

## 2. 指标定义（与既有工程口径逐位一致）

- **行级 AUROC**：`sklearn.metrics.roc_auc_score(label, score)`，score 越高越可疑；正例 = `is_mislabel == 1`。
- **FOIL within-pair accuracy**：同一 `pair_id` 下 `score(negative) > score(positive)` 记 1，`==` 记 0.5，`<` 记 0。
- **置信区间**：行级按行 bootstrap（B=1000，分层保持患病率）；FOIL 按 **pair** 重采样（B=1000）。
- **平局率**：分别报告行级与成对协议的精确平局比例；成对协议同时报告"平局记 0 / 记 0.5"两档。
- **单一分数**：以上两种协议必须调用同一个 `score_pairs()`，不允许任何分支或二次映射。

## 3. 数据与协议（冻结）

| bench | 文件 | 规模（train/val/test） | 门槛 |
|---|---|---|---|
| COCO-cat | `数据集/derived/mscoco/mscoco__cat__lvl0.40__seed0.csv` | 113,287 / 5,000 / 5,000 | AUROC ≥ 0.960 |
| Flickr30k-noun | `数据集/derived/flickr30k/flickr30k__noun__lvl0.40__seed0.csv` | 29,000 / 1,014 / 1,000 | AUROC ≥ 0.960 |
| MM-IMDb-cat | `数据集/derived/mmimdb/mmimdb__cat__lvl0.40__seed0.csv` | 15,552 / 2,608 / 7,799 | AUROC ≥ 0.880 |
| FOIL-COCO | `数据集/FOIL-COCO/pairs_{train,test}.csv` | 197,788 / 10,000(自建 val) / 99,480 | acc ≥ 0.920 |

- 文本截断统一 77 token（MM-IMDb 的 plot outline 会截断，此设定对全部 bench 一致）。
- 参考库只由**官方 train** 构成，且不携带标签；查询自身从库中排除（按 image_id / text key）。
- 任何用到标签的计算只能出现在 `scripts/`（Gate、评测），不得进入 `src/`。

## 4. 自检清单（每条都要有产物）

1. `tests/test_single_config.py`：唯一入口 + 常数哈希稳定 + 无 dataset/protocol 分支键。
2. `tests/test_no_label_leak.py`：`src/` 不出现标签列名；测试脚本未在 test 上做选择。
3. `scripts/leak_check.py`：库重叠比例、自匹配排除、test 单次运行守卫（常数哈希校验）。
4. `analysis/reduction_table.md`：候选公式对已知算子的逐条归约结论。
5. `analysis/gate_report.md`：四列上界与放行/判死依据。
6. `RESULTS.md`：四列数字 + CI + 平局率 + 污染披露，全部由 CSV 渲染。

## 5. 运行环境（唯一）

```powershell
$py = "D:\PyCharm\PythonProject\异常标签检测\LEMoN\LEMoN-main\.venv\Scripts\python.exe"
```

Python 3.10.22 / torch 2.5.1+cu121 / transformers 4.53.2 / numpy 1.26.4 / pandas 2.2.3 / sklearn 1.5.2；
RTX 4060 Laptop 8.6 GB，**单进程串行**（并行两个 GPU 进程会因 WDDM 溢出慢百倍）。

## 6. 定位与准入（2026-11-05 冻结）

### 6.1 研究定位

> A Backbone-Agnostic Mechanism for Image–Text Mismatch Detection.
> 贡献落在**检测原理**，不落在编码器；方法以插件形式作用于任意能稳定输出标量相关性评分的基座。

### 6.2 输入接口（M1）

**2026-11-05 第二次修订（M1′）**：接口从标量升级为**全局归一化嵌入**：

```
encode_image(image)      -> u ∈ R^D,  ||u||=1
encode_text(caption)     -> v ∈ R^D,  ||v||=1
score(u, v)              -> u·v
```

允许不同基座使用自己的编码器，方法只依赖 $(u,v)$ 两个标准化向量。
**仍禁止**：读取 patch / token / attention / 中间层编号，或引入第二个联合编码器。
此前的标量接口（$s_b(I,c)$）作为该接口的特例保留。

### 6.3 五条准入

| 编号 | 要求 | 判定 |
|---|---|---|
| M1 | 基座无关 | 代码只出现 `score(I, c) -> float`；无 `patch`/`hidden_state`/`attn` 读取 |
| M2 | 统一机制 | 四列同一算法、同一常数；不按基座或数据集分支 |
| M3 | 独立增量 | 同基座上优于原始 Cosine 与已复现基线（CSLS / LEMoN / TRACED） |
| M4 | 跨基座有效 | CLIP 与 FG-CLIP 上增益同号；跨基座迁移（不重训检测器）仍为正 |
| M5 | 计算可行 | 每样本基座调用 ≤ 40 次，不调用生成模型 |

### 6.4 验收口径

$$
G_{b,d}=\frac{M_{b,d}^{\text{ours}}-B_{b,d}^{\text{base}}}{1-B_{b,d}^{\text{base}}}\;\ge\;0.10,
$$
要求配对置信区间下界 > 0；行级用 AUROC，FOIL 用组内二选一准确率，二者不得混算。

### 6.5 已被占用、不得作为创新点的空间

| 算子 | 归属 | 我们状态 |
|---|---|---|
| 单标量得分 | CLIP / FG-CLIP | 基线 |
| 对参考池的密度校正 | CSLS / LEMoN | 已测，四列净正 |
| 文本编辑轨迹 + 分类器 | TRACED (ICLR 2026) | 禁止照搬，需边界区分 |
| 图像遮挡 / 局部删除 | CRG / VES | 已测，条件 MCA 0.44–0.49 |
| 表征融合头（MLP/LR/GBM） | 常规做法 | 已测，属校准 |
