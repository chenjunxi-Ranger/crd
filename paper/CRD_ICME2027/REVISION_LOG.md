# CRD ICME 2027 · 修改日志

> 本轮流程：修改 → 3 子代理评审 → 修改 → 2 审稿人视角 → 记录待补实验；循环 3 次。
> 待补实验一律不在本轮执行，只登记到 §D。

## A · Round 1（正确性修复）

| # | 文件 | 修改 |
|---|---|---|
| 1 | `sec/6_experiments.tex` | 2AFC 的 Action 值由错误的 1.0000 改为 **0.7831**；Entity 保持 1.0000 |
| 2 | `sec/0_abstract.tex` | probe 优势由只写 Attribute +0.4 改为 **+0.4 / +1.5 / +7.6**（Attribute/Quantity/Action），并注明消耗全量标签 |
| 3 | `sec/0_abstract.tex` | 删除与外部结果冲突的“does not improve same-image ranking”，改为“leaves cycle-level 2AFC unchanged by construction” |
| 4 | `sec/6_experiments.tex` | Table II 表注改为“表体=逐种子 AUROC 均值；正文 Δ=三种子集成分数，差 ≤0.05 pp” |
| 5 | `sec/6_experiments.tex` | Quantity 统一为 +5.38（集成）/ +5.32（种子均值） |
| 6 | `sec/6_experiments.tex` | Action 统一为 +0.47（集成）/ +0.43（种子均值），与 CI [−1.71, 2.72] 配套 |
| 7 | `sec/6_experiments.tex` | “neither uses a label or a gradient”改为“no gradient training；k/λ 同一 val 选择” |
| 8 | `sec/6_experiments.tex` | 19/22 后补一句：Entity 的 +0.04 pp 是饱和 facet 上的差异，不作实际增益 |
| 9 | `sec/6_experiments.tex` | 去偏对照明确写“on the integrated score” |
| 10 | `sec/6_experiments.tex` | probe 段明确 0.43 是集成口径、0.47 是种子均值口径 |
| 11 | `sec/6_experiments.tex` | 外部段补“2 个 same-image accuracy + 3 个 row-level AUROC，不可混比；MM-IMDb 是显著负” |
| 12 | `sec/6_experiments.tex` | Setup 补 CSLS 的 λ 与 CRD 同一 val 预算、网格取内部最优 |
| 13 | `sec/5_fgbench.tex` | “tier-A set”统一为“strict tier” |

编译：10 页（修复性文字增加了约 0.3 页；Round 2 负责压回 6 页）。

## B · Round 1 评审结论与落实

三个子代理（统计审计 / 声明-证据 / 结构）与两位审稿人（方法 A：4/10 borderline；
基准 B：6/10 weak accept）的共同结论：证据链完整，但统计口径与“so what”仍是弱点。

已落实的文字修改：
- Table II 表注：24 个非空单元 vs 22 项 Holm 家族，说明两项 Entity 饱和对照被排除；
- 去偏族改为“9/10 显著”，Relation vs ABTT 标为 borderline；
- “3.12–3.81”改为可复现的“3.16–3.81”；
- Entity 的 +0.04 明确为“vs cosine、饱和 facet、非实际增益”；
- 2AFC 最大偏移 0.0104 → 0.0089；
- 摘要 0.2 ms → 0.83 ms（5.5%），并注明标签只用于选一个标量；
- 引言“per-modality component”改为“residual predictable from u,v”，消除与不可识别声明的矛盾；
- 探针段改为“线性可提取信号在交互项”，并承认非线性单侧头可恢复部分信号（0.6944/0.6956）；
- 结论“reproduce none”改为“不复现 Attribute 增益；mean-centred 恢复约一半 Quantity 增益”。

## C · Round 2：压缩到 6 页

| 动作 | 结果 |
|---|---|
| 摘要 330→197 词 | — |
| Related 3 小节→2 段（253 词） | 保留“CSLS 也是加性但来源不同”锚点 |
| Intro 4 段→3 段（约 430 词） | 删除与 §VI 重复的容量/基准预览 |
| Method 删除 Training Protocol（与 §VI-A 重复） | — |
| 实验 2,270→约 1,450 词 | 跨骨干表 15→3 行；删除 Efficiency 与 Scope 小节；外部两段→一段 |
| 结论 275→约 200 词 | — |
| Table I 增加 Test 列 | 暴露 Quantity 114 / Action 166 的功效限制 |
| Fig.1 仍为通栏 | 版面允许，未改单栏 |

编译：**6 页**，无 overfull/undefined。

## D · 待补实验登记（本轮不执行）

## D · 待补实验登记（本轮不执行）

| # | 实验 | 目的 | 成本 | 优先级 | 状态 |
|---|---|---|---|---|---|
| E1 | top-10% precision（最可疑 10% 中噪声比例，CRD vs cosine/CSLS） | 把 +3.8 pp AUROC 翻译成清洗收益，回答 “so what” | 1–2 h（现有逐样本分数） | P0 | 未跑 |
| E2 | CSLS 的 λ 网格在 FGBench 上复核（放宽到 0–30） | 排除 “CSLS 领先是网格截断假象” | 0.5 天 | P0 | 未跑 |
| E3 | 200 对盲标一致率（五 facet 分层） | 银标签的唯一硬证据 | 0.5–1 天 | P1 | 未跑 |
| E4 | 下游清洗缩减版（ρ=0.4、4 臂、2 seed、R@1） | 端到端收益 | 16–24 GPU-h | P1 | 仅有 protocol |
| E5 | 外部表补同源纯文本头行（SugarCrepe 0.8263 / +5.71） | 把 “文本风格捷径” 变成可控结论 | 0.5 天 | P1 | 已有数字，未入表 |
| E6 | FGBench 上原 LEMoN 协议（COCO-cat / Flickr-noun） | 外部效度 | 1 天 | P2 | 未跑 |
| E7 | 单头跨骨干（不重拟合）负结果 | 把 “15/15 需重拟合” 的限制写成发现 | 2 h | P2 | 未跑 |
| E8 | Action 的 GMM/LEMoN/TRACED 三项配对统计 | 统计完整性 | 2 h | P2 | 未跑 |
| E9 | Table III 由硬编码改为读 CSV；bootstrap 提到 10,000 | 复现性 | 2 h | P2 | 未做 |
| E10 | 官方 LEMoN / TRACED 数字参照区间 | 降低 “-style 复现” 质疑 | 1 天 | P2 | 未做 |
| E11 | **标签预算对齐探针**：LR 只用与 λ 相同的 val 子集（Attribute 2,084 cycles）训练 | 回应“λ 也用了 8,336 行标签” | 0.5 天 | **P0** | 未跑 |
| E12 | **机制归因**：a(u)/b(v) 换 ridge 线性头 vs MLP；把 g 回归到 object count / caption 长度 / token 频率 | 解释“线性去偏 0.5 但非线性单头 +1.38” | 2 h | **P0** | 未跑 |
| E13 | **24 单元 Holm 重算**：补 Entity×LEMoN、Entity×TRACED；bootstrap 提到 10,000 | 把 “19/22” 变为可复算的 21/24 | 2 h | **P0** | 未跑 |
| E14 | 2AFC 的整数计数与 CI（Quantity 110/114、Action 130/166） | 2AFC 作为 sanity check 的置信区间 | 1 h | P1 | 未跑 |

> 以上实验在 Round 3 结束时按「是否影响录用」重新排序，不改变本轮正文的事实陈述。

## E · Round 3 评审与最终结论

三个子代理（一致性 / 数字终审 / 图表与复现）与两位审稿人（E：6/10 weak accept；
AC：P0 约 3.5 人日）的结论：

- 一致性：无 undefined、无悬空引用；修了 Table I/III 未 `\ref`、CRD/2AFC 未展开、
  “centred”→“mean-centred”、LR/MLP probe 混称。
- 数字：3.12–3.81（种子均值）修正了此前的口径混用；“narrowest interval”删除；
  去偏 10 项加 facet 标签；Table IV 恢复 cycle/placebo 两行。
- 复现：`krishna2017vg` 的 bib 类型/刊名修正，FOIL 标题改正；正文 3 处 “released”
  仍无链接，投稿前需补仓库/DOI。
- 审稿人 E：**6/10 weak accept**，可投；最关键一条是去偏句的 facet 顺序与
  “over every detector”措辞（已改）。
- AC：投稿前 P0 五项约 3.5 人日；其中 CSLS 全网格复核若反超，摘要需重写。

### 最终 P0（投稿前必须完成）

| # | 实验 | 工时 | 风险 |
|---|---|---|---|
| P0-1 | CSLS λ 全网格 0–30（FGBench 5 facet + 外部三协议） | 1.5 d | 若反超，主结论与摘要需降级 |
| P0-2 | 标签预算对齐探针（LR/MLP 只用 2,084-cycle val） | 0.5 d | 不补则“轻标签”卖点可被推翻 |
| P0-3 | 表格数字从 CSV 重建 + 24 单元 Holm + 10k bootstrap + 2AFC 整数 CI | 1 d | 审稿人无法复算 |
| P0-4 | top-10% precision / 阈值清洗曲线 | 0.5 d | “so what” 缺落地数字 |

### P1（camera-ready 或加分）

机制归因（ridge vs MLP，2 h）；200 对盲标（0.5–1 d）；官方 LEMoN/TRACED 区间（1 d）；
下游清洗 16–24 GPU-h；单头跨骨干负结果（2 h）；重打 release 包（0.5 d）。

### 最终状态

- `main.pdf`：**6 页**，无 overfull / undefined。
- 参考文献：10 条，bib 已修。
- 未解决风险：CSLS 网格、标签预算、无清洗收益数字；这三项决定 weak accept 能否升到 7。

## F · P0 执行结果（2026-11-16）

四项 P0 全部执行，脚本在 `fgbench/experiments/p0_baselines/`，完整报告在
`results/fgbench/p0_baselines/REPORT_p0_final.md`。

| 项 | 结果 | 对论文的影响 |
|---|---|---|
| P0-1 CSLS 0–30 | Quantity λ=0.5 为内部最优，CSLS 0.8891 vs CRD 0.8848（−0.43，ns）；Entity 平 | 不是网格假象；Setup 写明 0–30 复核不改变 Quantity/Entity 排序 |
| P0-2 预算对齐探针 | 同一 8,336 行预算下，LR 在 Attribute 0.6751（−4.39）、Quantity 0.7949（−8.99） | “轻标签”从辩解变成实证；正文加一句 |
| P0-3 24 单元重算 | **21/24 显著**（4 个 GMM 重复 → 17/20 独立）；2AFC 整数计数与 CI 已得 | 摘要 19/22→21/24；表注改为 24 全入族 |
| P0-4 top-10% 精度 | Attribute 84.89% vs cosine 76.80%（+8.04 pp，CI [7.11,8.93]） | 新增 “From ranking to cleaning” 段，回答 so what |

编译仍为 **6 页**。仍未做：200 对盲标、下游清洗、官方 LEMoN/TRACED 数字、release 重打包。

## G · Cycle 1（P0 回填后的正确性复审）

三个子代理（数字 / 叙事 / 版面）与两位审稿人（A：5/10 borderline；B：6/10 weak accept）
的意见已落实：

- CI 由 [3.72,3.96] 改为 **[3.73,3.96]**；bootstrap 均值 3.85→**3.84**。
- Action 改为 **+0.43（CI [−1.75,2.54]）**，删除无 P0 支撑的 +0.47/[−1.71,2.72]。
- top-10% 段改为“最可疑 10% 中 84.9% 为噪声”，并注明其他 facet 不显著（p=0.25–0.37）。
- 摘要删除无出处的 “not surviving Holm”，补“同预算探针 −4.39”。
- 引言 g 的表述改为“z 的加性可预测部分”，与“g 不是 c 的估计量”一致。
- 结论补回 “cycle-level 2AFC”。
- Action MLP +7.62 明确为 **166-cycle test**，λ 在 80 个内部 val cycle 上选。
- seed σ 注明为初始化方差（数据子集固定）；top-10% CI 注明 cycle-clustered。

审稿人 A 的 M1（跨 facet 单阈值可比性）、M2（ridge 线性头 + R²）、M3（Action 预算对齐）
均需新实验，按约定只登记，不执行。

## H · Cycle 2（行文、图表、复现）

三个子代理与两位审稿人（C：6.5/10 weak accept；D：6/10 weak accept）的意见已落实：

- 结论首段压缩，删除与摘要逐字重复的方法描述；补回 Entity 的 seed-mean 限定。
- 摘要“it falls 4.39”改为“the logistic-regression probe falls …”，消除指代歧义。
- Related 的 GMM 前向引用补 Table~\ref{tab:main}；Method 改为“from the two sides
  jointly, with no cross-side interaction term”。
- “decisive”与“not treated as decisive”矛盾改为“consistent”。
- Setup 改为“rows differ in post-processing and auxiliary inputs”，不再暗示只差检测器。
- Table I 表头补 “Test (cycles)”；Table IV 补 “Δ (pp)”；GMM 行改为 “GMM (monotone)”。
- Table III caption 删除无法兑现的 “supplementary material” 承诺。
- 3 处 “released” 改为 “supplementary/benchmark audit”。
- Fig.1 宽度 1.0→0.94，避免上采样。

编译仍为 **6 页**。复现包的旧 PDF、绝对路径、`afriat2026traced` 出处问题记入 §I。

## I · Cycle 3（终稿核对与最终决定）

三个子代理（数字 / 交叉引用 / 投稿清单）与两位审稿人（E：6/10 weak accept；
F：AC，5 项 P0）的意见已落实：

- 3.12–3.81 与 3.84 同句冲突：改为“seed-mean 区间”，并保留 bootstrap 3.84。
- Quantity +5.32 标注为 seed-mean；top-10% 的 p 区间改为 0.245–0.37，Entity 注明 tie。
- GMM 的 Relation +1.00 无当前 CSV 支撑，删除，只保留 Attribute/Quantity 两个值。
- Table III caption 列出 5 backbone × 3 facet，解决 15 cells 不可核验。
- 摘要删除与缩写不符的 “cross-modal score calibration”；§IV 改为
  “CRD: Cross-modal Residual Debiasing”。
- Figure 1 改为 0.94 宽；Table I/IV 补单位；GMM 行标 “(monotone)”。
- `afriat2026traced` 由 ICLR 2026 改为 arXiv preprint 2025（投稿前需补 arXiv 号）。

编译：**6 页**，无 overfull、无 undefined。

### 投稿前 P0（不执行，仅登记）

| # | 实验/材料 | 工时 | 风险 |
|---|---|---|---|
| P0-A | 400 对分层盲标（双人、Cohen κ） | 1 人日 | 银标签是 FGBench 唯一硬伤 |
| P0-B | 官方 LEMoN/TRACED 在 COCO-cat / Flickr-noun 上 3 seed | 1.5 天 | 四行基线全是 -style 复现 |
| P0-C | ridge 线性头 + g 对 object count/长度/token 频率的 R² | 2–4 h | “加性”目前是断言 |
| P0-D | Action 的预算对齐探针（80 val cycles）与 CI | 0.5 天 | +7.62 与 +0.43 口径不可比 |
| P1 | 下游清洗缩减版（16–24 GPU-h）；release 重打包；外部纯文本头行 | 2–3 天 | 从 6 升 7 的杠杆 |

### 最终状态

- `main.pdf`：**6 页**，10 条参考文献，无 overfull/undefined。
- 审稿人 E：**6/10 weak accept，可投**；审稿人 F：P0 五项未做，但不影响“可投”，影响“能否升到 7”。
- 仍未做：盲标、官方实现、机制归因、Action 对齐、下游清洗、release 重打包。
