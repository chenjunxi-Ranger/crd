# REVIEW v2 · 完整三审 + 交叉综合（nature-reviewer 口径）

## Review setup

- **Input scope**：`METHOD.md`（v1.0 与 v2.0）、`RESULTS.md`（四轮迭代、CI、种子复核）、
  `analysis/gate_report.md`（十类信息源上界）、`analysis/reduction_table.md`；无完整稿件、无图表正文。
- **Assessment boundary**：只评「方法—证据—主张」一致性；不评写作、不评图表美观度；
  约束集以 2026-11-04 状态为准（C4、C7 已由作者解除，C1/C2/C3/C5/C6/C8 仍生效）。
- **Shared manuscript claim summary**：
  1. 冻结 CLIP 双塔上存在一个单公式、单常数、免训练的错标打分器；
  2. 加入库侧归一化（CSLS 形式）后 COCO/Flickr 可达门槛；
  3. 加入语言模型先验后 FOIL 可达门槛；
  4. 两者在单常数下互斥，MM-IMDb 与 FOIL 仍不可达。
- **Visible evidence base**：
  - v2.0（α=6, β=0.05, k=10，哈希 `69f79050e2473142`）四列 val：
    COCO 0.9603 [0.9549, 0.9656]、Flickr 0.9641 [0.9509, 0.9753]、
    MM-IMDb 0.8641 [0.8509, 0.8770]、FOIL 0.9020 [0.8960, 0.9075]；
  - 种子/划分复核：COCO 0.9581、0.9576（不过）；Flickr 0.9778、0.9779（过）；
    FOIL 0.9018、0.8986（不过）；
  - Gate：无库融合上界 0.9245；令牌×图块 0.63/0.70；注意力加权 0.50（退化）；
    LOO 0.88/0.70；库侧秩 0.9556/0.8770；见证词频 0.83；GPT-2 先验 0.93（FOIL）。
- **Missing materials affecting confidence**：
  - test 结果（按 C6 未消耗）；
  - MM-IMDb 的第二、第三个噪声种子（数据侧不存在 seed1/2）；
  - 与 CMC/LEMoN 的同管线重算；
  - SugarCrepe / SugarCrepe++ 两列；
  - 人工抽检与错误类型分解。

---

## Reviewer 1（强调创新性与定位）

- **Overall assessment**：在解除 C4/C7 后，工作从"不可达的负结果"变成一个**可运行的组合方案**，
  但方法学贡献仍是"已知算子的线性组合"，创新性主张必须收缩。
- **Who would be interested, and why**：做图文数据清洗与基准构建的团队——因为论文给出了
  「哪些信息源能推到哪一列」的定量边界，这比再提一个 SOTA 更有用。
- **Major strengths**：
  1. 单公式、单常数、无分支，工程上可审计（哈希贯穿全部结果）；
  2. 四轮迭代每轮都有可复现的否决理由，没有把失败包装成成功；
  3. 帕累托前沿被量化（MM-IMDb 上限 0.8688 vs FOIL 上限 0.9346），这是任务结构性的。
- **Major concerns**：
  1. CSLS 与 GPT-2 先验都是已知算子；解除 C7 只是让"合规性"消失，没有让"新意"出现；
  2. 论文若声称"方法"，审稿人会问"这与 PCPES 的区别是什么"——目前答案是"常数选择方式不同"；
  3. COCO 的达标依赖 seed0，换两个种子就掉线，这会直接削弱"达标"叙事。
- **Technical failings that need to be addressed before the case is established**：
  - 必须给出与 PCPES/CMC 的**同管线**对照（同库、同 k、同协议）；
  - 必须把贡献重新定位为"约束下的最优组合 + 不可达边界"，而不是新算子。
- **Assessment against Nature-style criteria**：originality 低–中；technical soundness 高；
  significance 中（边界结论有价值）。
- **Recommendation posture**：作为方法论文不足；作为"数据审计的可行性与边界"论文可考虑。

---

## Reviewer 2（强调统计与证据完整性）

- **Overall assessment**：统计纪律明显改善（CI、种子、平局率、哈希），但**"达标"在统计上不成立**。
- **Who would be interested, and why**：基准构建者与评测方法研究者；他们会引用"CI 下界不过线"这一做法。
- **Major strengths**：
  1. 四列都给了 bootstrap CI，且用三个 bootstrap 种子验证区间稳定；
  2. 划分/噪声种子复核做了，且如实报告 COCO 掉线；
  3. FOIL 平局率 0.0000，避免了历史上级联裁决的歧义。
- **Major concerns**：
  1. **COCO 与 Flickr 的 CI 下界都低于门槛**（0.9549 / 0.9509），按点估计宣称达标属于选择性报告；
  2. COCO 3 个种子里 2 个不过（0.9581 / 0.9576），Flickr 3 个都过（0.9641 / 0.9778 / 0.9779），
     两列的稳健性完全不同，不能并列陈述；
  3. MM-IMDb 只有 seed0，数据侧不存在 seed1/2，因此该列的方差**不可评估**，CI 只反映抽样误差；
  4. FOIL 的三个划分都是 5k/10k 对子集，未做全量 99,480 对的一致性检查。
- **Technical failings that need to be addressed before the case is established**：
  - 主表必须同时给点估计与 CI 下界，且以 CI 下界作为达标判据；
  - MM-IMDb 需要补充至少一种方差来源（bootstrap 之外，例如 bank 抽样）；
  - 需要报告 COCO 的 seed 间方差，而不是只报 seed0。
- **Assessment against Nature-style criteria**：technical soundness 高；evidence completeness 中–高。
- **Recommendation posture**：结论可信（1/4 稳健达标）；作为论文需按上述改表。

---

## Reviewer 3（强调机制与可证伪性）

- **Overall assessment**：机制解释是本轮最强项，但**仍未形成可证伪的统一命题**。
- **Who would be interested, and why**：研究"冻结双塔能做什么/不能做什么"的人；组合性基准作者。
- **Major strengths**：
  1. 四轮迭代各自给出否决点，且第四轮的帕累托前沿有明确数值；
  2. 指出 FOIL 需要"词级证据"、MM-IMDb 需要"库侧竞争"，两者在同一常数下互斥；
  3. 注意力加权退化（0.5000）被如实记录，没有被隐藏。
- **Major concerns**：
  1. "互斥"目前是**网格观察**，不是命题；需要写成"在单常数线性组合下，FOIL 与 MM-IMDb 的最优权重区间不交"；
  2. 库侧项的 k=10 未做敏感性（k=5/20/50），因此 0.8641（MM-IMDb）可能不是该族上限；
  3. 语言先验的 λ 只在 {0,0.02,0.05} 三点采样，0.05 附近的最优值未被定位；
  4. 未讨论"如果允许按协议切换权重"（违反 C1）能把四列推到多少——这是读者最想知道的反事实。
- **Technical failings that need to be addressed before the case is established**：
  - 把互斥写成命题并给反例边界；
  - 补 k 与 λ 的二维敏感性；
  - 给"解除 C1"的反事实上界。
- **Assessment against Nature-style criteria**：originality 中（边界结论为真贡献）；significance 中。
- **Recommendation posture**：作为"边界分析 + 可行组合"论文可考虑，但必须补上述三项。

---

## Cross-review synthesis

### Consensus strengths
1. 评测纪律（CI、种子、平局、哈希、test 未消耗）达到可复现标准；
2. 四轮迭代全部如实报告，没有把 2/4 包装成成功；
3. Gate 的十类信息源上界构成了可引用的边界证据。

### Consensus technical risks
1. **统计上只有 Flickr 稳健达标**（COCO 靠 seed0，Flickr 三种子都过）；
2. 方法学贡献是已知算子的组合，CSLS + LM 先验都不新；
3. 关键敏感性缺失：k、λ 网格、MM-IMDb 方差来源、与 CMC 的同管线对照。

### Where emphasis differs across reviewers
- R1：定位与创新性（组合方案 vs 新算子）；
- R2：达标判据（CI 下界、种子方差、MM-IMDb 方差不可评估）；
- R3：机制命题化（把帕累托前沿写成可证伪命题）与缺失的反事实。

### Broad-interest / significance readout
- 直接读者：多模态数据清洗、数据审计、基准构建；
- 广泛意义：中等偏低。价值集中在"冻结双塔 + 单常数能推到哪一列"的边界，而非新 SOTA。

### Most important issues to resolve before a strong case is established
1. 主表改用「CI 下界 + 种子方差」判达标，并据此把结论改为 **1/4 稳健达标（仅 Flickr）**；
2. 补 k∈{5,10,20,50} 与 β∈{0,0.02,0.03,0.05} 的二维敏感性，确认 MM-IMDb 0.8688 是否真是族上限；
3. 把互斥写成命题，并给出"解除 C1"的反事实数字（按协议切换权重时的四列上限）；
4. 与 CMC/LEMoN 同管线重算，替换跨管线引用。

## Risk / unsupported claims

| 主张 | 判定 |
|---|---|
| "COCO 达标（0.9603）" | ❌ 不稳健：CI 下界 0.9549，seed1/2 为 0.9581/0.9576 |
| "Flickr 达标（0.9641）" | ✅ 稳健：CI 下界 0.9509 仍偏低，但三种子 0.9641/0.9778/0.9779 均过线 |
| "2/4 通过" | ⚠️ 点估计口径成立，稳健口径应为 **1/4** |
| "单公式单常数" | ✅ 哈希 `69f79050e2473142`，src 无分支 |
| "免训练" | ✅ 未更新任何参数（GPT-2 与 CLIP 均冻结前向） |
| "MM-IMDb 0.8641 是族上限" | ⚠️ 仅 k=10、β 三点采样下的上界观察，未做敏感性 |
| "FOIL 不可达" | ✅ 三个划分 0.9020/0.9018/0.8986，稳定低于 0.92 |
| "无需 test" | ⚠️ 合理但需正文声明；严格判定仍以 test 为准 |
