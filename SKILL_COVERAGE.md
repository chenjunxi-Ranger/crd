# SKILL_COVERAGE · 阶段与 skill 覆盖对照

> 纪律：每个阶段开始前先确认 skill 是否提供指导；**有则按其指导执行，无则显式写明"无覆盖，按仓库既有纪律执行"**。
> 记录时间：2026-11-03。skill 根目录：`C:\Users\mechrevo\.codebuddy\skills`。

| 阶段 | 计划使用的 skill | 可用性核查（2026-11-03） | 实际执行方式 | 无覆盖时的替代纪律 |
|---|---|---|---|---|
| Stage 0 资产盘点 | 无 | 无 skill 覆盖工程盘点 | 直接读仓库 + 跑只读命令 | 仓库纪律：数字来自 CSV，不手填 |
| Stage 1 文献调研 | `nature-academic-search` | **可用**；按 router 已加载 `manifest.yaml`、`static/core/tools.md`、`static/core/routing-and-ops.md`、`references/workflows/wf1-multi-source-search.md` | MCP 学术检索工具**未挂载**（ToolSearch 无匹配）→ 走 skill 规定的 **No-MCP fallback**：`scripts/academic_search.py`（OpenAlex）+ `scripts/format-converter.py` | 若 OpenAlex 不通：改用 `web_search`/`web_fetch`，并在文档中标注降级来源 |
| Stage 1 引用核验 | `nature-ref-verifier` | **不可用**：目录 `nature-ref-verifier/` 为空（0 文件） | 无 skill 指导 → 按仓库纪律：每条引用给出 DOI/会议，逐条在 OpenAlex/CrossRef 回查标题、年份、作者；冲突项标注并剔除 | 用 `scripts/format-converter.py` 取元数据交叉核对 |
| Stage 1 候选生成 | `scientific-brainstorming` | 可用（6 md + 4 py） | 按其流程做独立生成→对抗评审→决策日志 | — |
| Stage 2 可行性 Gate | 无专用 skill | `nature-statistics` 只覆盖统计口径，不覆盖上界估计 | 自行设计 Gate（逐特征 AUROC、线性/非线性监督融合上界、噪声地板），统计口径部分参照 `nature-statistics` | 仓库纪律：Gate 只出报告，不进 `src/` |
| Stage 3 方法冻结 | `scientific-brainstorming`（决策） + `nature-writing`（方法节表述） | 均可用 | 先写公式与归约检验，再写方法节 | — |
| Stage 4 评测 | `nature-statistics` | 可用（8 md + 1 yaml） | 按其口径核定 bootstrap、平局、多重比较、效应量 | — |
| Stage 5 评审 | `nature-reviewer` | 可用（11 md） | 三审 + 交叉综合；逐条对应 C7/C8 | — |
| Stage 6 出图出表 | `nature-figure` | 可用（31 py + 23 md） | 生成 ROC / 分数分布 / 边际分桶图，导出期刊格式 | — |

## 关键结论（Stage 0 已确认）

1. **MCP 学术检索不可用**：`ToolSearch` 对 `search_papers` / `academic search` / `semantic scholar` 均无返回；
   仅存在 `web_search` / `web_fetch` / `RAG_search` 等。因此 Stage 1 必须走 skill 明确允许的 OpenAlex 回退路径，
   不得假装调用了 MCP。
2. **`nature-ref-verifier` 目录为空**：引用核验环节无 skill 指导，按第 2 行替代纪律执行，并在调研文档中声明。
3. **建模阶段无"建模"专名 skill**：`scientific-brainstorming` 负责候选生成与对抗筛选，`nature-statistics` 负责统计口径，
   实现层面按仓库既有工程规范执行。这一点在交付物中如实披露。
