# Stage 1 · 文献调研（已知算子清单与空白定位）

> 执行日期：2026-11-03。执行依据：`nature-academic-search` skill 的 No-MCP 回退路径
> （`scripts/academic_search.py` → OpenAlex；`scripts/format-converter.py` → CrossRef 元数据核验）。
> **披露**：本机未挂载 academic-search MCP（`ToolSearch` 对 `search_papers` / `semantic scholar` 无返回），
> 因此本次检索**不是** MCP 多库并发检索，而是 skill 明确允许的 OpenAlex 单源回退；Semantic Scholar / Scopus 未覆盖。
> `nature-ref-verifier` 目录为空（0 文件），无 skill 指导，引用核验按仓库纪律手工执行（见 §5）。

## 1. 检索日志

12 组查询（`--limit 12 --year-from 2019`），原始 JSON 落盘于 `analysis/raw/q_*.json`：

| # | 查询 | 文件 |
|---|---|---|
| 1 | noisy correspondence cross-modal retrieval | `q_noisy_correspondence.json` |
| 2 | training-free image-text alignment score vision language | `q_training_free_alignment.json` |
| 3 | token patch alignment CLIP compositional reasoning | `q_token_patch_alignment.json` |
| 4 | hubness correction cross-modal retrieval embedding | `q_hubness.json` |
| 5 | image-text mismatch detection caption | `q_caption_mismatch.json` |
| 6 | contrastive language image pretraining similarity calibration | `q_similarity_calibration.json` |
| 7 | image-text misalignment detection multimodal | `q_misalignment_detection.json` |
| 8 | compositional misalignment CLIP region binding | `q_compositional_misalignment.json` |
| 9 | entity swap negative caption benchmark | `q_swap_benchmark.json` |
| 10 | training-free reranking CLIP retrieval | `q_training_free_rerank.json` |
| 11 | modality gap geometry cross-modal embedding | `q_modality_gap.json` |
| 12 | vision-language model uncertainty misalignment detection | `q_vlm_uncertainty.json` |

## 2. 与我们同任务的检测范式（必须逐条做归约检验）

| 工作 | 年份/会议 | 算子形式（摘要级） | 是否免训练 | 与本方案冲突点 |
|---|---|---|---|---|
| CLIPScore (Hessel et al.) | EMNLP 2021 | `2.5·max(cos(z,t),0)` | 是 | 只是余弦，已被 CLIP-sim 覆盖 |
| LEMoN (Zhang et al.) | ICML 2025 | `d1 + β·s_n + γ·s_m`，邻居校准 | 是 | 已知算子，C7 禁用；且现有 0.9216–0.9456 |
| CLIP4DM | AAAI 2025 | 词级梯度归因聚合（`score_attr` / `score_fclip`） | 是 | 只认"多出来的词"，replace/swap 上失败（0.80） |
| TRACED | ICLR 2026 | 编辑轨迹特征 + GBDT | 否（训练） | 违反 C2 |
| Mismatch Quest | ECCV 2024 | VLM 判断 + 文本解释 | 否（微调） | 违反 C2/C3 |
| FineMatch | ECCV 2024 | 方面级错配检测 + 纠正头 | 否 | 违反 C2 |
| CMC（本仓库） | — | `mean(m_img, m_txt)` 硬负例边际 | 是 | 已知算子，C7 禁用；0.9608/0.9631 |
| SCOPE/SPARC/PACT（本仓库） | — | 池竞争 / 跨度头 / LoRA | 否 | 违反 C2 |

## 3. 可复用的"算子零件"与其归约风险

| 零件 | 代表工作 | 数学形式 | 归约风险 | 结论 |
|---|---|---|---|---|
| 余弦相似度 | CLIPScore | `cos(z,t)` | 已知算子 | 只能作对照项 |
| 邻居校准 / hubness 归一化 | CSLS (Conneau 2018)；NNN (EMNLP 2024, `10.18653/v1/2024.emnlp-main.1257`) | `2cos − r_x − r_t` | **高**：PCPES 已因此被审稿否决 | 禁止作为主项 |
| 检索边际 | CMC / LEMoN | `cos_self − max_j cos` | 高 | 禁止作主项 |
| 词级归因 | CLIP4DM | 梯度 × 嵌入 | 中：需要梯度但免训练 | 可作为辅助诊断，不能作主贡献 |
| 语料语言先验 | VisualGPTScore (ICML 2024) | `log p(c)` 加权 | **直接违反 C4** | 禁止 |
| 模态间隙几何 | "One Geometry, Different Outcomes" (2026, `10.48550/arxiv.2609.36101`) | 锥形/角度统计 | 中 | 仅可作解释，不可作主项 |
| token–patch 相似度矩阵 | CLIP 用于指代分割（`10.18653/v1/2024.naacl-long.258`）；FG-CLIP2（`10.48550/arxiv.2510.10921`） | `S = T W Vᵀ`，`T∈R^{L×d}`、`V∈R^{P×d}` | **低–中**：现有工作用它做定位/分割或需要训练，**尚未见**把它的一阶/二阶统计量直接作为"错标可疑度"的免训练分数 | 唯一未被占据的结构性空白，需在 Gate 中验证信息量 |

## 4. 由调研得到的候选方向（交 Stage 1 ideate 做对抗筛选）

1. **对齐矩阵的覆盖/集中度统计**：`S = T W Vᵀ` 归一化后，文本词元对图像块的"支持度分布"（每词的 max/熵/覆盖率），聚合成句级可疑度。理由：池化余弦把 77×49 的匹配结构压成一个标量，错标在这张矩阵上表现为"某些词找不到支撑"或"支撑分散"。
2. **双向最近邻一致性**：词→块与块→词的互选比例（cycle-consistent fraction），不使用参考库，因此不受 CSLS 归约影响。
3. **绑定位移量**：同一句内两个实体/属性的块归属是否互换（swap_obj 的已知弱点），用矩阵的行列耦合度量，不依赖外部语料。
4. **文本侧上下文残差**：词表示相对其静态嵌入的偏移范数（措辞 vs 语义），用于抑制 SugarCrepe++ 式同义改写假阳性。
5. **免参考库的分数校准**：按句长/token 数做解析式归一（非拟合），规避"每 bench 选参"。

> 注意：方向 1/2 在**同一张图二选一**下不会像图像侧可加项那样恒等抵消（矩阵随文本改变），这是它能进入 FOIL 列的结构性理由。

## 5. 引用核验记录（`nature-ref-verifier` 无内容，手工执行）

| DOI | 回查结果 | 状态 |
|---|---|---|
| `10.18653/v1/2021.emnlp-main.595` | CrossRef 返回标题 "CLIPScore: A Reference-free Evaluation Metric for Image Captioning" | ✅ 一致 |
| `10.1007/978-3-031-72998-0_18` | CrossRef 返回 "Mismatch Quest: Visual and Textual Feedback for Image-Text Misalignment" | ✅ 一致 |
| `10.1007/978-3-031-72673-6_26` | CrossRef 返回 "FineMatch: Aspect-Based Fine-Grained Image and Text Mismatch Detection and Correction" | ✅ 一致 |
| `10.18653/v1/2024.emnlp-main.1257` | CrossRef 首次请求 SSL EOF（代理抖动），待重试 | ⚠️ 未完成 |
| `10.18653/v1/2023.conll-1.24` | 同上 | ⚠️ 未完成 |

> 未完成项在 Stage 3 前补齐；在补齐前，正文不得引用其卷期页码，只能标注"待核验"。

## 6. 对四门槛的直接含义

- COCO 0.96 / Flickr 0.96：现有免训练最高（CLIP-sim）为 0.9352 / 0.9437，缺口 +2.5 / +1.6 点；
  调研显示**唯一未被用过的信息源是 token–patch 二维结构**，其余都是池化向量的再加工。
- MM-IMDb 0.88：缺口 +2.8 点，且文本被 77 token 截断。该列必须靠"部分文本也有支撑"的局部统计，
  而不是整句池化；这正好与方向 1/2 一致，Gate 将优先验证。
- FOIL 0.92：CLIP-sim 0.8916，缺口 +2.8 点；图像侧项在此协议下恒等抵消，方向 1/2/3 是少数能改变
  成对排序的结构性信号。

## 7. 阶段性结论

调研未发现"冻结 CLIP + token–patch 统计 + 单一公式"已被发表为错标检测器的工作；
但**该方向与 CLIP 指代分割/细粒度对齐工作在算子层面相邻**，因此 C7 的归约检验必须写清
"我们用的是对齐矩阵的分布统计量，而非定位/分割输出，也非梯度归约"，否则仍可能被判为已知算子复用。
