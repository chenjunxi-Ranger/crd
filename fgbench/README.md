# FG-ITM Benchmark（第一版 · COCO pilot）

自动化构造的图文错配 benchmark。**不含任何新增人工标注**；所有负样本的 caption 都是
**donor 图像上的原始自然 caption**，未改写、未生成。

---

## 1. 目录与产物

```
fgbench/
├── common.py                 路径、COCO 读取、类别别名、按 image_id 划分
├── 01_build_image_facts.py   图像事实库（类别 + 计数 + 框）
├── 02_extract_caption_facts.py  spaCy 解析 caption（实体提及 / 数量断言）
├── 03_construct_entity.py    Entity 候选（倒排索引求交）
├── 04_construct_quantity.py  Quantity 候选
├── 07_filter_and_label.py    去重、限流、加入 clean 对照、打标签
├── 08_split_dataset.py       落 train/val/test + 泄漏审计
├── 09_benchmark_audit.py     分维度统计、长度捷径检查、donor 复用
├── evaluate/run_eval.py      任意 scorer 的整体 + 分维度评测
└── data/
    ├── intermediate/         image_facts / caption_facts / *_candidates
    └── benchmark/            train/val/test.parquet + audit.json
```

环境变量：`COCO_ANN`（标注目录）、`COCO_IMG`（图片目录）、`FGBENCH_OUT`（输出目录）。

---

## 2. 构造规则（可复现）

**划分先于挖掘**：按 `image_id` 哈希切成 train/val/test，target 与 donor 必须同 split。

**Entity**（目标缺该类，donor 有且 caption 提到它）

1. donor 图在 COCO instances 中该类计数 ≥ 1；
2. donor 的某条原始 caption 经 spaCy 解析后**明确提及**该类（名词短语 + 词形还原 + 别名表）；
3. 目标图该类计数 = 0；
4. **该 caption 提到的其它所有类别，在 donor 图和目标图中都存在**（保证唯一冲突）。

**Quantity**（数量冲突）

1. donor caption 中数词与 donor 图实例数一致，且 ≥ 2；
2. 目标图同类别计数为正且不等于该数词；
3. 其它提及类别在两图中都存在。

**不做**：Omit（"没提到"不等于错）、需要属性/关系的维度（留待 Visual Genome 版本）。

---

## 3. 当前数据规模（COCO val2014 全量挖掘后抽样）

| split | Entity noisy | Entity clean | Quantity noisy | Quantity clean | 合计 |
|---|---:|---:|---:|---:|---:|
| train | 4,000 | 3,425 | 1,820 | 1,401 | 10,646 |
| val | 1,500 | 1,286 | 601 | 445 | 3,832 |
| test | 2,000 | 1,609 | 599 | 444 | 4,652 |

上游候选：Entity 200,009 条、Quantity 12,045 条（已过滤到下表的配额）。

---

## 4. 审计结果

| 项 | 值 | 判读 |
|---|---|---|
| split 冲突（同一图出现在两个 split） | **0** | 无泄漏 |
| 证据完整率（noisy 行含 `evidence.fact`） | **100%** | 每条可追溯 |
| 文本长度差（noisy − clean，词） | Entity +0.41，Quantity −0.09 | 无长度捷径 |
| donor 复用 | 均值 2.32，最大 16 | 有限复用，已按 caption 限 2 次 |

---

## 5. 首轮评测（CLIP ViT-B/32 余弦，test split）

| facet | n(noisy/clean) | AUROC | AUPRC | TPR@1%FPR |
|---|---|---:|---:|---:|
| **Overall** | 2599 / 2053 | **0.9322** | 0.9522 | 0.579 |
| Entity | 2000 / 1609 | **0.9531** | 0.9662 | 0.657 |
| Quantity | 599 / 444 | **0.8624** | 0.9003 | 0.317 |
| **Macro** | — | **0.9078** | — | — |

**核心读法：Quantity 明显难于 Entity（0.862 vs 0.953，差 9 个点）**，
说明这个 benchmark 确实能区分"实体存在性"与"数量一致性"两种能力，
而不是一个被单一捷径支配的集合。

---

## 6. 已知局限

1. **Entity 存在构造捷径**：负样本的冲突是"图里没有 caption 提到的类别"，
   模型若学会类别存在性检测即可拿分。这也是该 facet 分数偏高的原因之一。
   后续应把主体放在**类别存在但数量/属性/关系冲突**的样本上，并分 facet 报告。
2. **只有两个维度**：Attribute / Action / Relation 需要 Visual Genome，
   官方站点 403，HF 镜像 `AsphyXIA/visual-genome` 提供 attributes / relations parquet（约 12.5 GB），
   但缺独立 objects 文件，需额外适配。
3. **银标签**：全部为规则生成的 silver label，不是人工确认的 gold。
4. **仅 COCO val2014**：这是 pilot，不能声称为官方独立测试集。

---

## 7. 复现命令

```bash
python fgbench/01_build_image_facts.py
python fgbench/02_extract_caption_facts.py          # ~8 分钟（spaCy 全量解析）
python fgbench/03_construct_entity.py --cap 200000
python fgbench/04_construct_quantity.py --cap 100000
python fgbench/07_filter_and_label.py
python fgbench/08_split_dataset.py
python fgbench/09_benchmark_audit.py
python fgbench/evaluate/run_eval.py --split test --scorer clip
```
