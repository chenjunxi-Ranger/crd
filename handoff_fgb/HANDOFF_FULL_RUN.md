# TRACED 完整 FGBench 五维复现 — 交接文档

版本：2026-11-17。目标：在同伴的 Linux/GPU 服务器上，用**官方 TRACED + BLIP**
跑**完整** FGBench 五个 facet，产出可和 `-style` 主表对比的 test AUROC。
本地只跑过 500/100/100 的 pilot；**不要**把 pilot 数字当正式结果。

> 交接包位置：`handoff_fgb/`。冻结划分在 `handoff_fgb/data/fgbench_v1_strict/`，
> 官方代码在 `handoff_fgb/traced_code.tar.gz`，适配器在 `handoff_fgb/code/`。
> 先读 `handoff_fgb/README.md` 的快速开始，再按本文档执行。

---

## 1. 服务器要求

| 项 | 要求 |
|---|---|
| OS | Linux（Ubuntu 20.04/22.04 推荐） |
| GPU | 至少 1 张 **16 GB** 显存；8 GB 可跑但 batch 只能 4–8，时间翻倍 |
| CUDA | 12.1（torch 2.5.1+cu121 已验证） |
| Python | 3.10 或 3.12 |
| 磁盘 | ≥ 80 GB 空闲（数据 + checkpoint + 轨迹缓存） |
| 内存 | ≥ 32 GB 推荐 |
| 网络 | 能下载 BLIP checkpoint（1.77 GB）和 HuggingFace `bert-base-uncased` tokenizer |

依赖（pip）：

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.3 timm==0.9.16 xgboost faiss-cpu netcal \
            pandas pyarrow scikit-learn tqdm ftfy regex
```

注意：`transformers` 必须 **4.46.3**，5.x 会要求 torch≥2.6；`timm` 必须
**0.9.16**，新版 BLIP 的 `_cfg` 会报错。

---

## 2. 必须拷贝的文件

从本地机器拷贝整个 `_traced/`，但要排除 `.git`、`checkpoints/` 和
`local_data/`（checkpoint 在服务器单独下载，数据用服务器本地路径）。

必须包含：

```
_traced/
  create_trajectory.py
  add_text_sim.py
  train_classifiers.py
  utils_TRACED.py
  fgb_datasets.py          # FGBench 适配器（本次新增）
  run_fgb_full.sh          # 完整五维长跑脚本（本次新增）
  lib/                     # 已含 fgb_* 分支
  BLIP/                    # 官方 BLIP 代码
  path_datasets.txt
  path_checkpoints.txt
  save_folder.txt
```

### 2.1 三处路径文件（必须改）

`path_datasets.txt`、`path_checkpoints.txt`、`save_folder.txt` 各只有一行，改成
服务器上的绝对路径：

```text
/path/to/data_root
/path/to/checkpoints_root
/path/to/output_root
```

### 2.2 FGBench 数据路径

`fgb_datasets.py` 用环境变量，默认值指向本地；服务器上导出：

```bash
export FGB_BENCH=/path/to/fgbench_v1_strict
export FGB_COCO=/path/to/MS-COCO
export FGB_VG=/path/to/VisualGenome
export FGB_VG_INDEX=/path/to/image_index.parquet
```

`FGB_BENCH` 下必须有 `train.parquet`、`val.parquet`、`test.parquet`。
COCO 图像路径由 parquet 的 `target_file` 拼到 `FGB_COCO`；VG 由
`FGB_VG_INDEX` 的 `rel_path` 拼到 `FGB_VG`。

### 2.3 BLIP checkpoint

下载：

```bash
mkdir -p /path/to/checkpoints_root/BLIP_models/blip_vit_b_ft_coco_itr
wget -O /path/to/checkpoints_root/BLIP_models/blip_vit_b_ft_coco_itr/model_base_retrieval_coco.pth \
  https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth
```

约 1.77 GB。`create_trajectory.py` 和 `add_text_sim.py` 会从这个路径加载；
不需要 Flickr checkpoint（五个 FGBench facet 都用 COCO BLIP）。

---

## 3. 运行方式

### 3.1 先跑一个 gate：完整 Quantity

不要直接开五维。先改 `run_fgb_full.sh` 第 5 行的 `cd` 为服务器上的
`_traced` 绝对路径，再跑 Quantity（最小，约 40 分钟轨迹）：

```bash
cd /path/to/_traced
bash run_fgb_full.sh   # 会从 Quantity 开始
```

或手动只跑 Quantity：

```bash
python3 create_trajectory.py --output_dir traced_full --dataset fgb_quantity \
  --noise_type cat --noise_level 0.5 --exploration_strat 0 \
  --top_k_val 8 --alignment_metric BLIP --batch_size_GCD 8 \
  --num_steps_GCD 3 --match_head itm --seed 0
python3 add_text_sim.py  # 同上参数
python3 train_classifiers.py  # 同上参数
```

### 3.2 完整五维

确认 Quantity 的 `results.json` 正常后，用 tmux 跑完整脚本：

```bash
tmux new-session -d -s traced_full "bash run_fgb_full.sh"
tmux attach -t traced_full
```

顺序：`Quantity → Relation → Action → Entity → Attribute`。
脚本写日志到 `full_logs/<dataset>_<step>.log`，每步结束打印 `rc`。

### 3.3 预期时间（单卡 16 GB，batch 8）

| Facet | train | val | test | 轨迹估算 |
|---|---:|---:|---:|---:|
| Quantity | 10,828 | 476 | 456 | ~40 min |
| Relation | 8,396 | 324 | 2,824 | ~40 min |
| Action | 1,596 | 8 | 664 | ~10 min |
| Entity | 112,000 | 9,164 | 9,192 | ~7.5 h |
| Attribute | 130,708 | 8,336 | 57,124 | ~11 h |

总计 **约 20–22 小时**轨迹，加 `add_text_sim` 和分类器约 **22–26 小时**。
8 GB 卡请把 `batch_size_GCD` 改成 4，时间约翻倍。

---

## 4. 输出要求（必须回传）

每个 facet 的输出目录：

```
<save_folder>/saved_exp/traced_full_<dataset>_50_TRACED_elim_cat_BLIP_itm/
```

其中 `<dataset>` 是 `fgb_quantity`、`fgb_relation`、`fgb_action`、`fgb_entity`、
`fgb_attr`。目录里必须有以下文件：

| 文件 | 用途 |
|---|---|
| `results.json` | **最重要**：test AUC / Accuracy、CV AUC、best params；两套特征（Alignment Score / Alignment+Cosine）各 xgboost/CART |
| `args.json` | 记录完整参数，核对没有跑成 subset |
| `d_cos_sim_sentences_{train,val,test}.pt` | 文本相似度轨迹，审计用 |
| `time_alg_{train,val,test}.json` | 每 split 的运行时间 |
| `full_logs/<dataset>_<step>.log` | 三步的完整 stdout/stderr |

回传方式：

```bash
cd <save_folder>/saved_exp
tar -czf traced_full_results.tar.gz \
  traced_full_fgb_quantity_50_TRACED_elim_cat_BLIP_itm/results.json \
  traced_full_fgb_relation_50_TRACED_elim_cat_BLIP_itm/results.json \
  traced_full_fgb_action_50_TRACED_elim_cat_BLIP_itm/results.json \
  traced_full_fgb_entity_50_TRACED_elim_cat_BLIP_itm/results.json \
  traced_full_fgb_attr_50_TRACED_elim_cat_BLIP_itm/results.json \
  traced_full_fgb_*_50_TRACED_elim_cat_BLIP_itm/args.json
```

再附一份 `SUMMARY.csv`：

```csv
facet,feature,model,test_auc,test_acc,n_test
Quantity,Alignment Score + Cosine Sim Text,xgboost,...
...
```

**同时回传**：`nvidia-smi` 输出、`git diff`（或 `fgb_datasets.py` 与三个
path 文件的最终版本）、实际耗时。

---

## 5. 验收标准

1. `args.json` 里 `dataset` 是 `fgb_*`，`noise_level=0.5`，`batch_size_GCD=8`
   （或 4），`top_k_val=8`，`match_head=itm`，`seed=0`。
2. `results.json` 存在，且 `Alignment Score + Cosine Sim Text` 的 xgboost
   `Test AUC` 在 0.5–1.0 之间；Quantity 预期 ~0.9，Entity ~1.0，Attribute
   ~0.7–0.8，Action/Relation ~0.6–0.7。
3. `d_cos_sim_sentences_test.pt` 存在；`time_alg_test.json` 的样本数与 test 行数
   一致（Quantity 456、Relation 2,824、Action 664、Entity 9,192、Attribute 57,124）。
4. 日志里没有 `Traceback`、`CUDA out of memory`、`FileNotFoundError`。
5. 没有把 `500/100/100` 写死；`fgb_datasets.py` 里 `n_train=0, n_val=0, n_test=0`
   表示完整 split。

---

## 6. 常见问题

| 现象 | 原因/处理 |
|---|---|
| `ModuleNotFoundError: open_clip` | 已把 open_clip 改成 lazy import；若仍报，检查是否用了旧版 `lib/models/utils.py` |
| `timm` `_cfg` 报错 | 装 `timm==0.9.16` |
| `transformers` 要求 torch≥2.6 | 装 `transformers==4.46.3` |
| `bert-base-uncased` 下载失败 | 先在有网环境 `huggingface-cli download bert-base-uncased`，或设置 `HF_ENDPOINT=https://hf-mirror.com` |
| `cat_labels` 解析报 `max() arg is empty` | CSV 里 `cat_labels` 必须是 `[1, 2, 3]` 形式；当前 `fgb_datasets.py` 直接构造 DataFrame，不走 CSV，正常不会触发 |
| `KeyError (cycle_id, target_image_id)` | 某个 cycle 没有 clean 行；检查 parquet 是否完整，或该 cycle 被 strict 过滤 |
| OOM | `--batch_size_GCD 4`；仍 OOM 用 2 |
| 结果目录多了 `_subset` | 说明用了旧 `fgb_datasets.py`；必须 `n_*=0` |

---

## 7. 不要做的事

- 不要用 500/100/100 的 pilot 结果冒充完整结果。
- 不要改 `noise_level=0.5`、`noise_type=cat`、`top_k_val=8`、`match_head=itm`。
- 不要跳过 `add_text_sim.py`；`train_classifiers.py` 依赖它的 `.pt`。
- 不要把 BLIP checkpoint 放进 `_traced/checkpoints` 之外的路径后忘记改
  `path_checkpoints.txt`。
- 不要把原始 COCO/VG 数据打进回传包。

---

## 8. 本地已完成的参考结果（pilot，仅供 sanity check）

| Facet | pilot test AUC（100 对） | 说明 |
|---|---:|---|
| Entity | 1.0000 | 饱和 |
| Quantity | 0.9199 | 与完整预期接近 |
| Attribute | 0.7321 | 100 对，CI 很宽 |
| Action | 0.6636 | 100 对 |
| Relation | 0.6506 | 100 对 |

完整跑出来后，用 `results.json` 里的 `Test AUC` 替换上表；预期会有几个 pp 的
波动，Entity/Attribute 的 test 规模变化最大。

---

## 9. 回传后我们做什么

1. 把五维 `results.json` 汇总成 `official_baselines.csv` 的 `traced_official` 列；
2. 与 `strict_main_table.csv` 的 `-style` 行做同 facet 对比；
3. 只在补充材料写 official TRACED；主表仍保留 `-style`，避免 100 对/完整口径混用；
4. 如果完整结果与 pilot 方向相反，先查 `args.json` 和 `time_alg_test.json` 的样本数，
   再决定是否重跑。
