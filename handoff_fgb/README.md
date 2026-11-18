# FGBench × TRACED 交接包

这个包让同伴服务器**不需要本地的数据集划分**也能跑完整 TRACED：冻结的
`train/val/test.parquet` 和 VG `image_index.parquet` 已经放在 `data/`，只需要
服务器上的 COCO/VG 原图。

> 为什么不“重建划分”：FGBench v1-strict 是冻结版本，重建脚本会引入版本漂移；
> 交接冻结 parquet 更可靠，也保证和论文主表完全同一批样本。

## 1. 包内容

```
handoff_fgb/
├── README.md                       # 本文件
├── HANDOFF_FULL_RUN.md             # 服务器要求、运行、输出、验收（必读）
├── requirements.txt
├── data/
│   ├── fgbench_v1_strict/
│   │   ├── train.parquet           # 11.1 MB
│   │   ├── val.parquet             # 0.8 MB
│   │   └── test.parquet            # 2.5 MB
│   └── image_index.parquet         # 108,249 张 VG 图的 id -> 相对路径
├── code/
│   ├── fgb_datasets.py             # FGBench -> TRACED 适配器（环境变量）
│   ├── run_fgb_full.sh             # 五维完整长跑（不写死路径）
│   ├── verify_data.py              # 校验所有图片存在 + cycle 形状
│   └── prepare_vg_index.py         # 如果 image_index 不匹配，重新扫描 VG
└── traced_code.tar.gz              # 官方 TRACED 代码（2.9 MB，不含数据/checkpoint）
```

## 2. 服务器快速开始

### 2.1 解压代码

```bash
tar -xzf traced_code.tar.gz          # 得到 _traced/
cp code/fgb_datasets.py _traced/
cp code/run_fgb_full.sh _traced/
chmod +x _traced/run_fgb_full.sh
```

### 2.2 准备原图

`FGB_COCO` 指向包含 `train2014/`、`val2014/` 的 COCO 根目录；
`FGB_VG` 指向包含 `VG_100K/`、`VG_100K_2/` 的 Visual Genome 根目录。

COCO 2014（官方）：

```bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip && unzip val2014.zip
```

Visual Genome（官方或镜像；只需要图片，不需要 annotations）：

```text
https://visualgenome.org/static/data/dataset/VG_100K.zip
https://visualgenome.org/static/data/dataset/VG_100K_2.zip
```

如果官方 403，用 HuggingFace 镜像 `AsphyXIA/visual-genome` 的图片文件，
保持目录名为 `VG_100K/` 和 `VG_100K_2/`。

### 2.3 设置路径并校验

```bash
export FGB_BENCH=$PWD/data/fgbench_v1_strict
export FGB_COCO=/path/to/COCO
export FGB_VG=/path/to/VisualGenome
export FGB_VG_INDEX=$PWD/data/image_index.parquet

python code/verify_data.py
```

必须看到 `OK: all images exist; every cycle has 4 rows and 2 clean pairs.`。
如果 VG 图片命名/目录不同，先跑：

```bash
export FGB_VG=/path/to/VisualGenome
python code/prepare_vg_index.py
```

### 2.4 配置 TRACED

在 `_traced/` 下改三个只有一行的文件：

```text
path_datasets.txt     -> /path/to/data_root
path_checkpoints.txt  -> /path/to/checkpoints_root
save_folder.txt       -> /path/to/output_root
```

下载 BLIP checkpoint：

```bash
mkdir -p /path/to/checkpoints_root/BLIP_models/blip_vit_b_ft_coco_itr
wget -O /path/to/checkpoints_root/BLIP_models/blip_vit_b_ft_coco_itr/model_base_retrieval_coco.pth \
  https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth
```

### 2.5 跑

先跑 Quantity gate（约 40 分钟）：

```bash
cd _traced
bash run_fgb_full.sh
```

确认 `results.json` 正常后，用 tmux 跑完整五维：

```bash
tmux new-session -d -s traced_full "bash run_fgb_full.sh"
tmux attach -t traced_full
```

## 3. 数据规模（完整，不是 500/100/100）

| Facet | train | val | test |
|---|---:|---:|---:|
| Quantity | 10,828 | 476 | 456 |
| Relation | 8,396 | 324 | 2,824 |
| Action | 1,596 | 8 | 664 |
| Entity | 112,000 | 9,164 | 9,192 |
| Attribute | 130,708 | 8,336 | 57,124 |

## 4. 必须回传

每个 facet 的 `results.json`、`args.json`、`d_cos_sim_sentences_*.pt`、
`time_alg_*.json` 和 `full_logs/`；再加一份 `SUMMARY.csv`。格式和打包命令见
`HANDOFF_FULL_RUN.md` 第 4 节。

## 5. 边界

- 完整跑约 22–26 小时（16 GB 卡，batch 8）。
- 8 GB 卡用 `TRACED_BATCH=4`，时间约翻倍。
- 包内 `data/` 是冻结划分，**不要**用 500/100/100 的 pilot 结果替代。
- 本地 pilot 只用于 sanity check，不作为正式数字。
