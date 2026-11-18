# CRD / FGBench anonymous review release

This repository is an anonymized release for peer review. It contains the
FGBench v1-strict manifests and audit artifacts, per-row detector scores, the
main-table CSV, the joint-head ablation CSV, and the evaluation code used to
produce the reported numbers.

## What is included

```text
data/manifests/          train / val / test / all FGBench v1-strict manifests
data/audit/              reciprocal-cycle and merge audits, strict statistics
data/image_index.parquet COCO/VG image-id to relative-path index
scores/                  per-row scores for cosine/CSLS/GMM/LEMoN-style/TRACED-style/CRD
scores/official_lemon/   official LEMoN scores; test is complete, train is a subset
results/                 main table, official LEMoN statistics, joint ablation
code/                    evaluation code snapshot
scripts/reproduce.sh     release consistency check
```

## What is not included

- Raw COCO or Visual Genome images. They remain under their original licenses.
- Frozen CLIP/BLIP features and model checkpoints.
- Training logs and large intermediate tensors.

The manifests use COCO/Visual Genome image identifiers. Download the images from
the original sources and use `data/image_index.parquet` to resolve relative
paths.

## Quick verification

```bash
pip install -r requirements.txt
bash scripts/reproduce.sh
```

This checks:

- FGBench cycle and pair counts;
- the reciprocal-cycle audit result;
- the main-table CRD and official LEMoN rows;
- the joint-head ablation table.

## Main table

`results/paper_ready/main_table_strict.csv` is the exact table used in the
paper. The main-table LEMoN row is the **official LEMoN adapter** evaluated on
the full strict test split with cached CLIP ViT-B/16 features. The per-row
official LEMoN files have complete test segments; their train segments are
exported subsets (32k rows for Entity/Attribute, 8k for Relation). The
`LEMoN-style` scores are kept separately for reference.

`results/paper_ready/official_lemon_stats.csv` contains the paired cycle-clustered
bootstrap and Holm results after replacing the LEMoN-style row with the
official adapter.

## Joint-head ablation

`results/paper_ready/joint_vs_additive.csv` contains all runs for:

- additive heads: `a(u) + b(v) + b0`;
- joint MLP: `MLP([u; v])`.

Both use the same cached features, 1,500-step row target, optimizer, batch size,
lambda grid, and three seeds. The parameter counts are 65,795 (additive) and
65,665 (joint). Action is excluded from this control because it uses the
internal-validation protocol. The additive column is a **same-cache re-run**;
it differs from the frozen Table II CRD row by up to 0.003 AUROC (0.3 points) on
Quantity (Relation differs by at most 0.001) because Table II aggregates the
three frozen score files, not this re-run.

## Code map

- `code/fgbench/models/crd_backbone.py`: additive CRD head and training loop.
- `code/fgbench/models/crd_action.py`: Action internal-split protocol.
- `code/fgbench/evaluate/strict_main_table.py`: main AUROC table; requires
  frozen feature tensors that are not shipped, so use the CSV for exact numbers.
- `code/fgbench/evaluate/final_stats_strict.py`: paired cycle-clustered bootstrap.
- `code/fgbench/evaluate/traced_fgbench.py`: deletion-only TRACED-style baseline.
- `code/fgbench/evaluate/lemon_fgbench.py`: LEMoN-style baseline.
- `code/fgbench/models/gmm_filter.py`: GMM control.
- `code/fgbench/evaluate/csls_gmm_blip.py`: CSLS/GMM helpers.
- `code/fgbench/experiments/joint_ablation.py`: joint-head vs additive control.

The code is a research snapshot. Paths to datasets/checkpoints are configured at
the top of each script. The CSV artifacts are the authoritative results for the
paper.

## Data schema

Each manifest row contains the FGBench fields (`dataset`, `facet`, `cycle_id`,
`pair_label`, image/caption identifiers, and source metadata). A reciprocal
cycle has four rows: two clean and two noisy pairs. The strict test split
contains 17,565 cycles and 70,260 pairs; the full v1-strict benchmark contains
88,024 cycles and 352,096 pairs.

## Anonymity

This release contains no author names, affiliations, personal URLs, or
institutional paths. It is intended to be linked from the paper through an
anonymous repository proxy.
