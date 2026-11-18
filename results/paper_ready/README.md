# Canonical result artifacts

This directory contains the exact CSVs used for the paper.

## Main table

- `main_table_strict.csv`: main AUROC table. The `lemon` row is the official
  LEMoN adapter on the full strict test split; `traced` is the deletion-only
  TRACED-style baseline.
- `official_lemon_stats.csv`: paired cycle-clustered bootstrap and Holm
  correction for the official LEMoN row.
- `official_baselines_supplement.csv`: official LEMoN / TRACED scope table.

## Joint-head control

- `joint_vs_additive.csv`: additive vs joint MLP, four facets × three seeds.
  The additive column is a same-cache re-run; see the top-level README for the
  difference from Table II.

## Negative controls

`p1_negative/` contains the P1 classification, hard-neighbour, and full-pool
retrieval results. These are included for transparency; the paper does not
claim a downstream utility gain.

## Notes

- Main-table CRD values use `auroc_seedmean` for the seed mean; `auroc` is the
  integrated score.
- Official LEMoN per-row files are in `../../scores/official_lemon/`.
- Images are not redistributed; use the manifests and `data/image_index.parquet`
  with the original COCO / Visual Genome downloads.
