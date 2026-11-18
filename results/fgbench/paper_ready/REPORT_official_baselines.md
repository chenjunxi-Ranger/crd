# Official LEMoN / TRACED on FGBench — reproduction report

Date: 2026-11-17. Environment: WSL2 Ubuntu, Python 3.12, torch 2.5.1+cu121,
transformers 4.46.3, faiss-cpu, timm 0.9.16, xgboost 3.4.1, one RTX 4060 (8 GB).
The frozen `-style` numbers are from `strict_main_table.csv`; the official rows were
produced here, not copied from the companion run.

## LEMoN (official algorithm, FGBench adapter)

Adapter: cached CLIP ViT-B/16 features; train index `IndexFlatIP`; k=5; the official
`d1 + beta*d_n + gamma*d_m` score; hyperparameters fitted on FGBench validation rows
with `lib.metrics.utils.maximize_metric`; test scored once.

| Facet | -style | official | Δ (pp) | n_test |
|---|---:|---:|---:|---:|
| Entity | 0.9990 | 0.9990 | 0.00 | 9,192 |
| Quantity | 0.8495 | 0.8266 | −2.29 | 456 |
| Attribute | 0.6843 | 0.6755 | −0.88 | 57,124 |
| Action | 0.5964 | 0.5923 | −0.41 | 664 |
| Relation | 0.5436 | 0.5218 | −2.18 | 2,824 |

The companion's official LEMoN rows are 0.9994 / 0.8324 / 0.6780 / 0.5914 / 0.5366;
our adapter differs by ≤1.5 pp (largest on Relation). The difference is attributable
to adapter details (neighbour exclusion / hyperparameter search), not to the data.
**Conclusion:** official LEMoN does not beat the `-style` row on any FGBench facet;
the main-table `-style` row is conservative.

## TRACED (official BLIP pipeline, FGBench adapter)

Official code + BLIP COCO retrieval checkpoint
(`model_base_retrieval_coco.pth`, 1.77 GB). FGBench adapter: one row per
(image, caption) pair, `gold_sentence` = the caption belonging to that image in the
same reciprocal cycle, 50% noisy labels; 500 train / 100 val / 100 test per facet;
elimination trajectory, `top_k_val=8`, `batch_size_GCD=8`, `match_head=itm`, seed 0.
Classifier: official xgboost on `Alignment Score + Cosine Sim Text`, 3-fold CV on
train+val, test AUC on 100 pairs.

| Facet | -style | official | Δ (pp) | n_test |
|---|---:|---:|---:|---:|
| Entity | 0.9985 | 1.0000 | +0.15 | 100 |
| Quantity | 0.8412 | 0.9199 | +7.87 | 100 |
| Attribute | 0.6875 | 0.7321 | +4.46 | 100 |
| Action | 0.6102 | 0.6636 | +5.34 | 100 |
| Relation | 0.5356 | 0.6506 | +11.50 | 100 |

**Caveats:** this is a **100-pair test subset**, not the full FGBench test split; the
CIs are wide and the absolute values are not comparable to the main table. The
direction is consistent and large, but it should be reported as a pilot / supplement,
not as a replacement for the `-style` row. Full-split TRACED would require ~70k
trajectories and is a separate compute campaign.

## Commands

```bash
# LEMoN adapter (WSL)
cd /mnt/d/PyCharm/PythonProject/异常标签检测/mymodel7
PYTHONPATH=.../_clip_pkg:.../LEMoN-main python3 \
  fgbench/experiments/p0_baselines/fgb_lemon_official.py --facet all

# TRACED (WSL, per facet)
cd /mnt/d/PyCharm/PythonProject/异常标签检测/_traced
python3 create_trajectory.py --output_dir traced_fgb --dataset fgb_attr \
  --noise_type cat --noise_level 0.5 --exploration_strat 0 --top_k_val 8 \
  --alignment_metric BLIP --batch_size_GCD 8 --num_steps_GCD 3 --match_head itm --seed 0
python3 add_text_sim.py ...   # same args
python3 train_classifiers.py ...  # same args
```
