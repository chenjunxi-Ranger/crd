# Paper-ready FGBench results · 2026-11-18

This folder is the canonical snapshot for writing the paper.

## Decision

1. **Use official LEMoN in the main table; keep TRACED-style.**  
   LEMoN official is the full FGBench test split and shares the CLIP feature/test
   protocol. TRACED remains `TRACED-style` because its CLIP-backbone official run is
   not available.

2. **No human-agreement claim.**  
   P0-A blind annotation was stopped after 89/400 items. Therefore:
   - do **not** report Cohen's κ;
   - do **not** claim that FGBench labels are human-verifiable;
   - keep the annotation kit only as future work / internal artifact.

3. **No P1 utility claim.**  
   All P1 variants were negative or inconclusive:
   - classification probe: CRD below Cosine on the primary setting;
   - hard-neighbour P1: CRD not best among non-Oracle arms;
   - full-113k retrieval: CRD below Cosine; one seed CI excluded zero on the negative side.
   Do **not** claim that CRD improves downstream cleaning or retrieval.

4. **Official TRACED goes to supplement, not the main table.**

## Main-table numbers to use

Use `main_table_strict.csv` as-is. The relevant rows are:

| Facet | CRD | LEMoN official | TRACED-style |
|---|---:|---:|---:|
| COCO Entity | 0.9996 | 0.9990 | 0.9985 |
| COCO Quantity | 0.8848 | 0.8266 | 0.8412 |
| VG Attribute | 0.7190 | 0.6755 | 0.6875 |
| VG Action | 0.6145 | 0.5923 | 0.6102 |
| VG Relation | 0.5531 | 0.5218 | 0.5356 |

Cosine / CSLS / GMM values are in `main_table_strict.csv`.

## Official-method supplement

See `official_baselines_supplement.csv`.

Important distinctions:

- **LEMoN official** was run on the full FGBench test split for all five facets.
- **TRACED official full COCO** exists only for Entity and Quantity.
- **TRACED official VG** is only a 100-pair pilot.
- TRACED official uses a **BLIP COCO ITM** backbone, while `TRACED-style` uses **CLIP ViT-B/16**.  
  They are not backbone-comparable, so do not mix them in the main table.

## P1 negative results

All P1 artifacts are in `p1_negative/`.

### Classification probe

Files:

- `p1_classification_seeds0_2.csv`
- `p1_classification_seeds3_4.csv`

Together: 270 result rows, seeds 0–4, 3 noise levels, 3 retention budgets, 6 arms.

Primary setting (ρ=0.4, 50% retention) had no CRD advantage:

| Arm | val_acc |
|---|---:|
| CSLS | 0.9570 |
| ABTT | 0.9564 |
| Cosine | 0.9560 |
| Oracle | 0.9560 |
| CRD | 0.9543 |

### Hard-neighbour P1

File: `p1_hard.csv`.

Three rounds used semantic-neighbour donors, low retention, and stronger probes.
CRD was never the best non-Oracle arm.

### Full-pool retrieval

File: `p1_retrieval_full.csv`.

Mean R@1 over three seeds:

| Arm | R@1 |
|---|---:|
| Cosine | 0.1090 |
| ABTT | 0.1087 |
| CSLS | 0.1060 |
| CRD | 0.1010 |
| Oracle | 0.0987 |
| None | 0.0957 |

Bootstrap CRD − Cosine:

| Seed | Diff | 95% CI |
|---:|---:|---|
| 0 | −0.0101 | [−0.0240, +0.0030] |
| 1 | +0.0054 | [−0.0080, +0.0190] |
| 2 | −0.0192 | [−0.0320, −0.0060] |

Conclusion: no downstream utility claim.

## Not done, by decision

- P0-A human annotation: stopped at 89/400.
- Full official TRACED on VG facets: not run.
- TRACED multi-seed: not run; Quantity seed replication issue was not resolved.
- Further P1 redesign: stopped after three hard variants and full retrieval.

## Writing checklist

1. Main table must say `LEMoN (official)` and `TRACED-style`.
2. Supplement may say `LEMoN (official adapter)` and `TRACED (official BLIP pipeline)`.
3. For TRACED official, explicitly state the backbone difference.
4. For TRACED VG, explicitly state the 100-pair pilot scope.
5. Do not mention human agreement or Cohen's κ.
6. Do not mention downstream cleaning/retrieval improvement.
