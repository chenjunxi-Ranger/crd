# CRD / FGBench

Research code for cross-modal residual debiasing (CRD) and the FGBench
benchmark for global noisy image--text pair detection.

## Start here

Read `HANDOVER.md` first: it summarizes the project goal, module map, completed
work, known issues, run instructions, and handover cautions.

## Repository layout

```text
fgbench/              benchmark construction, detectors, evaluation, experiments
src/                  CLIP encoders and scoring utilities
scripts/              data preparation and analysis helpers
data/                 small manifests / FOIL validation files
results/fgbench/      paper-ready CSVs and small result tables
paper/CRD_ICME2027/   LaTeX source, figures, and compiled PDFs
handoff_fgb/          handoff bundle for the FGBench data pipeline
analysis/             analysis scripts and figures
```

Large assets are intentionally not committed:

- COCO / Visual Genome images;
- CLIP/BLIP feature caches (`*.npz`, `*.pt`);
- trajectory dumps and per-run checkpoints.

The image manifests and `image_index.parquet` identify the expected files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The main scripts expect a CUDA GPU for CLIP feature extraction; evaluation
scripts can run on CPU.

## Data

Download COCO 2014 and Visual Genome from their official sources. The benchmark
manifests are under `fgbench/data/fgbench_v1_strict/` in the original workspace;
the paper-ready split files are included in `results/fgbench/paper_ready/` and
the handoff bundle.

Set the following environment variables if your layout differs:

```text
VG_ROOT          Visual Genome root
COCO_ANN         COCO annotations directory
FGBENCH_OUT      derived benchmark directory
FGBENCH_BENCH    benchmark manifest directory
CRD_WEIGHTS      CLIP/BLIP weight root
```

## Main entry points

```text
fgbench/models/crd_backbone.py       additive CRD head and training
fgbench/models/crd_action.py         Action internal-split protocol
fgbench/evaluate/strict_main_table.py main AUROC table
fgbench/evaluate/final_stats_strict.py bootstrap + Holm
fgbench/evaluate/traced_fgbench.py   TRACED-style deletion baseline
fgbench/evaluate/lemon_fgbench.py    LEMoN-style baseline
fgbench/experiments/joint_ablation.py joint vs additive control
```

The paper-ready CSVs in `results/fgbench/paper_ready/` are the authoritative
numbers for the current manuscript.

## Paper

```bash
cd paper/CRD_ICME2027
powershell -ExecutionPolicy Bypass -File tools/build.ps1
```

The compiled English version is `paper/CRD_ICME2027/CRD_ICME2027_6pages.pdf`.
