# CRD — IEEE ICME 2027 submission draft

Template migration from the CVPR author kit to the **generic IEEE conference
template**. This is a working draft, not a final submission package.

## Current status

| Item | Value |
|---|---|
| Document class | `\documentclass[conference]{IEEEtran}` |
| IEEEtran version | **V1.8b (2015/08/26)**, shipped with MiKTeX |
| Compiler | pdfLaTeX + BibTeX (no `latexmk`; this machine has no Perl) |
| **Current PDF length** | **9 pages** (two-column, US Letter; built 2026-11-16) |
| Bibliography style | `IEEEtran` |
| Figures | Figure 1 (`fig/fig1_coco.pdf`), Figure 2 (`fig/fig2_crd.pdf`), both vector, both `figure*` |
| Tables | Table I (FGBench statistics), Table II (main results, `table*`), Table III (cross-backbone), Table IV (ablation) |

## Build

```powershell
cd paper\CRD_ICME2027
powershell -ExecutionPolicy Bypass -File tools\build.ps1          # pdflatex -> bibtex -> pdflatex x2
powershell -ExecutionPolicy Bypass -File tools\build.ps1 -Clean   # drop intermediates first
```

The script prints `OK  main.pdf  N pages  ...`. Manual equivalent:

```
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Layout of the project

```
CRD_ICME2027/
├─ main.tex            document class, title, keywords, section order
├─ references.bib      IEEE-style bibliography database
├─ fig/
│   ├─ fig1_coco.pdf   Figure 1 — raw-similarity inversion example
│   ├─ fig2_crd.pdf    Figure 2 — CRD training / inference overview
│   ├─ fig1.tex        figure* wrapper + caption
│   └─ fig2.tex
├─ sec/
│   ├─ 0_abstract.tex
│   ├─ 1_intro.tex
│   ├─ 2_related.tex
│   ├─ 3_problem.tex
│   ├─ 4_method.tex
│   ├─ 5_fgbench.tex
│   ├─ 6_experiments.tex   includes the former analysis subsections
│   └─ 8_conclusion.tex
├─ tools/
│   ├─ build.ps1
│   ├─ merge_analysis.py   one-off: merged the old chapter 7 into chapter 6
│   └─ _preview/           page renders used during checking
└─ main.pdf
```

## Section structure (seven first-level sections)

| # | Section | Source file |
|---|---|---|
| I | Introduction | `sec/1_intro.tex` |
| II | Related Work | `sec/2_related.tex` |
| III | Problem Formulation and Motivation | `sec/3_problem.tex` |
| IV | Cross-modal Score Calibration | `sec/4_method.tex` |
| V | FGBench: Benchmark Construction | `sec/5_fgbench.tex` |
| VI | Experiments | `sec/6_experiments.tex` |
| VII | Conclusion | `sec/8_conclusion.tex` |

The CVPR draft had a separate *Further Analysis and Discussion* chapter. Under the
ICME structure it is now subsections F–I of Section VI:

- VI-F What the Calibration Changes and Does Not Change
- VI-G External Transfer (five external protocols, CRD vs the supervised probe)
- VI-H Scope of the Additive Reading
- VI-I Limitations

Nothing was deleted; only the heading level changed.

## Content freeze — what was and was not touched

This was a **template migration**. Scientific content is unchanged. Verified by a
line diff against `../draft/sec/`:

| File | Changed lines |
|---|---|
| `0_abstract.tex` | 0 |
| `1_intro.tex` | 1 (`\needspace` to avoid an orphan heading) |
| `2_related.tex` | 0 |
| `3_problem.tex` | 0 |
| `4_method.tex` | 0 |
| `5_fgbench.tex` | 0 |
| `8_conclusion.tex` | 0 |
| `6_experiments.tex` | merged chapter + table layout only; no numeric edits |

In particular these were **not** modified: the CRD definition and training
objective, row-level Huber regression, the standardisation with training-set
$\mu,\sigma$, validation-set selection of $\lambda$, the five FGBench facets, the
dataset statistics and splits, all CLIP / CSLS / LEMoN-style / TRACED-style / CRD
numbers, the cross-backbone results, the ablations and the bootstrap statistics.

Numbers trace back to `results/fgbench/strict_main_table.csv` and
`results/fgbench/final_stats_strict.csv` in the parent project.

## ICME 2027 submission facts (verified 2026-11-16)

| Item | Value | Source |
|---|---|---|
| Conference | July 13–17, 2027, Xiamen, China | official CFP PDF (`icme27-cfp_web-05.pdf`) |
| Regular paper deadline | **2026-12-30, 23:59 AoE** (incl. special sessions) | official CFP PDF + CFP page |
| Acceptance notification | 2027-03-24 | official CFP page |
| Camera-ready | 2027-05-10 | official CFP page |
| Review system | Microsoft CMT | official CFP page |
| Page limit | **not yet published for 2027** | — |
| Template | **not yet published for 2027** | — |
| Double-blind | **not yet published for 2027** | — |

**Precedent (ICME 2026, official camera-ready page):** "You are allowed a total of
**6 pages** for your paper." ICME has used the IEEE two-column conference format and
double-blind review in recent editions. Treat **6 pages total** as the working
target and re-check the 2027 author kit when it appears (likely Nov–Dec 2026).

### Page budget — currently over

The draft is **9 pages**, i.e. **3 pages over** the working target. Do not shrink
margins or fonts. Candidate cuts, in order of least damage:

| Cut | Saving (est.) |
|---|---:|
| Merge Section VI-F/G/H into one "Discussion" subsection, keep the 2AFC identity proof and the scope caveat | 0.8 page |
| Move Table III (cross-backbone) to supplementary; keep two sentences and the 15/15 claim | 0.6 page |
| Make Figure 1 single-column instead of `figure*` | 0.4 page |
| Trim Related Work to one paragraph per family | 0.4 page |
| Fold the new "supervised reference" paragraph into the main comparison paragraph (keep the table row) | 0.3 page |
| Shorten Table IV to four rows (base / image-only / text-only / additive); cycle & placebo to text | 0.3 page |
| Reference list compaction (ICME style, drop unused entries) | 0.2 page |

Open items in the LaTeX source: author block is still a placeholder
(`Anonymous Submission`, `Paper ID 0000`); no copyright block; camera-ready extras
(DOI, funding, final author list) are deferred.

## Relation to the CVPR project

The CVPR author-kit project is kept untouched at `../draft/` (with its own
`main.tex`, `main.bib`, `preamble.tex` and figure copies). The two drafts share the
same figures by copy, not by symlink, so editing one does not affect the other.

A Chinese reading version lives at `../cn/main_cn.tex` (compile with XeLaTeX).
