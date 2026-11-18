# Paper update plan (staged; do not compile until approval)

Status: the `.tex` currently contains the P0-C/D additions and a trim pass from the
last session, but it was **not recompiled** after those edits. Expected length is
7 pages; the plan below returns it to 6. No further edit/compile will be made until
the user approves.

## A. P0-C / P0-D wording fixes

1. `sec/6_experiments.tex` (Head class):
   - `matches CRD on Attribute and Entity ($0.7187$/$0.9996$)`
     -> `matches CRD on Attribute ($0.7187$) and Entity (single-seed $0.9996$;
     $0.9995$ under the seed-mean convention)`.
   - `explains $33\%$ of $z$'s variance ... below $0.5\%$`
     -> `reaches $R^2=0.326$ on Attribute, while caption length and numeral count
     reach $R^2=0.005$`.

2. `sec/6_experiments.tex` (Action budget):
   - `With the $320$-row budget, ... so the budget penalty is model-dependent.`
     -> `With the $320$-row ($80$-cycle) budget the LR probe falls to $0.5687$
     ($-4.59$ vs CRD) and the MLP holds at $0.6517$ ($+3.71$); because the 5-fold
     validation AUC on 80 cycles is only $0.14$, we report this as model-dependent
     rather than as a clean ordering.`

3. Remove the `+0.43` collision: use `+0.44` for the LR probe on Attribute
   (integrated $0.7234-0.7190$), or add `(Action)` to the CRD-vs-TRACED $+0.43$.
   Preferred: keep the Action value `+0.43` and write the probe as
   `$+0.44$ on the integrated score`.

4. Expand ABTT at first use: `all-but-the-top (ABTT) cosine` in §VI-A; the table row
   already says `ABTT cosine`.

5. Budget terminology: write `8,336-row Attribute budget` and
   `320-row (80-cycle) Action budget` once each; do not use "label budget" alone.

## B. Page recovery (target -15 lines)

| # | Cut | Est. lines |
|---|---|---:|
| 1 | `6_experiments.tex` Limitations: remove the repeated “supervised classifier is stronger” first half; keep the closing sentence | −4 |
| 2 | `6_experiments.tex` GMM duplicate explanation: keep “monotone, 2AFC moves −1.24/−3.51”, drop the 24-cell aside | −3 |
| 3 | `4_method.tex` Overview: two sentences (training path / inference path) | −4 |
| 4 | `6_experiments.tex` cross-backbone: drop “Because heads and λ are refitted…” (caption already says per-cell) | −3 |
| 5 | `8_conclusion.tex`: drop the repeated 0.6806→0.7187 and 88,024 numbers | −3 |

If still over: `3_problem.tex` within-cycle numbers (`0.324→0.188/0.253`) can be
shortened by one line.

## C. After the one-time edit

1. Compile once; target 6 pages, 0 overfull, 0 undefined.
2. Run the three-cycle review loop (3 subagents + 2 reviewers per cycle) on the
   compiled 6-page PDF.
3. Record any new experiment request in `REVISION_LOG.md`; do not run it.

## D. P0-A / P0-B / P1 status (not paper text)

- P0-A: blind kit ready at
  `results/fgbench/p0_baselines/annotation_kit/blind/`; 400 pairs, shuffled,
  labels stripped, 400/400 images exist. **Human labels and κ are still missing**;
  `kappa.py` is validated on synthetic labels only.
- P0-B: official LEMoN blocked by Windows `freeze_support` (no main guard);
  TRACED blocked by missing BLIP checkpoints. No official number; keep `-style`
  rows. See `P0_B_FEASIBILITY.md`.
- P1: design frozen in `results/fgbench/p1/P1_DESIGN.md`; not run.
