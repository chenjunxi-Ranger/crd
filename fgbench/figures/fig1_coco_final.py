"""Figure 1 - problem motivation.

Two real image-text pairs are scored by one frozen vision-language encoder.
The incorrect pair obtains the higher cosine similarity, so raw scores do not
order pairs globally.

Layout: pair card -> shared frozen encoder -> score block, one row per pair,
plus a comparison strip. Monochrome plus two muted accents.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

FIG = Path(__file__).resolve().parents[2] / "results" / "fig1_coco"
PANEL_A_ROW = 3
INK, GREY, HAIR = "#111111", "#5f5f5f", "#c4c4c4"
CLEAN, NOISY = "#31536b", "#8a4b3a"
FROZEN = "#dce5ee"
W_IN, H_IN = 7.09, 3.74          # 180 x 95 mm

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 6,
    "mathtext.fontset": "dejavusans",
})


def box(fig, x, y, w, h, fc="white", ec=INK, lw=0.8, r=0.006, z=2):
    fig.add_artist(FancyBboxPatch((x, y), w, h,
                                  boxstyle=f"round,pad=0,rounding_size={r}",
                                  transform=fig.transFigure, facecolor=fc,
                                  edgecolor=ec, lw=lw, zorder=z))


def arrow(fig, x1, y1, x2, y2, lw=0.75, mut=5.5):
    fig.add_artist(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                   mutation_scale=mut, lw=lw, color=INK,
                                   shrinkA=0, shrinkB=0, zorder=3))


def txt(fig, x, y, s, size=6, color=INK, ha="center", va="center",
        weight="normal", style="normal", ls_=1.25):
    return fig.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                    weight=weight, style=style, linespacing=ls_, zorder=4)


def wrap(text, width):
    words, out, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            out.append(cur); cur = w
    out.append(cur)
    return "\n".join(out)


def thumb(fig, path, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_zorder(1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color(HAIR)


def main() -> int:
    d = pd.read_csv(FIG / "fig1_coco_candidates.csv")
    r = d.iloc[PANEL_A_ROW]

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)


    rows = [(0.545, r.clean_image_path, r.clean_caption, r.clean_score,
             "clean", CLEAN),
            (0.285, r.noisy_image_path, r.noisy_caption, r.noisy_score,
             "noisy", NOISY)]

    for y0, path, cap, score, lab, col in rows:
        cy = y0 + 0.125
        box(fig, 0.020, y0, 0.400, 0.250, z=0.5)
        thumb(fig, path, 0.032, y0 + 0.019, 0.135, 0.211)
        txt(fig, 0.182, cy + 0.004, wrap(cap, 40), size=5.7, color=INK,
            va="center", ha="left")

        arrow(fig, 0.422, cy, 0.468, cy)
        arrow(fig, 0.618, cy, 0.654, cy)

        box(fig, 0.656, cy - 0.068, 0.260, 0.136)
        txt(fig, 0.786, cy + 0.024, f"{score:.3f}", size=17, color=col,
            weight="bold")
        txt(fig, 0.786, cy - 0.040, lab, size=6.6, color=col)

    # shared encoder
    box(fig, 0.470, 0.285, 0.146, 0.510, fc=FROZEN)
    txt(fig, 0.543, 0.596, "CLIP", size=7.6, weight="bold")
    txt(fig, 0.543, 0.530, "cosine", size=5.4, color=GREY)
    txt(fig, 0.543, 0.498, "$s(I,T)$", size=6.4)

    out = FIG / "Fig1_COCO"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    print("wrote", out)
    print("clean", r.clean_pair_id, f"{r.clean_score:.3f}",
          "| noisy", r.noisy_pair_id, f"{r.noisy_score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
