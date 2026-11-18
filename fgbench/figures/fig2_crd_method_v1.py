"""Figure 2 - V1 baseline (restored), used only as the editing base.

V1 characteristics kept here:
  (a) encoders are standalone boxes labelled "Frozen Image Encoder" /
      "Frozen Text Encoder" (no dashed container), Similarity has no inputs
  (b) the two encoders sit in a dashed "Frozen Dual Encoder" box and a single
      line leaves that box towards Similarity
  module sizes differ from row to row
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "results" / "fig2"

INK = "#111111"
GREY = "#555555"
FROZEN = "#dce5ee"
W_IN, H_IN = 7.09, 3.95

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 6,
    "mathtext.fontset": "dejavusans",
})


def box(fig, x, y, w, h, fc="white", lw=0.8, ls="-", ec=INK, r=0.006, z=2):
    fig.add_artist(FancyBboxPatch((x, y), w, h,
                                  boxstyle=f"round,pad=0,rounding_size={r}",
                                  transform=fig.transFigure, facecolor=fc,
                                  edgecolor=ec, lw=lw, linestyle=ls, zorder=z))


def arrow(fig, x1, y1, x2, y2, lw=0.75, mut=5.5, ls="-"):
    fig.add_artist(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                   mutation_scale=mut, lw=lw, color=INK,
                                   linestyle=ls, shrinkA=0, shrinkB=0, zorder=3))


def path(fig, pts, lw=0.75, head=True):
    xs, ys = zip(*pts)
    if head:
        fig.add_artist(plt.Line2D(xs[:-1], ys[:-1], color=INK, lw=lw,
                                  solid_capstyle="butt", zorder=3))
        arrow(fig, xs[-2], ys[-2], xs[-1], ys[-1], lw=lw)
    else:
        fig.add_artist(plt.Line2D(xs, ys, color=INK, lw=lw,
                                  solid_capstyle="butt", zorder=3))


def txt(fig, x, y, s, size=6, color=INK, ha="center", va="center",
        weight="normal", style="normal", ls_=1.2):
    return fig.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                    weight=weight, style=style, linespacing=ls_, zorder=4)


def main() -> int:
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)

    txt(fig, 0.015, 0.962, "(a) Training: Learning Additive Score Components",
        size=7.2, weight="bold", ha="left")
    txt(fig, 0.520, 0.962, "(b) Inference: Pair-wise Score Calibration",
        size=7.2, weight="bold", ha="left")

    # ------------------------------ (a) ------------------------------
    box(fig, 0.020, 0.640, 0.072, 0.220)
    txt(fig, 0.056, 0.828, "Training", size=5.4, weight="bold")
    txt(fig, 0.056, 0.782, "image-text\npairs", size=5.0, color=GREY)
    txt(fig, 0.056, 0.700, "$\\{(I_i,T_i)\\}$", size=5.6)

    box(fig, 0.112, 0.745, 0.095, 0.085, fc=FROZEN)
    txt(fig, 0.1595, 0.803, "Frozen Image", size=5.2, weight="bold")
    txt(fig, 0.1595, 0.772, "Encoder", size=5.2, weight="bold")
    box(fig, 0.112, 0.615, 0.095, 0.085, fc=FROZEN)
    txt(fig, 0.1595, 0.673, "Frozen Text", size=5.2, weight="bold")
    txt(fig, 0.1595, 0.642, "Encoder", size=5.2, weight="bold")

    arrow(fig, 0.093, 0.7875, 0.110, 0.7875)
    arrow(fig, 0.093, 0.6575, 0.110, 0.6575)

    box(fig, 0.240, 0.745, 0.095, 0.085)
    txt(fig, 0.2875, 0.803, "Image Head", size=5.2, weight="bold")
    txt(fig, 0.2875, 0.770, "$a_\\phi(u)$", size=6.0)
    box(fig, 0.240, 0.615, 0.095, 0.085)
    txt(fig, 0.2875, 0.673, "Text Head", size=5.2, weight="bold")
    txt(fig, 0.2875, 0.640, "$b_\\psi(v)$", size=6.0)

    arrow(fig, 0.208, 0.7875, 0.238, 0.7875)
    arrow(fig, 0.208, 0.6575, 0.238, 0.6575)
    txt(fig, 0.223, 0.799, "$u$", size=6.2, style="italic")
    txt(fig, 0.223, 0.669, "$v$", size=6.2, style="italic")

    box(fig, 0.368, 0.630, 0.115, 0.175)
    txt(fig, 0.4255, 0.772, "$\\oplus$", size=7.0)
    txt(fig, 0.4255, 0.742, "$+\\,b_0$", size=5.6)
    txt(fig, 0.4255, 0.700, "$g_\\theta(I,T)\\,=$", size=5.8)
    txt(fig, 0.4255, 0.668, "$a_\\phi(u)+b_\\psi(v)+b_0$", size=4.8)

    arrow(fig, 0.336, 0.7875, 0.366, 0.7875)
    arrow(fig, 0.336, 0.6575, 0.366, 0.6575)

    box(fig, 0.300, 0.230, 0.165, 0.085)
    txt(fig, 0.3825, 0.284, "Huber Loss", size=5.4, weight="bold")
    txt(fig, 0.3825, 0.252, "$\\mathcal{L}(g_\\theta,\\,z)$", size=5.8)
    arrow(fig, 0.4255, 0.628, 0.4255, 0.317)

    box(fig, 0.020, 0.430, 0.100, 0.080)
    txt(fig, 0.070, 0.480, "Similarity", size=5.2, weight="bold")
    txt(fig, 0.070, 0.452, "$s(I,T)$", size=5.8)
    box(fig, 0.150, 0.430, 0.110, 0.080)
    txt(fig, 0.205, 0.480, "Standardization", size=4.8, weight="bold")
    txt(fig, 0.205, 0.452, "$z=(s-\\mu)/\\sigma$", size=5.4)

    arrow(fig, 0.121, 0.470, 0.148, 0.470)
    path(fig, [(0.261, 0.470), (0.330, 0.470), (0.330, 0.317)])
    txt(fig, 0.205, 0.407, "row-level target", size=4.6, color=GREY, style="italic")

    # ------------------------------ (b) ------------------------------
    box(fig, 0.520, 0.640, 0.075, 0.220)
    txt(fig, 0.5575, 0.828, "Test pair", size=5.4, weight="bold")
    txt(fig, 0.5575, 0.782, "$(I,T)$", size=6.0)

    box(fig, 0.604, 0.598, 0.128, 0.248, fc="white", ec="#9a9a9a", lw=0.6,
        ls=(0, (2.4, 1.6)), r=0.008)
    txt(fig, 0.668, 0.858, "Frozen Dual Encoder", size=5.0, color=GREY)

    box(fig, 0.620, 0.745, 0.095, 0.085, fc=FROZEN)
    txt(fig, 0.6675, 0.803, "Image", size=5.2, weight="bold")
    txt(fig, 0.6675, 0.772, "Encoder", size=5.2, weight="bold")
    box(fig, 0.620, 0.615, 0.095, 0.085, fc=FROZEN)
    txt(fig, 0.6675, 0.673, "Text", size=5.2, weight="bold")
    txt(fig, 0.6675, 0.642, "Encoder", size=5.2, weight="bold")

    arrow(fig, 0.596, 0.7875, 0.618, 0.7875)
    arrow(fig, 0.596, 0.6575, 0.618, 0.6575)

    box(fig, 0.745, 0.745, 0.095, 0.085)
    txt(fig, 0.7925, 0.803, "Image Head", size=5.2, weight="bold")
    txt(fig, 0.7925, 0.770, "$a_\\phi(u)$", size=6.0)
    box(fig, 0.745, 0.615, 0.095, 0.085)
    txt(fig, 0.7925, 0.673, "Text Head", size=5.2, weight="bold")
    txt(fig, 0.7925, 0.640, "$b_\\psi(v)$", size=6.0)

    arrow(fig, 0.716, 0.7875, 0.743, 0.7875)
    arrow(fig, 0.716, 0.6575, 0.743, 0.6575)
    txt(fig, 0.7295, 0.799, "$u$", size=6.2, style="italic")
    txt(fig, 0.7295, 0.669, "$v$", size=6.2, style="italic")

    box(fig, 0.872, 0.630, 0.108, 0.175)
    txt(fig, 0.926, 0.772, "$\\oplus$", size=7.0)
    txt(fig, 0.926, 0.742, "$+\\,b_0$", size=5.6)
    txt(fig, 0.926, 0.700, "$g_\\theta(I,T)\\,=$", size=5.6)
    txt(fig, 0.926, 0.668, "$a_\\phi(u)+b_\\psi(v)+b_0$", size=4.5)

    arrow(fig, 0.841, 0.7875, 0.870, 0.7875)
    arrow(fig, 0.841, 0.6575, 0.870, 0.6575)

    path(fig, [(0.668, 0.596), (0.668, 0.490), (0.743, 0.490)])

    box(fig, 0.745, 0.455, 0.100, 0.070)
    txt(fig, 0.795, 0.498, "Similarity", size=5.2, weight="bold")
    txt(fig, 0.795, 0.472, "$s(I,T)$", size=5.8)
    box(fig, 0.745, 0.360, 0.100, 0.070)
    txt(fig, 0.795, 0.403, "Standardization", size=4.7, weight="bold")
    txt(fig, 0.795, 0.377, "$z=(s-\\mu)/\\sigma$", size=5.4)
    arrow(fig, 0.795, 0.453, 0.795, 0.432)

    box(fig, 0.545, 0.200, 0.410, 0.100, lw=1.3)
    txt(fig, 0.750, 0.272, "Calibration", size=5.6, weight="bold")
    txt(fig, 0.750, 0.234, "$z' \\;=\\; z \\;-\\; \\lambda\\,g_\\theta(I,T)$",
        size=11.0)

    arrow(fig, 0.795, 0.358, 0.795, 0.302)
    arrow(fig, 0.926, 0.628, 0.926, 0.302)

    box(fig, 0.650, 0.090, 0.200, 0.062)
    txt(fig, 0.750, 0.120, "Calibrated Score  $z'$", size=5.8, weight="bold")
    arrow(fig, 0.750, 0.198, 0.750, 0.154)

    arrow(fig, 0.487, 0.560, 0.517, 0.560, lw=0.75, mut=5.2, ls=(0, (2.2, 1.5)))
    txt(fig, 0.502, 0.583, "Learned parameters", size=4.6, color=GREY)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "Fig2_v1.png", dpi=600, bbox_inches="tight")
    print("V1 baseline restored -> results/fig2/Fig2_v1.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
