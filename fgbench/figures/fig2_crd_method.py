"""Figure 2 - CRD overview (single canonical column grid).

Both panels are laid out on the SAME grid, offset by PANEL_DX:

    input | Frozen Dual Encoder | heads | additive
     0.058 |        0.117       | 0.105 |  0.105

Sizes
    BW, BH      = 0.105, 0.080   every single-row module
    additive    = BW x (BH + ROW_PITCH)   spans exactly the two rows it merges
    container   = (BW + 2*PAD) x (2*BH + 2*PAD + 0.040)
Every module joined by a horizontal arrow shares the same height; every module
joined by a vertical arrow shares the same width.
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
W_IN, H_IN = 7.09, 3.95          # 180 x 100 mm

BW, BH = 0.105, 0.080            # standard module
PAD = 0.006                      # container padding
Y_TOP, Y_BOT = 0.785, 0.665      # row centres
ROW_PITCH = Y_TOP - Y_BOT        # 0.120
Y_SIM = 0.420                    # similarity row
PANEL_DX = 0.505                 # (b) = (a) + PANEL_DX

CW_IN, CW_DUAL, CW_ADD = 0.048, BW + 2 * PAD, BW
GAP = 0.020
X_IN, X_DUAL = 0.015, 0.083
X_HEAD = X_DUAL + CW_DUAL + 0.050          # 0.050 corridor for the u/v drops
X_ADD = X_HEAD + BW + GAP
X_SIM = X_DUAL + CW_DUAL - 0.030           # centred under the two drops (0.170)

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


def elbow(fig, pts, lw=0.75):
    xs, ys = zip(*pts)
    fig.add_artist(plt.Line2D(xs[:-1], ys[:-1], color=INK, lw=lw,
                              solid_capstyle="butt", zorder=3))
    arrow(fig, xs[-2], ys[-2], xs[-1], ys[-1], lw=lw)


def txt(fig, x, y, s, size=6, color=INK, ha="center", va="center",
        weight="normal", style="normal", ls_=1.2):
    return fig.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                    weight=weight, style=style, linespacing=ls_, zorder=4)


def block(fig, x, y, name, sym, size_name=5.2, size_sym=5.8):
    box(fig, x, y, BW, BH)
    txt(fig, x + BW / 2, y + BH / 2 + 0.014, name, size=size_name, weight="bold")
    txt(fig, x + BW / 2, y + BH / 2 - 0.018, sym, size=size_sym)


def dual_encoder(fig, x):
    """Dashed container + two identical frozen encoders."""
    top = Y_TOP + BH / 2 + PAD
    bot = Y_BOT - BH / 2 - PAD
    box(fig, x, bot, CW_DUAL, top - bot,
        fc="white", ec="#9a9a9a", lw=0.6, ls=(0, (2.4, 1.6)), r=0.008)
    txt(fig, x + CW_DUAL / 2, top + 0.014, "Frozen Dual Encoder",
        size=4.9, color=GREY)
    for cy, name in ((Y_TOP, "Image"), (Y_BOT, "Text")):
        box(fig, x + PAD, cy - BH / 2, BW, BH, fc=FROZEN)
        txt(fig, x + PAD + BW / 2, cy + 0.014, name, size=5.0, weight="bold")
        txt(fig, x + PAD + BW / 2, cy - 0.018, "Encoder", size=5.0, weight="bold")


def panel(fig, dx, input_title, input_body, input_set):
    """Everything that is identical in both panels."""
    xin, xdual = X_IN + dx, X_DUAL + dx
    xhead, xadd = X_HEAD + dx, X_ADD + dx

    box(fig, xin, 0.630, CW_IN, 0.190)
    txt(fig, xin + CW_IN / 2, 0.790, input_title, size=5.2, weight="bold")
    txt(fig, xin + CW_IN / 2, 0.748, input_body, size=5.0)
    txt(fig, xin + CW_IN / 2, 0.686, input_set, size=5.4)

    dual_encoder(fig, xdual)
    arrow(fig, xin + CW_IN + 0.001, Y_TOP, xdual - 0.001, Y_TOP)
    arrow(fig, xin + CW_IN + 0.001, Y_BOT, xdual - 0.001, Y_BOT)

    block(fig, xhead, Y_TOP - BH / 2, "Image Head", "$a_\\phi(u)$")
    block(fig, xhead, Y_BOT - BH / 2, "Text Head", "$b_\\psi(v)$")

    # main arrows with u / v, plus the two auxiliary drops into Similarity
    arrow(fig, xdual + CW_DUAL + 0.002, Y_TOP, xhead - 0.002, Y_TOP)
    arrow(fig, xdual + CW_DUAL + 0.011, Y_BOT, xhead - 0.002, Y_BOT)
    txt(fig, xdual + CW_DUAL + 0.022, Y_TOP + 0.019, "$u$", size=6.0,
        style="italic")
    txt(fig, xdual + CW_DUAL + 0.022, Y_BOT + 0.019, "$v$", size=6.0,
        style="italic")
    arrow(fig, xdual + CW_DUAL + 0.006, Y_TOP, xdual + CW_DUAL + 0.006,
          Y_SIM + BH, lw=0.7)
    arrow(fig, xdual + CW_DUAL + 0.038, Y_BOT, xdual + CW_DUAL + 0.038,
          Y_SIM + BH, lw=0.7)

    # additive score block: exactly two rows tall
    box(fig, xadd, Y_BOT - BH / 2, BW, BH + ROW_PITCH)
    txt(fig, xadd + BW / 2, Y_TOP - 0.013, "$\\oplus$", size=7.0)
    txt(fig, xadd + BW / 2, Y_TOP - 0.043, "$+\\,b_0$", size=5.6)
    txt(fig, xadd + BW / 2, Y_TOP - 0.085, "$g_\\theta(I,T)\\,=$", size=5.6)
    txt(fig, xadd + BW / 2, Y_TOP - 0.117,
        "$a_\\phi(u)+b_\\psi(v)+b_0$", size=4.6)
    arrow(fig, xhead + BW + 0.002, Y_TOP, xadd - 0.002, Y_TOP)
    arrow(fig, xhead + BW + 0.002, Y_BOT, xadd - 0.002, Y_BOT)

    # similarity branch: sits under the corridor so both drops land inside it
    xs = X_SIM + dx
    box(fig, xs, Y_SIM, BW, BH)
    txt(fig, xs + BW / 2, Y_SIM + 0.054, "Similarity", size=5.2, weight="bold")
    txt(fig, xs + BW / 2, Y_SIM + 0.022, "$s(I,T)=\\mathrm{sim}(u,v)$", size=4.6)
    box(fig, xs, Y_SIM - 0.105, BW, BH)
    txt(fig, xs + BW / 2, Y_SIM - 0.051, "Standardization", size=4.7,
        weight="bold")
    txt(fig, xs + BW / 2, Y_SIM - 0.083, "$z=(s-\\mu)/\\sigma$", size=5.4)
    arrow(fig, xs + BW / 2, Y_SIM - 0.002, xs + BW / 2, Y_SIM - 0.023)
    return xin, xdual, xhead, xadd, xs


def main() -> int:
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)

    txt(fig, 0.015, 0.962, "(a) Training: Learning Additive Score Components",
        size=7.2, weight="bold", ha="left")
    txt(fig, 0.520, 0.962, "(b) Inference: Pair-wise Score Calibration",
        size=7.2, weight="bold", ha="left")

    # ---------------- (a) training ----------------
    _, _, _, xadd_a, xs_a = panel(fig, 0.0, "Training", "image-text\npairs",
                                  "$\\{(I_i,T_i)\\}$")

    box(fig, xadd_a, 0.230, BW, BH)
    txt(fig, xadd_a + BW / 2, 0.284, "Huber Loss", size=5.2, weight="bold")
    txt(fig, xadd_a + BW / 2, 0.254, "$\\mathcal{L}(g_\\theta,\\,z)$", size=5.6)
    arrow(fig, xadd_a + BW / 2, 0.623, xadd_a + BW / 2, 0.312)
    elbow(fig, [(xs_a + BW / 2, Y_SIM - 0.107),
                (xs_a + BW / 2, 0.270),
                (xadd_a - 0.002, 0.270)])

    # ---------------- (b) inference ----------------
    _, _, _, xadd_b, xs_b = panel(fig, PANEL_DX, "Testing", "image-text\npairs",
                                  "$\\{(I,T)\\}$")

    box(fig, 0.545, 0.175, 0.410, 0.095, lw=1.3)
    txt(fig, 0.750, 0.242, "Calibration", size=5.6, weight="bold")
    txt(fig, 0.750, 0.206, "$z' \\;=\\; z \\;-\\; \\lambda\\,g_\\theta(I,T)$",
        size=11.0)

    arrow(fig, xs_b + BW / 2, Y_SIM - 0.107, xs_b + BW / 2, 0.272)
    arrow(fig, xadd_b + BW / 2, 0.623, xadd_b + BW / 2, 0.272)

    box(fig, 0.650, 0.070, 0.200, 0.060)
    txt(fig, 0.750, 0.100, "Calibrated Score  $z'$", size=5.8, weight="bold")
    arrow(fig, 0.750, 0.173, 0.750, 0.134)

    arrow(fig, 0.492, 0.560, 0.518, 0.560, lw=0.75, mut=5.2, ls=(0, (2.2, 1.5)))
    txt(fig, 0.505, 0.583, "Learned parameters", size=4.6, color=GREY)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"Fig2.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
    print("wrote", OUT / "Fig2.{svg,pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
