"""Figure 1 - why raw cross-modal similarity is hard to compare globally.

(a) cross-pair ranking conflict, (b) 2x2 reciprocal cycle with image/text swap
differences, (c) overlapping clean/noisy distributions.
All numbers come from results/fig1/*.json (fig1_extract.py, fig1_stats.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "fig1"

CLEAN, NOISY = "#2f7d4f", "#e07b39"
INK, GREY, RED = "#1a1a1a", "#8a8a8a", "#b03a2e"
CLEAN_IDX, NOISY_IDX = 7, 3

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})


def add_img(fig, path, rect, border=None):
    ax = fig.add_axes(rect)
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7 if border is None else 1.3)
        s.set_edgecolor(GREY if border is None else border)
    return ax


def wrap(text, width):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur); cur = w
    lines.append(cur)
    return "\n".join(lines)


def main() -> int:
    cand = pd.read_csv(FIG / "fig1_candidates.csv")
    cyc = json.loads((FIG / "fig1_cycle.json").read_text(encoding="utf-8"))
    st = json.loads((FIG / "fig1_stats.json").read_text(encoding="utf-8"))
    dist = np.load(FIG / "fig1_dist.npz")

    cr, nr = cand.iloc[CLEAN_IDX], cand.iloc[NOISY_IDX]
    fig = plt.figure(figsize=(7.09, 4.13), dpi=300)

    # ================= (a) =================
    fig.text(0.015, 0.975, "(a) Cross-pair ranking conflicts",
             fontsize=7.8, weight="bold", color=INK, va="top")
    for rec, col, tag, ytop in ((cr, CLEAN, "clean", 0.80),
                                (nr, NOISY, "noisy", 0.44)):
        path = rec.clean_image_path if tag == "clean" else rec.noisy_image_path
        cap = rec.clean_caption if tag == "clean" else rec.noisy_caption
        sc = rec.clean_score if tag == "clean" else rec.noisy_score
        add_img(fig, path, [0.015, ytop - 0.30, 0.155, 0.30], border=col)
        fig.text(0.185, ytop - 0.035, tag, fontsize=7, color=col, weight="bold")
        fig.text(0.185, ytop - 0.095, f"s = {sc:.3f}", fontsize=10, color=col,
                 weight="bold")
        fig.text(0.185, ytop - 0.135, wrap(cap, 22), fontsize=6.2, color=INK,
                 va="top", linespacing=1.3)

    arr = FancyArrowPatch((0.318, 0.475), (0.318, 0.685), arrowstyle="-|>",
                          mutation_scale=7, lw=0.9, color=RED)
    fig.add_artist(arr)
    fig.text(0.326, 0.575, "higher\nscore,\nwrong\npair", fontsize=6.0, color=RED,
             va="center", linespacing=1.25)

    # ================= (b) =================
    fig.text(0.375, 0.975, "(b) Image- and text-conditioned score variation",
             fontsize=7.8, weight="bold", color=INK, va="top")

    M = np.array(cyc["score_matrix"]); L = np.array(cyc["label_matrix"])
    caps, imgs = cyc["captions"], cyc["images"]
    x0, cw, ch = 0.445, 0.105, 0.150
    ytop = 0.735
    for i in range(2):
        add_img(fig, imgs[i]["path"],
                [0.368, ytop - i * (ch + 0.02) - ch, 0.055, ch])
        fig.text(0.396, ytop - i * (ch + 0.02) - ch / 2, f"$I_{'AB'[i]}$",
                 fontsize=7.5, ha="center", va="center", color="white",
                 bbox=dict(boxstyle="round,pad=0.12", fc=INK, ec="none", alpha=0.75))
        for j in range(2):
            col = CLEAN if L[i, j] == 0 else NOISY
            yy = ytop - i * (ch + 0.02) - ch
            ax = fig.add_axes([x0 + j * cw, yy, cw - 0.006, ch])
            ax.set_xticks([]); ax.set_yticks([])
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   facecolor=col, alpha=0.10, edgecolor=col, lw=1.0))
            ax.text(0.5, 0.60, f"{M[i, j]:.3f}", ha="center", va="center",
                    fontsize=9.5, color=col, weight="bold")
            ax.text(0.5, 0.22, "clean" if L[i, j] == 0 else "noisy", ha="center",
                    va="center", fontsize=6.2, color=col)

    for j in range(2):
        fig.text(x0 + j * cw + cw / 2 - 0.003, ytop + 0.098,
                 f"$T_{'AB'[j]}$", fontsize=7.5, ha="center", va="bottom")
        fig.text(x0 + j * cw + cw / 2 - 0.003, ytop + 0.086,
                 wrap(caps[j], 18), fontsize=5.0, ha="center", va="top",
                 color=GREY, style="italic", linespacing=1.15)

    d_img, d_img2 = M[0, 0] - M[1, 0], M[0, 1] - M[1, 1]
    d_txt, d_txt2 = M[0, 0] - M[0, 1], M[1, 0] - M[1, 1]
    gap = st["clean"]["mean"] - st["noisy"]["mean"]
    lines = [
        (f"swap image, text fixed   $\\Delta$ = {d_img:+.3f} / {d_img2:+.3f}", INK),
        (f"swap text, image fixed   $\\Delta$ = {d_txt:+.3f} / {d_txt2:+.3f}", INK),
        (f"mean |$\\Delta$| over {st['n_cycles']:,} cycles:  image "
         f"{st['swap_image_abs_mean']:.3f},  text {st['swap_text_abs_mean']:.3f}", GREY),
        (f"clean$-$noisy mean gap: {gap:.3f}  (smaller than either swap)", RED),
    ]
    for k, (txt, col) in enumerate(lines):
        fig.text(0.368, 0.335 - k * 0.045, txt, fontsize=6.3, color=col)

    # ================= (c) =================
    fig.text(0.762, 0.975, "(c) Overlapping distributions", fontsize=7.8,
             weight="bold", color=INK, va="top")
    axc = fig.add_axes([0.775, 0.285, 0.205, 0.60])
    lo = min(dist["clean"].min(), dist["noisy"].min())
    hi = max(dist["clean"].max(), dist["noisy"].max())
    bins = np.linspace(lo, hi, 55)
    axc.hist(dist["clean"], bins=bins, density=True, color=CLEAN, alpha=0.45,
             lw=0, label="clean")
    axc.hist(dist["noisy"], bins=bins, density=True, color=NOISY, alpha=0.45,
             lw=0, label="noisy")
    axc.set_xlim(lo, hi)
    axc.set_xlabel("raw CLIP cosine", fontsize=6.6)
    axc.set_ylabel("density", fontsize=6.6)
    axc.set_yticks([])
    axc.tick_params(labelsize=5.8, length=2)
    axc.spines[["top", "right", "left"]].set_visible(False)
    axc.legend(loc="upper left", fontsize=5.6, handlelength=1.0,
               borderaxespad=0.15, labelspacing=0.15)
    axc.text(0.97, 0.95, f"AUROC = {st['auroc_raw']:.3f}", transform=axc.transAxes,
             ha="right", va="top", fontsize=6.6, color=INK,
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    axc.text(0.97, 0.85, f"P(noisy > clean)\n= {st['p_noisy_above_clean']:.1%}",
             transform=axc.transAxes, ha="right", va="top", fontsize=5.9,
             color=GREY, linespacing=1.2,
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))

    out = FIG / "Fig1"
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    print("wrote", out)
    print("a:", cr.clean_pair_id, cr.noisy_pair_id)
    print("b:", cyc["cycle_id"], M.tolist(), L.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
