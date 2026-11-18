"""Standalone export of panel (c): clean vs noisy raw-score distributions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parents[2] / "results" / "fig1"
CLEAN, NOISY = "#2f7d4f", "#e07b39"

mpl.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "font.size": 8})


def main() -> int:
    st = json.loads((FIG / "fig1_stats.json").read_text(encoding="utf-8"))
    d = np.load(FIG / "fig1_dist.npz")
    fig, ax = plt.subplots(figsize=(3.3, 2.4), dpi=300)
    lo = min(d["clean"].min(), d["noisy"].min())
    hi = max(d["clean"].max(), d["noisy"].max())
    bins = np.linspace(lo, hi, 60)
    ax.hist(d["clean"], bins=bins, density=True, color=CLEAN, alpha=0.5, lw=0,
            label=f"clean (n={st['clean']['n']:,})")
    ax.hist(d["noisy"], bins=bins, density=True, color=NOISY, alpha=0.5, lw=0,
            label=f"noisy (n={st['noisy']['n']:,})")
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.set_xlabel("raw CLIP cosine similarity")
    ax.set_ylabel("density")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.98, 0.55, f"AUROC = {st['auroc_raw']:.3f}", transform=ax.transAxes,
            ha="right", fontsize=8)
    ax.text(0.98, 0.44, f"P(noisy > clean) = {st['p_noisy_above_clean']:.1%}",
            transform=ax.transAxes, ha="right", fontsize=7, color="#555555")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"fig1c_distribution.{ext}", bbox_inches="tight")
    print("wrote fig1c_distribution.{png,pdf,svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
