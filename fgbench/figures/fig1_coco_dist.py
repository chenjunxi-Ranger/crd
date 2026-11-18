"""Standalone clean/noisy distribution (analysis material, NOT part of Figure 1)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

FIG = Path(__file__).resolve().parents[2] / "results" / "fig1_coco"
CLEAN, NOISY = "#2f7d4f", "#e07b39"

mpl.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "font.size": 8})


def main() -> int:
    d = np.load(FIG / "fig1_coco_dist.npz")
    clean, noisy = d["clean"], d["noisy"]
    y = np.r_[np.zeros(len(clean)), np.ones(len(noisy))]
    s = np.r_[clean, noisy]
    auc = roc_auc_score(y, -s)
    fig, ax = plt.subplots(figsize=(3.3, 2.4), dpi=300)
    bins = np.linspace(s.min(), s.max(), 32)
    ax.hist(clean, bins=bins, density=True, color=CLEAN, alpha=0.55, lw=0,
            label=f"clean (n={len(clean)})")
    ax.hist(noisy, bins=bins, density=True, color=NOISY, alpha=0.55, lw=0,
            label=f"noisy (n={len(noisy)})")
    ax.set_yticks([])
    ax.set_xlabel("raw CLIP cosine similarity")
    ax.set_ylabel("density")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.98, 0.58, f"AUROC = {auc:.3f}", transform=ax.transAxes, ha="right",
            fontsize=8)
    ax.text(0.98, 0.46, f"mean clean {clean.mean():.3f} / noisy {noisy.mean():.3f}",
            transform=ax.transAxes, ha="right", fontsize=6.6, color="#555555")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"fig1c_coco_distribution.{ext}", bbox_inches="tight")
    print(f"wrote distribution; AUROC={auc:.4f} "
          f"clean {clean.mean():.4f} noisy {noisy.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
