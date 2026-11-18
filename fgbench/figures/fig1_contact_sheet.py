"""Contact sheet of candidate ranking-conflict pairs (human inspection step)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

FIG = Path(__file__).resolve().parents[2] / "results" / "fig1"
CLEAN, NOISY = "#2f7d4f", "#e07b39"
NROW = 20


def main() -> int:
    df = pd.read_csv(FIG / "fig1_candidates.csv").head(NROW)
    fig, axes = plt.subplots(NROW, 2, figsize=(7.2, 1.05 * NROW), dpi=200)
    for r, row in df.iterrows():
        for c, (path, cap, sc, lab) in enumerate((
                (row.clean_image_path, row.clean_caption, row.clean_score, "clean"),
                (row.noisy_image_path, row.noisy_caption, row.noisy_score, "noisy"))):
            ax = axes[r, c]
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            col = CLEAN if lab == "clean" else NOISY
            for s in ax.spines.values():
                s.set_edgecolor(col); s.set_linewidth(2.2)
            ax.set_title(f"{lab}  s={sc:.3f}", fontsize=7, color=col, pad=2)
            ax.set_xlabel(cap[:52] + ("…" if len(cap) > 52 else ""),
                          fontsize=5.5, labelpad=1)
        axes[r, 0].set_ylabel(f"#{r}  Δ={row.gap:.3f}", fontsize=6.5, rotation=0,
                              ha="right", va="center", labelpad=22)
    fig.suptitle("Candidate cross-pair ranking conflicts (clean left, noisy right)",
                 fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 0.985))
    out = FIG / "fig1_contact_sheet.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
