"""Large-view shortlist for human verification of the final picks."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

FIG = Path(__file__).resolve().parents[2] / "results" / "fig1"
PICK = [0, 1, 3, 4, 5, 7, 8, 11, 12, 13]


def main() -> int:
    df = pd.read_csv(FIG / "fig1_candidates.csv")
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), dpi=140)
    for c, i in enumerate(PICK):
        r = df.iloc[i]
        blk, col = divmod(c, 5)
        for rr, (p, cap, s, lab) in enumerate((
                (r.clean_image_path, r.clean_caption, r.clean_score, "clean"),
                (r.noisy_image_path, r.noisy_caption, r.noisy_score, "noisy"))):
            ax = axes[blk * 2 + rr, col]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"#{i} {lab} s={s:.3f}", fontsize=10,
                         color="#2f7d4f" if lab == "clean" else "#e07b39")
            ax.set_xlabel(cap[:58], fontsize=9, wrap=True)
    fig.tight_layout()
    out = FIG / "fig1_shortlist.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
