"""Zoom on the two best panel-(a) candidates to verify the object count."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

FIG = Path(__file__).resolve().parents[2] / "results" / "fig1_coco"


def main() -> int:
    d = pd.read_csv(FIG / "fig1_coco_candidates.csv")
    rows = [3, 5]                      # stop-sign image, zebra image
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=170)
    for c, i in enumerate(rows):
        r = d.iloc[i]
        for rr, (p, cap, s, lab) in enumerate((
                (r.clean_image_path, r.clean_caption, r.clean_score, "clean"),
                (r.noisy_image_path, r.noisy_caption, r.noisy_score, "noisy"))):
            ax = axes[rr, c]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"row{i} {lab} s={s:.3f}  (true count of image: "
                         f"{r.noisy_true_count if lab=='noisy' else r.clean_count})",
                         fontsize=10)
            ax.set_xlabel(cap[:70], fontsize=9, wrap=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_coco_zoom.png", bbox_inches="tight")
    for i in rows:
        r = d.iloc[i]
        print(f"row{i}: clean='{r.clean_caption}' ({r.clean_image_path})")
        print(f"      noisy='{r.noisy_caption}' true={r.noisy_true_count} "
              f"says={r.noisy_count} ({r.noisy_image_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
