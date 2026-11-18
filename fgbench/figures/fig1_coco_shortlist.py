"""Shortlist + large view of the easiest-to-count COCO Quantity candidates."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

FIG = Path(__file__).resolve().parents[2] / "results" / "fig1_coco"
CLEAN, NOISY = "#2f7d4f", "#e07b39"


def main() -> int:
    d = pd.read_csv(FIG / "fig1_coco_candidates.csv")
    d = d[d.noisy_true_count.notna() & d.clean_count.notna()]
    d = d[(d.noisy_true_count <= 4) & (d.clean_count <= 4)]
    d = d[d.noisy_true_count != d.clean_count].reset_index(drop=True)
    d.to_csv(FIG / "fig1_coco_shortlist.csv", index=False, encoding="utf-8-sig")
    print("shortlist rows", len(d))
    for i, r in d.head(12).iterrows():
        print(f"row{i} gap={r.gap:.3f} clean({r.clean_score:.3f}, n={int(r.clean_count)}) "
              f"{r.clean_caption[:50]}")
        print(f"        noisy({r.noisy_score:.3f}, true n={int(r.noisy_true_count)}, "
              f"says n={int(r.noisy_count)}) {r.noisy_caption[:50]}")

    n = min(6, len(d))
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4), dpi=160)
    for c in range(n):
        r = d.iloc[c]
        for rr, (p, cap, s, lab, col) in enumerate((
                (r.clean_image_path, r.clean_caption, r.clean_score, "clean", CLEAN),
                (r.noisy_image_path, r.noisy_caption, r.noisy_score, "noisy", NOISY))):
            ax = axes[rr, c] if n > 1 else axes[rr]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{lab}  s={s:.3f}", fontsize=9, color=col)
            ax.set_xlabel(cap[:52], fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_coco_shortlist.png", bbox_inches="tight")
    print("wrote shortlist png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
