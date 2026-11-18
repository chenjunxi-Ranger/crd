"""Contact sheet for COCO Quantity candidates + the selected 2x2 cycle."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

FIG = Path(__file__).resolve().parents[2] / "results" / "fig1_coco"
CLEAN, NOISY = "#2f7d4f", "#e07b39"


def main() -> int:
    df = pd.read_csv(FIG / "fig1_coco_candidates.csv")
    cyc = json.loads((FIG / "fig1_coco_cycle.json").read_text(encoding="utf-8"))
    n = min(18, len(df))
    fig, axes = plt.subplots(6, 6, figsize=(15, 11), dpi=140)
    for i in range(n):
        r = df.iloc[i]
        blk, col = i // 3, (i % 3) * 2
        for k, (p, cap, s, lab, cc) in enumerate((
                (r.clean_image_path, r.clean_caption, r.clean_score, "clean", CLEAN),
                (r.noisy_image_path, r.noisy_caption, r.noisy_score, "noisy", NOISY))):
            ax = axes[blk, col + k]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(cc); sp.set_linewidth(2)
            ax.set_title(f"#{i} {lab}  s={s:.3f}", fontsize=9, color=cc)
            ax.set_xlabel(cap[:56], fontsize=7.5, wrap=True)
    for ax in axes[3:, :]:
        for a in ax:
            a.axis("off")
    # selected cycle in the lower block
    base = 3
    for k, im in enumerate(cyc["images"]):
        ax = axes[base, k]
        ax.imshow(Image.open(im["path"]).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"cycle {im['role']} {im['image_id']}", fontsize=9)
    axes[base, 2].axis("off")
    txt = (f"cycle {cyc['cycle_id']}\n"
           f"scores {cyc['score_matrix']}\nlabels {cyc['label_matrix']}\n"
           f"dI {cyc['deltas']['dI_ta']:+.3f}/{cyc['deltas']['dI_tb']:+.3f}  "
           f"dT {cyc['deltas']['dT_ia']:+.3f}/{cyc['deltas']['dT_ib']:+.3f}\n"
           f"same-sign img={cyc['same_sign_img']} txt={cyc['same_sign_txt']}")
    axes[base, 2].text(0, 0.9, txt, fontsize=9, va="top", family="monospace")
    for k, cap in enumerate(cyc["captions"]):
        axes[base + 1, k * 2].axis("off")
        axes[base + 1, k * 2].text(0, 0.8, f"T{'AB'[k]}: {cap}", fontsize=9, wrap=True)
    for k in (1,):
        axes[base + 1, k].axis("off")
    fig.tight_layout()
    out = FIG / "fig1_coco_contact_sheet.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
