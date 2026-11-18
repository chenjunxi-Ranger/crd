"""Figures for the frozen formula: ROC (val) + gap-to-threshold bars.

All data are read from ``results/bench/*.csv`` (no numbers typed here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, THRESHOLDS   # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LABEL = {"coco": "MSCOCO-cat", "flickr": "Flickr30k-noun", "mmimdb": "MM-IMDb-cat"}


def main() -> int:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    rows = []
    for bench, name in LABEL.items():
        path = RESULTS_DIR / "bench" / f"{bench}_val.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        auc = roc_auc_score(df["label"], df["score"])
        fpr, tpr, _ = roc_curve(df["label"], df["score"])
        ax.plot(fpr, tpr, lw=1.6, label=f"{name}  AUROC={auc:.4f}")
        rows.append({"bench": name, "metric": "AUROC", "value": auc,
                     "threshold": THRESHOLDS[bench], "gap": THRESHOLDS[bench] - auc})
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Frozen single formula — val ROC")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "roc_val.png", dpi=200)
    plt.close(fig)

    foil_path = RESULTS_DIR / "bench" / "foil_val.json"
    if foil_path.is_file():
        import json
        d = json.loads(foil_path.read_text(encoding="utf-8"))
        rows.append({"bench": "FOIL-COCO", "metric": "within-pair acc",
                     "value": d["accuracy"], "threshold": d["threshold"],
                     "gap": d["gap"]})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "bench" / "main_table.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    colors = ["#4C72B0" if g >= 0 else "#C44E52" for g in df["gap"]]
    ax.barh(df["bench"], df["gap"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    for i, (g, v) in enumerate(zip(df["gap"], df["value"])):
        ax.text(g / 2, i, f"{v:.4f} (gap {g:+.4f})", va="center",
                ha="center", fontsize=8, color="white")
    ax.set_xlabel("value − threshold (negative = below)")
    ax.set_title("Frozen formula vs hard thresholds (val)")
    fig.tight_layout()
    fig.savefig(FIG / "gap_to_threshold.png", dpi=200)
    plt.close(fig)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
