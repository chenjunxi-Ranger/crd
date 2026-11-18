"""GMM-based filtering baseline (Chen et al., TOIS 2024 style).

Fit a 2-component Gaussian mixture on TRAIN CLIP similarities, use the posterior
of the low-mean component as the "noisy" score, evaluate on TEST.

Note: the posterior is a monotone function of the similarity when the two
components are ordered, so AUROC is identical to cosine by construction --
the experiment verifies that and reports the estimated noise rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fgbench.common import jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
FACETS = [("VG", "Attribute", "crdbb_clip_VG_Attribute_c8000",
           "crdbb_clip_VG_Attribute"),
          ("VG", "Relation", "crdbb_clip_VG_Relation_c2000",
           "crdbb_clip_VG_Relation"),
          ("COCO", "Quantity", "crdbb_clip_COCO_Quantity_c2500",
           "crdbb_clip_COCO_Quantity"),
          ("COCO", "Entity", "crdbb_clip_COCO_Entity_c8000",
           "crdbb_clip_COCO_Entity")]


def main() -> int:
    rows = []
    for ds, facet, ptr, pte in FACETS:
        tr = np.load(OUT / f"{ptr}_train.npz", allow_pickle=True)
        te = np.load(OUT / f"{pte}_test.npz", allow_pickle=True)
        dtr = pd.read_parquet(OUT / f"{ptr}_train.meta.parquet")
        dte = pd.read_parquet(OUT / f"{pte}_test.meta.parquet")
        s_tr = (tr["U"] * tr["V"]).sum(1)
        s_te = (te["U"] * te["V"]).sum(1)
        y_te = dte.pair_label.to_numpy()

        gm = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(
            s_tr.reshape(-1, 1))
        order = np.argsort(gm.means_.ravel())     # component 0 = lowest mean
        noisy_comp = int(order[0])
        post_tr = gm.predict_proba(s_tr.reshape(-1, 1))[:, noisy_comp]
        post_te = gm.predict_proba(s_te.reshape(-1, 1))[:, noisy_comp]

        np.savez_compressed(OUT / f"score_gmm_{ds}_{facet}.npz",
                            cycle_id=dte.cycle_id.values, y=y_te,
                            score=post_te, cos=-s_te)
        rows.append({
            "dataset": ds, "facet": facet,
            "auc_cos": round(float(roc_auc_score(y_te, -s_te)), 4),
            "auc_gmm": round(float(roc_auc_score(y_te, post_te)), 4),
            "ap_cos": round(float(average_precision_score(y_te, -s_te)), 4),
            "ap_gmm": round(float(average_precision_score(y_te, post_te)), 4),
            "noise_rate_train": round(float((post_tr > 0.5).mean()), 4),
            "noise_rate_test": round(float((post_te > 0.5).mean()), 4),
            "true_train_noise": round(float(dtr.pair_label.mean()), 4),
            "means": [round(float(m), 4) for m in np.sort(gm.means_.ravel())],
        })
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gmm_filter.csv", index=False, encoding="utf-8-sig")
    jdump({"rows": rows}, OUT / "gmm_filter.json")
    print(df[["dataset", "facet", "auc_cos", "auc_gmm", "noise_rate_test"]]
          .to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
