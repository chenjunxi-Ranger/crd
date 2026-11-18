"""Feasibility Gate: how much signal exists in the token x patch family?

This script is **analysis only**.  It may read labels because it answers one
question: *is the information needed to clear a threshold present at all?*
Labels never enter ``src/`` and never influence a final score.

Reported per benchmark
----------------------
* single-feature AUROC (row-level) or within-pair accuracy (FOIL)
* cross-validated linear fusion (upper bound of the linear family)
* cross-validated gradient-boosting fusion (upper bound of the nonlinear family)
* gap to the hard threshold in CONSTRAINTS.md

Usage
-----
    python scripts/gate_oracle.py --bench coco
    python scripts/gate_oracle.py --bench foil --limit 2000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.features import FEATURE_KEYS, compute            # noqa: E402
from src.protocol import (CLIP_DIR, RESULTS_DIR, THRESHOLDS,  # noqa: E402
                          load_foil_val, load_split, write_json)


def setup_logger() -> logging.Logger:
    log = logging.getLogger("gate")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    return log


def _cv_scores(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """5-fold CV AUC for a linear and a nonlinear fusion (upper bounds)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    lin, gbm = [], []
    for tr, te in skf.split(x, y):
        if len(np.unique(y[te])) < 2:
            continue
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
        pipe.fit(x[tr], y[tr])
        lin.append(roc_auc_score(y[te], pipe.predict_proba(x[te])[:, 1]))
        g = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        g.fit(x[tr], y[tr])
        gbm.append(roc_auc_score(y[te], g.predict_proba(x[te])[:, 1]))
    return {"linear_cv_auc": float(np.mean(lin)), "linear_cv_sd": float(np.std(lin)),
            "gbm_cv_auc": float(np.mean(gbm)), "gbm_cv_sd": float(np.std(gbm))}


def _row_bench(args, enc: JointEncoder, log) -> dict:
    df = load_split(args.bench, "val", args.limit)
    log.info("%s val rows=%d", args.bench, len(df))
    st = enc.encode(df["path"].tolist(), df["caption"].tolist(),
                    batch_size=args.batch_size, num_workers=args.workers)
    feats = compute(st, enc.tokenizer, enc.grid)
    cosine = (st.pooled_img * st.pooled_txt).sum(axis=1)

    out = pd.DataFrame(feats)
    out["cosine"] = cosine
    out["label"] = df["label"].to_numpy()
    out["image_id"] = df["image_id"].to_numpy()
    csv = RESULTS_DIR / "gate" / f"{args.bench}_val_features.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv, index=False, encoding="utf-8-sig")

    y = out["label"].to_numpy()
    rows = []
    for k in FEATURE_KEYS + ["cosine"]:
        v = out[k].to_numpy()
        auc_pos = roc_auc_score(y, v)
        auc_neg = roc_auc_score(y, -v)
        rows.append({
            "feature": k,
            "auroc_as_is": float(auc_pos),
            "auroc_negated": float(auc_neg),
            "best_orientation_auroc": float(max(auc_pos, auc_neg)),
            "tie_rate": float(pd.Series(v).duplicated(keep=False).mean()),
        })
    single = pd.DataFrame(rows).sort_values("best_orientation_auroc", ascending=False)

    x = out[FEATURE_KEYS + ["cosine"]].to_numpy()
    fused = _cv_scores(x, y)
    fused_no_cos = _cv_scores(out[FEATURE_KEYS].to_numpy(), y)
    payload = {
        "bench": args.bench,
        "protocol": "row-level val AUROC",
        "n_val": int(len(out)),
        "prevalence": float(y.mean()),
        "threshold": THRESHOLDS[args.bench],
        "single_feature": single.to_dict(orient="records"),
        "fusion": fused,
        "fusion_without_cosine": fused_no_cos,
        "best_single": float(single["best_orientation_auroc"].max()),
        "best_single_feature": str(single.iloc[0]["feature"]),
        "gap_gbm": float(THRESHOLDS[args.bench] - fused["gbm_cv_auc"]),
        "csv": str(csv),
    }
    write_json(RESULTS_DIR / "gate" / f"{args.bench}_gate.json", payload)
    return payload


def _foil_bench(args, enc: JointEncoder, log) -> dict:
    df = load_foil_val(args.limit)
    log.info("foil val pairs=%d", len(df))
    paths = df["path"].tolist() * 2
    caps = df["positive"].tolist() + df["negative"].tolist()
    st = enc.encode(paths, caps, batch_size=args.batch_size, num_workers=args.workers)
    feats = compute(st, enc.tokenizer, enc.grid)
    n = len(df)
    cosine = (st.pooled_img * st.pooled_txt).sum(axis=1)

    rec = {"pair_id": df["pair_id"].to_numpy()}
    for k in FEATURE_KEYS + ["cosine"]:
        v = feats[k] if k in feats else cosine
        rec[f"{k}_pos"] = v[:n]
        rec[f"{k}_neg"] = v[n:]
    out = pd.DataFrame(rec)
    csv = RESULTS_DIR / "gate" / "foil_val_features.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv, index=False, encoding="utf-8-sig")

    rows = []
    for k in FEATURE_KEYS + ["cosine"]:
        diff = out[f"{k}_neg"].to_numpy() - out[f"{k}_pos"].to_numpy()
        # more suspicious in the negative caption = lower support / lower cosine
        correct = np.where(diff < 0, 1.0, np.where(diff > 0, 0.0, 0.5))
        rows.append({
            "feature": k,
            "accuracy_support_drop": float(correct.mean()),
            "tie_rate": float((diff == 0).mean()),
            "flip_accuracy": float(np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5)).mean()),
        })
    single = pd.DataFrame(rows).sort_values("accuracy_support_drop", ascending=False)

    # Supervised-transfer upper bound: fit a fusion on COCO *row* labels (val only)
    # and apply it to the FOIL pair rows.  The within-pair protocol has no negative
    # class by construction, so this is the only honest way to estimate how much a
    # supervised combination of these features could buy.
    transfer: dict[str, float | str | None] = {"available": False}
    coco_csv = RESULTS_DIR / "gate" / "coco_val_features.csv"
    if coco_csv.is_file():
        coco = pd.read_csv(coco_csv, encoding="utf-8-sig")
        x_train = coco[FEATURE_KEYS].to_numpy()
        y_train = coco["label"].to_numpy()
        x_pos = out[[f"{k}_pos" for k in FEATURE_KEYS]].to_numpy()
        x_neg = out[[f"{k}_neg" for k in FEATURE_KEYS]].to_numpy()
        for name, model in (
            ("linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
            ("gbm", HistGradientBoostingClassifier(max_iter=200, random_state=0)),
        ):
            model.fit(x_train, y_train)
            if name == "linear":
                s_pos, s_neg = model.decision_function(x_pos), model.decision_function(x_neg)
            else:
                s_pos = model.predict_proba(x_pos)[:, 1]
                s_neg = model.predict_proba(x_neg)[:, 1]
            acc = float(np.mean(np.where(s_neg > s_pos, 1.0,
                                        np.where(s_neg < s_pos, 0.0, 0.5))))
            transfer[f"{name}_transfer_acc"] = acc
        transfer["available"] = True
        transfer["note"] = ("fusion fitted on COCO-cat val rows (labels), applied to FOIL "
                            "val pairs; an upper bound, not the final method")
    else:
        transfer["note"] = "run --bench coco first to enable the transfer upper bound"

    payload = {
        "bench": "foil",
        "protocol": "within-pair val accuracy (support drop in negative)",
        "n_pairs": int(n),
        "threshold": THRESHOLDS["foil"],
        "single_feature": single.to_dict(orient="records"),
        "best_single": float(single["accuracy_support_drop"].max()),
        "best_single_feature": str(single.iloc[0]["feature"]),
        "gap_best_single": float(THRESHOLDS["foil"] - single["accuracy_support_drop"].max()),
        "supervised_transfer": transfer,
        "csv": str(csv),
    }
    write_json(RESULTS_DIR / "gate" / "foil_gate.json", payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    log = setup_logger()
    enc = JointEncoder(CLIP_DIR, args.device, log)
    payload = _foil_bench(args, enc, log) if args.bench == "foil" else _row_bench(args, enc, log)
    print(json.dumps({k: v for k, v in payload.items() if k != "single_feature"},
                     ensure_ascii=False, indent=2))
    df = pd.DataFrame(payload["single_feature"])
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
