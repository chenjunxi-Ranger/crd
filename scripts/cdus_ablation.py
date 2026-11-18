"""How much does CDUS actually add inside the fusion ceiling?

Same cached features and the same 5-fold CV as ``cdus_fusion.py``; only the CDUS
columns (D4, D12) are toggled.  Reports all / without-CDUS / CDUS-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402


def cv_auc(x, y, n_splits=5):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    vals = []
    for tr, te in skf.split(x, y):
        if len(np.unique(y[te])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(x[tr], y[tr])
        vals.append(roc_auc_score(y[te], m.predict_proba(x[te])[:, 1]))
    return float(np.mean(vals))


def main() -> int:
    out = {}
    for bench in ("coco", "flickr", "mmimdb"):
        bank = np.load(RESULTS_DIR / "gate" / f"final_feats2_{bench}.npz")
        d = np.load(RESULTS_DIR / "gate" / f"cdus_scores_{bench}.npz")
        n = len(d["s"])
        y = bank["label"][:n]
        base = np.column_stack([
            -bank["cos"][:n], bank["rx"][:n] - bank["cos"][:n],
            bank["rc"][:n] - bank["cos"][:n],
            bank["mxt"][:n] - bank["cos"][:n], bank["mxi"][:n] - bank["cos"][:n],
            -bank["lp"][:n]])
        cdus = np.column_stack([-d["d4"], -d["d12"]])
        out[bench] = {"n": int(n),
                      "all": cv_auc(np.column_stack([base, cdus]), y),
                      "no_cdus": cv_auc(base, y),
                      "cdus_only": cv_auc(cdus, y)}
        out[bench]["delta"] = out[bench]["all"] - out[bench]["no_cdus"]
        print(bench, json.dumps(out[bench], ensure_ascii=False), flush=True)

    bank = np.load(RESULTS_DIR / "gate" / "final_feats2_foil.npz")
    d = np.load(RESULTS_DIR / "gate" / "cdus_scores_foil.npz")
    ves = np.load(RESULTS_DIR / "gate" / "ves_pairs_VES-1.npz")
    n = min(len(d["s_pos"]), len(ves["s_pos"]))
    y = np.r_[np.zeros(n), np.ones(n)]

    def feats(sfx, with_cdus):
        c = np.column_stack([
            -bank["cos" + sfx][:n], bank["rx" + sfx][:n] - bank["cos" + sfx][:n],
            bank["rc" + sfx][:n] - bank["cos" + sfx][:n],
            bank["mxt" + sfx][:n] - bank["cos" + sfx][:n],
            bank["mxi" + sfx][:n] - bank["cos" + sfx][:n],
            -bank["lp" + sfx][:n]])
        e = -ves["e_pos" if sfx == "_pos" else "e_neg"][:n, None]
        cd = -d["d_pos" if sfx == "_pos" else "d_neg"][:n, None]
        parts = [c, e] + ([cd] if with_cdus else [])
        return np.column_stack(parts)

    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(n), 5)

    def pair_cv(with_cdus):
        xp, xn = feats("_pos", with_cdus), feats("_neg", with_cdus)
        x = np.vstack([xp, xn])
        yy = np.r_[np.zeros(n), np.ones(n)]
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            mtr = np.zeros(2 * n, bool); mte = np.zeros(2 * n, bool)
            mtr[tr] = True; mtr[n + tr] = True
            mte[te] = True; mte[n + te] = True
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            m.fit(x[mtr], yy[mtr])
            sp, sn = m.decision_function(xp[te]), m.decision_function(xn[te])
            accs.append(float(np.mean(np.where(sn > sp, 1.0,
                                               np.where(sn < sp, 0.0, 0.5)))))
        return float(np.mean(accs))

    out["foil"] = {"n_pairs": int(n), "all": pair_cv(True),
                   "no_cdus": pair_cv(False)}
    out["foil"]["delta"] = out["foil"]["all"] - out["foil"]["no_cdus"]
    print("foil", json.dumps(out["foil"], ensure_ascii=False), flush=True)

    write_json(RESULTS_DIR / "gate" / "cdus_ablation.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
