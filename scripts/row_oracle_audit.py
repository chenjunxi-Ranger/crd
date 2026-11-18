"""Row-wise Oracle decomposability audit.

u_i = mean_{j: y_j != y_i} [ I(CSLS_ij correct) - I(CLIP_ij correct) ]

Questions
  A. how many records have a positive net benefit, and how large is it
  B. is u_i stable across the opposite-label pool (5 fixed folds):
     sign agreement, between-record vs within-record variance
  C. can u_i be explained by LABEL-FREE per-record features
     (s, h, rx, rc, max-margin, top-k gap) -> Spearman, direction consistency

Also re-checks the reporting issues: 2x2 vs 4-way "all wrong", and the bin-wise
AUC contribution additivity with unrounded values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402

FOLDS = 5


def _u_matrix(sel_susp, opp_susp, sel_cos, opp_cos):
    """delta_ij for one set of records against the opposite-label pool."""
    csls_ok = sel_susp[:, None] < opp_susp[None, :]
    clip_ok = sel_cos[:, None] > opp_cos[None, :]
    return csls_ok.astype(np.int8) - clip_ok.astype(np.int8)


def main() -> int:
    out = {}
    for bench in ("coco", "flickr", "mmimdb"):
        r = np.load(RESULTS_DIR / "gate" / f"rsg_{bench}.npz")
        cos, h, y = r["cos"], r["h"], r["label"]
        n = len(cos)
        d = np.load(RESULTS_DIR / "gate" / f"cdus_scores_{bench}.npz")
        b = np.load(RESULTS_DIR / "gate" / f"bcsc_{bench}.npz")
        bank = np.load(RESULTS_DIR / "gate" / f"final_feats2_{bench}.npz")
        n = min(n, len(d["s"]), len(b["s"]), len(bank["cos"]))
        cos, h, y = cos[:n], h[:n], y[:n]
        csls = h - 2 * cos
        c = np.where(y == 0)[0]
        nz = np.where(y == 1)[0]

        # ---- per-record u_i
        u = np.zeros(n)
        dmat_c = _u_matrix(csls[c], csls[nz], cos[c], cos[nz])      # clean vs noisy
        dmat_n = _u_matrix(csls[nz], csls[c], cos[nz], cos[c])      # noisy vs clean
        u[c] = dmat_c.mean(1)
        u[nz] = dmat_n.mean(1)

        # ---- 5-fold stability of u_i over the opposite-label pool
        fold_assign_c = np.arange(len(nz)) % FOLDS
        fold_assign_n = np.arange(len(c)) % FOLDS
        uf = np.zeros((FOLDS, n))
        for k in range(FOLDS):
            uf[k, c] = dmat_c[:, fold_assign_c == k].mean(1)
            uf[k, nz] = dmat_n[:, fold_assign_n == k].mean(1)
        signs = np.sign(uf)
        agree = (np.abs(signs.sum(0)) == FOLDS).mean()             # all five folds agree
        major = (signs.mean(0) > 0)
        between = float(np.var(u))
        within = float(np.mean(np.var(uf, axis=0)))

        # ---- label-free features
        feats = {
            "s": cos,
            "h": h,
            "rx": bank["rx"][:n],
            "rc": bank["rc"][:n],
            "max_margin_img": bank["mxt"][:n] - cos,
            "max_margin_txt": bank["mxi"][:n] - cos,
        }
        corr = {}
        for name, v in feats.items():
            rho_c = spearmanr(v[c], u[c]).statistic
            rho_n = spearmanr(v[nz], u[nz]).statistic
            corr[name] = {"clean": float(rho_c), "noisy": float(rho_n)}

        # ---- reporting fixes
        ok_clip = cos[c][:, None] > cos[nz][None, :]
        ok_csls = csls[c][:, None] < csls[nz][None, :]
        ok_cdus = (-d["d4"][:n])[c][:, None] > (-d["d4"][:n])[nz][None, :]
        ok_bcsc = b["rit"][:n][c][:, None] < b["rit"][:n][nz][None, :]
        both_wrong_2 = (~ok_clip & ~ok_csls)
        all_wrong_4 = both_wrong_2 & ~ok_cdus & ~ok_bcsc

        # bin additivity with unrounded values
        absm = np.abs(cos[c][:, None] - cos[nz][None, :])
        edges = [0, 0.002, 0.005, 0.01, 0.02, np.inf]
        contrib_c, contrib_s = [], []
        for i in range(5):
            m = (absm >= edges[i]) & (absm < edges[i + 1])
            contrib_c.append(ok_clip[m].mean() * m.sum() / m.size)
            contrib_s.append(ok_csls[m].mean() * m.sum() / m.size)
        out[bench] = {
            "n": int(n), "n_clean": int(len(c)), "n_noisy": int(len(nz)),
            "u_mean": float(u.mean()), "u_share_positive": float((u > 0).mean()),
            "u_share_negative": float((u < 0).mean()),
            "u_share_zero": float((u == 0).mean()),
            "fold_sign_agreement": float(agree),
            "between_record_var": between, "within_record_var": within,
            "spearman": corr,
            "wrong": {"clip_and_csls": float(both_wrong_2.mean()),
                      "all_four": float(all_wrong_4.mean())},
            "bin_additivity": {
                "sum_clip": float(np.sum(contrib_c)),
                "sum_csls": float(np.sum(contrib_s)),
                "global_clip": float(ok_clip.mean()),
                "global_csls": float(ok_csls.mean()),
                "abs_error": float(abs(np.sum(contrib_s) - ok_csls.mean())),
            },
        }
        print(bench, json.dumps({k: v for k, v in out[bench].items() if k != "spearman"},
                                ensure_ascii=False), flush=True)
        print("   spearman", {k: (round(v["clean"], 3), round(v["noisy"], 3))
                              for k, v in corr.items()}, flush=True)

    write_json(RESULTS_DIR / "gate" / "row_oracle_audit.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
