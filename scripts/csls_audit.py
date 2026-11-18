"""Global pairwise contribution audit for the bank-side CSLS.

For every bench, uses ALL clean x noisy pairs (not only the matched band) and
decomposes the pairwise correctness by the CLIP margin |s+ - s-|:

  B1 [0,0.002) B2 [0.002,0.005) B3 [0.005,0.01) B4 [0.01,0.02) B5 >=0.02

Per bin: pair share, CSLS conditional accuracy, rescue/hurt vs CLIP, and the
bin's contribution to the overall pairwise AUC (mean correct x share).

Also computes the global joint oracle over the candidate set:
  P(exists a candidate k with delta_k > 0)
and the 2x2 table CLIP x CSLS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, write_json   # noqa: E402

EDGES = [0.0, 0.002, 0.005, 0.01, 0.02, np.inf]
NAMES = ["B1", "B2", "B3", "B4", "B5"]


def main() -> int:
    out = {}
    for bench in ("coco", "flickr", "mmimdb"):
        r = np.load(RESULTS_DIR / "gate" / f"rsg_{bench}.npz")
        cos, h, y = r["cos"], r["h"], r["label"]
        n = len(cos)
        d = np.load(RESULTS_DIR / "gate" / f"cdus_scores_{bench}.npz")
        b = np.load(RESULTS_DIR / "gate" / f"bcsc_{bench}.npz")
        n = min(n, len(d["s"]), len(b["s"]))
        cos, h, y = cos[:n], h[:n], y[:n]
        csls = h - 2 * cos                          # suspicion, higher = worse
        cdus = -d["d4"][:n]
        bcsc = b["rit"][:n]

        c = np.where(y == 0)[0]
        nz = np.where(y == 1)[0]
        # positive margin = CLIP ranks the clean above the noisy (correct)
        ds = cos[c][:, None] - cos[nz][None, :]
        dh = csls[c][:, None] - csls[nz][None, :]    # CSLS suspicion, higher = worse
        dc = cdus[c][:, None] - cdus[nz][None, :]    # higher = clean more suspicious (wrong way)
        db = bcsc[c][:, None] - bcsc[nz][None, :]    # r_it: clean should be LOWER

        ok_clip = ds > 0
        ok_csls = dh < 0                              # clean must have lower suspicion
        ok_cdus = (-cdus[c])[:, None] > (-cdus[nz])[None, :]
        ok_bcsc = db < 0

        total = ds.size
        bins = []
        absm = np.abs(ds)
        for i in range(len(EDGES) - 1):
            m = (absm >= EDGES[i]) & (absm < EDGES[i + 1])
            cnt = int(m.sum())
            if cnt == 0:
                continue
            bins.append({
                "bin": NAMES[i], "range": [EDGES[i], EDGES[i + 1] if np.isfinite(EDGES[i + 1]) else None],
                "pairs": cnt, "share": cnt / total,
                "mca_csls": float(ok_csls[m].mean()),
                "mca_clip": float(ok_clip[m].mean()),
                "rescue": int((~ok_clip[m] & ok_csls[m]).sum()),
                "hurt": int((ok_clip[m] & ~ok_csls[m]).sum()),
                "auc_contrib_clip": float(ok_clip[m].mean() * cnt / total),
                "auc_contrib_csls": float(ok_csls[m].mean() * cnt / total),
            })

        both = ok_clip & ok_csls
        clip_only = ok_clip & ~ok_csls
        csls_only = ~ok_clip & ok_csls
        neither = ~ok_clip & ~ok_csls
        any_ok = ok_clip | ok_csls | ok_cdus | ok_bcsc
        out[bench] = {
            "n_clean": int(len(c)), "n_noisy": int(len(nz)), "n_pairs": int(total),
            "auc": {"clip": float(ok_clip.mean()), "csls": float(ok_csls.mean()),
                    "cdus": float(ok_cdus.mean()), "bcsc": float(ok_bcsc.mean())},
            "bins": bins,
            "table_2x2": {"both": int(both.sum()), "clip_only": int(clip_only.sum()),
                          "csls_only": int(csls_only.sum()), "neither": int(neither.sum())},
            "joint_oracle": float(any_ok.mean()),
            "net_csls": int(csls_only.sum() - clip_only.sum()),
        }
        print(bench, json.dumps({k: v for k, v in out[bench].items() if k != "bins"},
                                ensure_ascii=False))
        for bb in bins:
            print("  ", bb["bin"], f"{bb['share']*100:.2f}%", "mca_csls",
                  round(bb["mca_csls"], 3), "R/H", bb["rescue"], bb["hurt"],
                  "dAUC", round(bb["auc_contrib_csls"] - bb["auc_contrib_clip"], 5))

    write_json(RESULTS_DIR / "gate" / "csls_audit.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
