"""Diagnostic: decompose each 2x2 cycle score matrix.

    S = [[s_AA, s_AB],
         [s_BA, s_BB]]

    cycle mean   m    = mean(S)
    row (image)  r_i  = mean_j S_ij - m
    col (text)   c_j  = mean_i S_ij - m
    interaction  R    = S - m - r - c      (R_11 = R_22 = -R_12 = -R_21 = Delta/4)

Answers: is the AUROC-vs-2AFC gap caused by between-cycle score offsets?
No training; uses cached CLIP embeddings and the public test split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH, jdump  # noqa: E402

OUT = ROOT / "results" / "fgbench"
VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b16")
    ap.add_argument("--model-dir", default="")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR

    enc = JointEncoder(args.model_dir or CLIP_DIR)   # must match the cached images

    @torch.no_grad()
    def enc_t(caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            tk = enc.processor(text=list(caps[i:i + bs]), return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=77).to(enc.device)
            to = enc.model.text_model(**tk)
            out.append(F.normalize(enc.model.text_projection(to.pooler_output).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    te = pd.read_parquet(BENCH / "test.parquet")
    rows = []
    detail = {}

    for ds in ("COCO", "VG"):
        d = te[te.dataset == ds].copy()
        z = np.load(OUT / f"emb_{args.tag}_{ds}_test.npz", allow_pickle=True)
        emb = {k: v for k, v in zip(z["ids"].astype(str), z["vec"])}
        d["k"] = (d.target_image_id.str.split(":").str[1].astype(str)
                  if ds == "VG" else d.target_image_id.astype(str))
        d = d[d.k.isin(emb)].reset_index(drop=True)
        Vi = np.stack([emb[k] for k in d.k])
        T = enc_t(d.caption.astype(str).tolist())
        d["s"] = (Vi * T).sum(1)
        print(f"{ds}: rows {len(d)} cycles {d.cycle_id.nunique()}", flush=True)

        for facet in sorted(d.facet.unique()):
            sub = d[d.facet == facet]
            recs = []
            for cid, g in sub.groupby("cycle_id"):
                if len(g) != 4:
                    continue
                cl = g[g.pair_label == 0]
                no = g[g.pair_label == 1]
                if len(cl) != 2 or len(no) != 2:
                    continue
                # order: clean = (A,cA),(B,cB); noisy = (B,cA),(A,cB)
                ia = cl.k.tolist()
                sAA = cl.s.iloc[0]; sBB = cl.s.iloc[1]
                sBA = no.s.iloc[0]; sAB = no.s.iloc[1]
                S = np.array([[sAA, sAB], [sBA, sBB]])
                m = S.mean()
                r = S.mean(1) - m
                c = S.mean(0) - m
                R = S - m - r[:, None] - c[None, :]
                delta = (sAA + sBB) - (sAB + sBA)          # 2AFC margin
                recs.append({"cycle_id": cid, "mean": m, "row_eff": r[0] - r[1],
                             "col_eff": c[0] - c[1], "delta": delta,
                             "clean": (sAA + sBB) / 2, "noisy": (sAB + sBA) / 2})
            cy = pd.DataFrame(recs)
            if len(cy) < 30:
                continue
            # ---- variance decomposition within cycles ----
            tot_var = np.var(np.concatenate([cy["clean"], cy["noisy"]]))
            var_mean = float(np.var(cy["mean"]))
            var_delta = float(np.var(cy["delta"]))
            # ---- AUROC under different removals (cycle mean is an ORACLE device) ----
            from sklearn.metrics import roc_auc_score
            y = np.r_[np.ones(len(cy)), np.zeros(len(cy))]
            raw = np.r_[cy["clean"], cy["noisy"]]
            deb = np.r_[cy["clean"] - cy["mean"], cy["noisy"] - cy["mean"]]
            auc_raw = roc_auc_score(y, raw)
            auc_deb = roc_auc_score(y, deb)
            # image-effect-only removal: subtract row effect of the target image is
            # unavailable without partner, so report the col(text)-free variant instead
            auc2afc = float((cy["delta"] > 0).mean())
            # error concentration: rank cycles by mean, bin into quintiles
            q = pd.qcut(cy["mean"], 5, labels=False, duplicates="drop")
            err = (cy["delta"] <= 0).astype(int)
            err_by_bin = err.groupby(q).mean().round(3).tolist()
            rows.append({
                "dataset": ds, "facet": facet, "cycles": int(len(cy)),
                "auc_raw": round(float(auc_raw), 4),
                "auc_debiased_oracle": round(float(auc_deb), 4),
                "auc_2afc": round(auc2afc, 4),
                "var_total": round(float(tot_var), 5),
                "var_cycle_mean": round(var_mean, 5),
                "share_cycle_mean": round(var_mean / float(tot_var), 3),
                "share_interaction": round(var_delta / 4 / float(tot_var), 3),
                "sd_cycle_mean": round(float(cy["mean"].std()), 4),
                "sd_delta": round(float(cy["delta"].std()), 4),
                "err_rate_by_mean_quintile": err_by_bin,
            })
            detail[f"{ds}|{facet}"] = {
                "corr_mean_delta": round(float(np.corrcoef(cy["mean"], cy["delta"])[0, 1]), 3),
                "corr_mean_err": round(float(np.corrcoef(cy["mean"], err)[0, 1]), 3)}
            print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    p = OUT / f"cycle_decomp_{args.tag}.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    jdump({"rows": rows, "detail": detail}, p.with_suffix(".json"))
    print(out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
