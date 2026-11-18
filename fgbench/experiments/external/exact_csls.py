"""CSLS exactly as published: no free weight, k = 10.

CSLS(x, y) = 2 cos(x, y) - r_T(x) - r_S(y)

with r_T(x) the mean similarity of x to its k nearest texts and r_S(y) the mean
similarity of y to its k nearest images, k = 10 as in Lample et al. (ICLR 2018).
There is no lambda in the paper, so nothing is selected here: every number below
uses the published weights verbatim.

Our earlier tables used cos - lam * (r_T + r_S) with lam chosen on validation,
which is a re-weighted variant. lam = 0.5 recovers the published ranking exactly
(the two expressions differ by a factor of 2), so this file is the faithful one.

Reports FGBench (5 facets) and the external protocols, next to the CRD numbers.

Run:  python fgbench/experiments/external/exact_csls.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

C = ROOT / "results" / "fgbench"
EXT = C / "external"
K = 10                      # fixed by the paper


def auc(y, s):
    return float(roc_auc_score(y, -s))


def ap(y, s):
    return float(average_precision_score(y, -s))


def topk_mean(a, b, k=K, chunk=512):
    k = min(k, len(b))
    return np.concatenate([
        np.sort(a[s:s + chunk] @ b.T, axis=1)[:, -k:].mean(1)
        for s in range(0, len(a), chunk)])


def csls(U, V, BI, BT):
    """The published formula, verbatim."""
    return 2.0 * (U * V).sum(1) - topk_mean(U, BT) - topk_mean(V, BI)


def fgbench_rows():
    rows = []
    for ds, facet in (("COCO", "Entity"), ("COCO", "Quantity"),
                      ("VG", "Attribute"), ("VG", "Relation"),
                      ("VG", "Action")):
        bank = np.load(C / f"bank_b16_{ds}.npz", allow_pickle=True)
        BI, BT = bank["img"].astype(np.float32), bank["txt"].astype(np.float32)
        if facet == "Action":
            z = np.load(C / "crd_b16_VG_Action_test.npz", allow_pickle=True)
            meta = pd.read_parquet(C / "crd_b16_VG_Action_test.meta.parquet")
        else:
            z = np.load(C / f"crdbb_clip_{ds}_{facet}_test.npz", allow_pickle=True)
            meta = pd.read_parquet(
                C / f"crdbb_clip_{ds}_{facet}_test.meta.parquet")
        U, V = z["U"].astype(np.float32), z["V"].astype(np.float32)
        y = meta.pair_label.to_numpy()
        s = csls(U, V, BI, BT)
        rows.append({"protocol": f"FGBench {facet}", "n": len(y),
                     "csls_exact": round(auc(y, s), 4),
                     "csls_auprc": round(ap(y, s), 4)})
        print(f"  FGBench {facet:9s} n={len(y):6d} CSLS(exact) {rows[-1]['csls_exact']:.4f}",
              flush=True)
    return rows


def external_rows():
    rows = []
    # ---------- Flickr30k (bank = its own train split)
    D = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\derived\flickr30k")
    Uf = pickle.load(open(EXT / "feat_flickr_images.pkl", "rb"))
    for noise in ("noun", "random"):
        df = pd.read_csv(D / f"flickr30k__{noise}__lvl0.40__seed0.csv",
                         encoding="utf-8-sig")
        p = EXT / f"feat_flickr_txt_{noise}0.40.pkl"
        if not p.is_file():
            p = EXT / f"feat_flickr_{noise}0.40.pkl"
        obj = pickle.load(open(p, "rb"))
        V = obj[1] if isinstance(obj, tuple) else obj
        y = df.is_mislabel.to_numpy()
        m_tr, m_te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
        s = csls(Uf[m_te], V[m_te], Uf[m_tr], V[m_tr])
        rows.append({"protocol": f"Flickr30k {noise}@0.4", "n": int(m_te.sum()),
                     "csls_exact": round(auc(y[m_te], s), 4),
                     "csls_auprc": round(ap(y[m_te], s), 4)})
        print(f"  Flickr {noise:6s} CSLS(exact) {rows[-1]['csls_exact']:.4f}",
              flush=True)

    # ---------- MM-IMDb (bank = its own train split)
    df = pd.read_csv(Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\derived"
                          r"\mmimdb\mmimdb__cat__lvl0.40__seed0.csv"),
                     encoding="utf-8-sig")
    U, V = pickle.load(open(EXT / "feat_mmimdb.pkl", "rb"))
    y = df.is_mislabel.to_numpy()
    m_tr, m_te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    s = csls(U[m_te], V[m_te], U[m_tr], V[m_tr])
    rows.append({"protocol": "MM-IMDb cat@0.4", "n": int(m_te.sum()),
                 "csls_exact": round(auc(y[m_te], s), 4),
                 "csls_auprc": round(ap(y[m_te], s), 4)})
    print(f"  MM-IMDb CSLS(exact) {rows[-1]['csls_exact']:.4f}", flush=True)

    # ---------- SugarCrepe (no train split: bank = FGBench COCO Entity train)
    sc = pd.read_csv(Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\derived"
                          r"\sugarcrepe\sugarcrepe__comp__lvl0.50__seed0.csv"),
                     encoding="utf-8-sig")
    Us, Vs = pickle.load(open(EXT / "feat_sugarcrepe.pkl", "rb"))
    ztr = np.load(sorted(C.glob("crdbb_clip_COCO_Entity_c*_train.npz"))[0],
                  allow_pickle=True)
    s = csls(Us, Vs, ztr["U"].astype(np.float32), ztr["V"].astype(np.float32))
    y = sc.is_mislabel.to_numpy()
    rows.append({"protocol": "SugarCrepe (zero-shot)", "n": len(y),
                 "csls_exact": round(auc(y, s), 4), "csls_auprc": round(ap(y, s), 4)})
    print(f"  SugarCrepe CSLS(exact) {rows[-1]['csls_exact']:.4f}", flush=True)
    return rows


def main() -> int:
    print("CSLS as published: 2*cos - r_T - r_S, k = 10, no free weight\n")
    rows = fgbench_rows() + external_rows()
    d = pd.DataFrame(rows)

    # CRD numbers already established, for side-by-side reference
    ref = {"FGBench Entity": 0.9995, "FGBench Quantity": 0.8837,
           "FGBench Attribute": 0.7187, "FGBench Action": 0.6145,
           "FGBench Relation": 0.5530, "Flickr30k noun@0.4": 0.9577,
           "Flickr30k random@0.4": 0.9963, "MM-IMDb cat@0.4": 0.9121,
           "SugarCrepe (zero-shot)": 0.8201}
    cos_ref = {"FGBench Entity": 0.9991, "FGBench Quantity": 0.8305,
               "FGBench Attribute": 0.6806, "FGBench Action": 0.5913,
               "FGBench Relation": 0.5413, "Flickr30k noun@0.4": 0.9497,
               "Flickr30k random@0.4": 0.9949, "MM-IMDb cat@0.4": 0.8805,
               "SugarCrepe (zero-shot)": 0.7691}
    d["cosine"] = d.protocol.map(cos_ref)
    d["CRD"] = d.protocol.map(ref)
    d["CSLS-CRD_pp"] = ((d.csls_exact - d.CRD) * 100).round(2)
    d.to_csv(EXT / "exact_csls.csv", index=False, encoding="utf-8-sig")
    print()
    print(d.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
