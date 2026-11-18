"""CMPI admission gate: does the 2x2 cross-modal interaction carry information
beyond the two marginal perturbation responses?

    s00 = s(I,  c )      s01 = s(I,  c')
    s10 = s(I', c )      s11 = s(I', c')
    kappa = s11 - s10 - s01 + s00

Text transform : fixed synonym table, else "A photo of <c>" (recorded per row).
Image transform: mild brightness/contrast/colour jitter (no crop, no flip).

M0 = [s00]   M1 = [s00, dI, dT]   M2 = M1 + [kappa]
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, load_foil_val, load_split, write_json  # noqa: E402
from src.scorer import build_scorer                                    # noqa: E402

SYN = {
    "man": "guy", "woman": "lady", "boy": "kid", "girl": "young girl",
    "big": "large", "small": "little", "large": "big", "little": "small",
    "riding": "on", "sitting": "seated", "standing": "stood",
    "red": "crimson", "blue": "navy", "green": "emerald", "black": "dark",
    "white": "pale", "yellow": "golden", "brown": "tan", "grey": "gray",
    "dog": "puppy", "cat": "kitten", "car": "vehicle", "bike": "bicycle",
    "street": "road", "field": "meadow", "water": "sea", "table": "desk",
    "plate": "dish", "ball": "sphere", "tree": "tree", "building": "structure",
}


def paraphrase(cap: str) -> tuple[str, str]:
    toks = re.findall(r"[A-Za-z']+|[^A-Za-z']+", cap)
    hit = 0
    for i, w in enumerate(toks):
        low = w.lower()
        if low in SYN:
            toks[i] = SYN[low]
            hit += 1
    if hit:
        return "".join(toks), "synonym"
    return "A photo of " + cap.strip(), "prefix"


def jitter(path: str, out_dir: Path, b: float = 1.06, c: float = 1.05,
           s: float = 1.05) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width * im.height > 20_000_000:
            im.draft("RGB", (1024, 1024))
        im = ImageEnhance.Brightness(im).enhance(b)
        im = ImageEnhance.Contrast(im).enhance(c)
        im = ImageEnhance.Color(im).enhance(s)
    out = out_dir / (Path(path).stem + "__jit.jpg")
    im.save(out, quality=92)
    return str(out)


def cv_auc(X, y, seed=0):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def mca(s, q, y, eps):
    """P(q_noisy > q_clean | |ds| < eps), nearest-clean matching."""
    cl, nz = np.where(y == 0)[0], np.where(y == 1)[0]
    order = np.argsort(s[cl]); cs = s[cl][order]
    pos = np.searchsorted(cs, s[nz])
    l = np.clip(pos - 1, 0, len(cs) - 1); r = np.clip(pos, 0, len(cs) - 1)
    ch = np.where(np.abs(cs[l] - s[nz]) <= np.abs(cs[r] - s[nz]), l, r)
    part = order[ch]
    d = np.abs(s[cl][part] - s[nz])
    k = d < eps
    if k.sum() < 30:
        return None, int(k.sum())
    return float(np.mean(q[nz][k] > q[cl][part][k])), int(k.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb", "foil"])
    ap.add_argument("--backbone", required=True, choices=["clip", "fg"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=0.01)
    args = ap.parse_args()

    sc = build_scorer(args.backbone)
    tmp = Path(tempfile.mkdtemp(prefix="cmpi_"))
    out = {"bench": args.bench, "backbone": args.backbone, "limit": args.limit}

    if args.bench == "foil":
        df = load_foil_val(args.limit)
        res = {}
        for tag, col in (("pos", "positive"), ("neg", "negative")):
            caps = df[col].astype(str).tolist()
            caps2 = [paraphrase(c)[0] for c in caps]
            img2 = [jitter(p, tmp) for p in df["path"]]
            s00 = sc.score_many(df["path"].tolist(), caps)
            s01 = sc.score_many(df["path"].tolist(), caps2)
            s10 = sc.score_many(img2, caps)
            s11 = sc.score_many(img2, caps2)
            res[tag] = dict(s00=s00, s01=s01, s10=s10, s11=s11)
        kap = (res["neg"]["s11"] - res["neg"]["s10"] - res["neg"]["s01"] + res["neg"]["s00"]) - \
              (res["pos"]["s11"] - res["pos"]["s10"] - res["pos"]["s01"] + res["pos"]["s00"])
        acc = float(np.mean(res["neg"]["s00"] < res["pos"]["s00"]))
        acc_k = float(np.mean(kap < 0))     # noisy should show smaller kappa
        out.update({"n_pairs": int(len(df)), "cos_acc": acc, "acc_kappa": acc_k,
                    "mean_kappa_pos": float(np.mean(res["pos"]["s11"] - res["pos"]["s10"]
                                                    - res["pos"]["s01"] + res["pos"]["s00"])),
                    "mean_kappa_neg": float(np.mean(res["neg"]["s11"] - res["neg"]["s10"]
                                                    - res["neg"]["s01"] + res["neg"]["s00"]))})
        np.savez_compressed(RESULTS_DIR / "gate" / f"cmpi_{args.backbone}_foil.npz",
                            kap=kap, s_pos=res["pos"]["s00"], s_neg=res["neg"]["s00"])
    else:
        df = load_split(args.bench, "val", args.limit)
        caps = df["caption"].astype(str).tolist()
        caps2 = [paraphrase(c)[0] for c in caps]
        kinds = pd.Series([paraphrase(c)[1] for c in caps]).value_counts().to_dict()
        img2 = [jitter(p, tmp) for p in df["path"].tolist()]
        s00 = sc.score_many(df["path"].tolist(), caps)
        s01 = sc.score_many(df["path"].tolist(), caps2)
        s10 = sc.score_many(img2, caps)
        s11 = sc.score_many(img2, caps2)
        dI, dT = s10 - s00, s01 - s00
        kappa = s11 - s10 - s01 + s00
        y = df["label"].to_numpy()
        s = -s00
        m0 = float(roc_auc_score(y, s))
        m1 = cv_auc(np.column_stack([s, dI, dT]), y)
        m2 = cv_auc(np.column_stack([s, dI, dT, kappa]), y)
        q, n = mca(s, -np.abs(kappa), y, args.eps)
        out.update({"n": int(len(y)), "paraphrase_kinds": kinds,
                    "auc_cos": m0, "auc_M1": m1, "auc_M2": m2, "delta_kappa": m2 - m1,
                    "mca_abs_kappa": q, "n_matched": n,
                    "mean_kappa_clean": float(kappa[y == 0].mean()),
                    "mean_kappa_noisy": float(kappa[y == 1].mean()),
                    "corr_kappa_dI": float(np.corrcoef(kappa, dI)[0, 1]),
                    "corr_kappa_dT": float(np.corrcoef(kappa, dT)[0, 1])})
        np.savez_compressed(RESULTS_DIR / "gate" / f"cmpi_{args.backbone}_{args.bench}.npz",
                            s00=s00, s01=s01, s10=s10, s11=s11, label=y,
                            image_id=df["image_id"].astype(str).to_numpy())
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"cmpi_{args.backbone}_{args.bench}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
