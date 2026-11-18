"""CLIP cosine vs CRD on FOIL-COCO (same-image, two-caption task).

FOIL gives, for one image, an original caption and a minimally edited "foil"
caption that differs in one word. Two metrics, both threshold-free:

  pair accuracy   fraction of pairs where s(image, original) > s(image, foil)
  row AUROC/AP    rows = every caption with label correct/wrong

FOIL ships train/test only, so 15% of the train *images* are held out for
lambda selection; the official test split is never touched.

Head training is label-free (row-level standardised cosine, as in CRD); labels
are used only to pick lambda on the held-out split. Everything else is the same
code path as FGBench.

Run:  python fgbench/experiments/foil/run_foil.py [--n-train 24000] [--n-test 30000]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fgbench" / "foil"
OUT.mkdir(parents=True, exist_ok=True)
FOIL = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\FOIL-COCO")
SEED = 0


def auc(y, x):
    return float(roc_auc_score(y, -x))


def load_pairs(n_train: int, n_test: int):
    tr = pd.read_csv(FOIL / "pairs_train.csv", encoding="utf-8-sig")
    te = pd.read_csv(FOIL / "pairs_test.csv", encoding="utf-8-sig")
    rng = np.random.default_rng(SEED)

    def cap(df, n):
        if n >= len(df):
            return df.reset_index(drop=True)
        idx = np.sort(rng.choice(len(df), n, replace=False))
        return df.iloc[idx].reset_index(drop=True)

    tr, te = cap(tr, n_train), cap(te, n_test)

    # split train by image so a caption's image never crosses the split
    ids = tr.image_id.unique().copy()
    rng.shuffle(ids)
    n_val = int(0.15 * len(ids))
    val_ids = set(ids[:n_val])
    tr["part"] = np.where(tr.image_id.isin(val_ids), "val", "train")
    te["part"] = "test"
    return tr, te


def feats(df: pd.DataFrame, tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (U_pos, V_pos, V_neg) aligned to df rows."""
    cache = OUT / f"feat_{tag}.pkl"
    if cache.is_file():
        with open(cache, "rb") as f:
            return pickle.load(f)

    from fgbench.models.crd_backbone import ClipBB
    bb = ClipBB()
    t0 = time.time()
    imgs = df.image_id.unique()
    path_of = df.drop_duplicates("image_id").set_index("image_id").path
    U = bb.images([path_of[i] for i in imgs])
    print(f"  [{tag}] {len(imgs)} images in {time.time()-t0:.0f}s", flush=True)
    pos = df.positive_caption.tolist()
    neg = df.negative_caption.tolist()
    Vp = bb.texts(pos)
    Vn = bb.texts(neg)
    print(f"  [{tag}] {len(pos)*2} captions in {time.time()-t0:.0f}s", flush=True)
    pos_of = {i: k for k, i in enumerate(imgs)}
    Uu = U[[pos_of[i] for i in df.image_id]]
    with open(cache, "wb") as f:
        pickle.dump((Uu, Vp, Vn), f)
    return Uu, Vp, Vn


def rowify(U, Vp, Vn):
    """Rows: 2N rows, positive caption first then its foil."""
    U2 = np.concatenate([U, U])
    V2 = np.concatenate([Vp, Vn])
    y = np.concatenate([np.zeros(len(U)), np.ones(len(U))])
    return U2, V2, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--n-test", type=int, default=30000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()

    df = pd.concat(load_pairs(a.n_train, a.n_test), ignore_index=True)
    print(df.groupby("part").agg(pairs=("pair_id", "size"),
                                 images=("image_id", "nunique")).to_string())

    splits = {}
    for part in ("train", "val", "test"):
        sub = df[df.part == part].reset_index(drop=True)
        U, Vp, Vn = feats(sub, part)
        splits[part] = (U, Vp, Vn,
                        np.einsum("ij,ij->i", U, Vp),
                        np.einsum("ij,ij->i", U, Vn))

    # standardisation constants from the train split only
    tr_s = np.concatenate([splits["train"][3], splits["train"][4]])
    mu, sd = float(tr_s.mean()), float(tr_s.std() + 1e-8)

    def zp(part):
        return (splits[part][3] - mu) / sd

    def zn(part):
        return (splits[part][4] - mu) / sd

    y_va = np.concatenate([np.zeros(len(zp("val"))), np.ones(len(zp("val")))])
    y_te = np.concatenate([np.zeros(len(zp("test"))), np.ones(len(zp("test")))])
    cos_va = np.concatenate([zp("val"), zn("val")])
    cos_te = np.concatenate([zp("test"), zn("test")])

    def pair_acc(sp, sn):
        return float((sp > sn).mean())

    from fgbench.models.crd_backbone import Off
    import torch

    rows = []
    for seed in (int(x) for x in a.seeds.split(",")):
        torch.manual_seed(seed)
        Utr, Vp_tr, Vn_tr = splits["train"][:3]
        U2, V2, _ = rowify(Utr, Vp_tr, Vn_tr)
        M2 = (np.einsum("ij,ij->i", U2, V2) - mu) / sd
        head = Off(d=U2.shape[1]).to("cuda" if torch.cuda.is_available() else "cpu")
        dev = next(head.parameters()).device
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        Ut = torch.tensor(U2, device=dev); Vt = torch.tensor(V2, device=dev)
        Mt = torch.tensor(M2, device=dev, dtype=torch.float32)
        rng = np.random.default_rng(seed)
        for _ in range(a.steps):
            idx = torch.tensor(rng.integers(0, len(Ut), 256), device=dev)
            loss = torch.nn.functional.huber_loss(head(Ut[idx], Vt[idx]), Mt[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            def pred(part):
                U, Vp_, Vn_ = splits[part][:3]
                return (head(torch.tensor(np.concatenate([U, U]), device=dev),
                             torch.tensor(np.concatenate([Vp_, Vn_]), device=dev))
                        .cpu().numpy().ravel())
            mh_va, mh_te = pred("val"), pred("test")

        lam = max(np.round(np.arange(0, 2.01, 0.1), 1),
                  key=lambda L: auc(y_va, cos_va - L * mh_va))
        crd_va, crd_te = cos_va - lam * mh_va, cos_te - lam * mh_te

        n = len(zp("test"))
        rows.append({
            "seed": seed, "lam": float(lam),
            "acc_cos": round(pair_acc(zp("test"), zn("test")), 4),
            "acc_crd": round(pair_acc(crd_te[:n], crd_te[n:]), 4),
            "auroc_cos": round(auc(y_te, cos_te), 4),
            "auroc_crd": round(auc(y_te, crd_te), 4),
            "auprc_cos": round(float(average_precision_score(y_te, -cos_te)), 4),
            "auprc_crd": round(float(average_precision_score(y_te, -crd_te)), 4),
        })
        rows[-1]["d_acc_pp"] = round((rows[-1]["acc_crd"] - rows[-1]["acc_cos"]) * 100, 2)
        rows[-1]["d_auroc_pp"] = round((rows[-1]["auroc_crd"] - rows[-1]["auroc_cos"]) * 100, 2)
        print(rows[-1], flush=True)
        np.savez_compressed(OUT / f"test_scores_s{seed}.npz",
                            image_id=splits["test"][0].shape and
                            df[df.part == "test"].image_id.values,
                            y=y_te, cos=cos_te, crd=crd_te, lam=np.array([lam]))

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "result.csv", index=False, encoding="utf-8-sig")
    print()
    print(r.to_string(index=False))
    print(f"\nmean  acc {r.acc_cos.mean():.4f}->{r.acc_crd.mean():.4f} "
          f"({r.d_acc_pp.mean():+.2f}pp)   auroc {r.auroc_cos.mean():.4f}->"
          f"{r.auroc_crd.mean():.4f} ({r.d_auroc_pp.mean():+.2f}pp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
