"""Which head drives the FOIL dissociation?

Within a pair the image is fixed, so a_phi(u) is identical for both captions and
cancels; only b_psi(v) can change the pair ordering. Across images, a_phi(u)
matters. So the prediction is:

    correcting with a only  -> pair accuracy unchanged, row AUROC moves
    correcting with b only  -> pair accuracy moves (down), row AUROC moves

Trains three heads on the cached FOIL features (image-only, text-only, both) and
reports pair accuracy and row AUROC for each correction, one seed.

Run:  python fgbench/experiments/foil/isolate_heads.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fgbench" / "foil"


def auc(y, x):
    return float(roc_auc_score(y, -x))


def main() -> int:
    splits = {}
    for part in ("train", "val", "test"):
        with open(OUT / f"feat_{part}.pkl", "rb") as f:
            splits[part] = pickle.load(f)

    tr_s = np.concatenate([np.einsum("ij,ij->i", splits["train"][0], splits["train"][1]),
                           np.einsum("ij,ij->i", splits["train"][0], splits["train"][2])])
    mu, sd = float(tr_s.mean()), float(tr_s.std() + 1e-8)

    def rows(part):
        U, Vp, Vn = splits[part]
        U2 = np.concatenate([U, U])
        V2 = np.concatenate([Vp, Vn])
        z = (np.einsum("ij,ij->i", U2, V2) - mu) / sd
        y = np.concatenate([np.zeros(len(U)), np.ones(len(U))])
        return U2, V2, z, y

    from fgbench.models.crd_backbone import Off
    import torch

    Utr, Vtr, Ztr, _ = rows("train")
    Uva, Vva, Zva, yva = rows("val")
    Ute, Vte, Zte, yte = rows("test")
    n = len(Zte) // 2

    def pair_acc(x):
        return float((x[:n] > x[n:]).mean())

    out = []
    for parts in ("image", "text", "both"):
        torch.manual_seed(0)
        head = Off(d=Utr.shape[1], parts=parts).to(
            "cuda" if torch.cuda.is_available() else "cpu")
        dev = next(head.parameters()).device
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        Ut = torch.tensor(Utr, device=dev); Vt = torch.tensor(Vtr, device=dev)
        Mt = torch.tensor(Ztr, device=dev, dtype=torch.float32)
        rng = np.random.default_rng(0)
        for _ in range(1500):
            idx = torch.tensor(rng.integers(0, len(Ut), 256), device=dev)
            loss = torch.nn.functional.huber_loss(head(Ut[idx], Vt[idx]), Mt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            mv = head(torch.tensor(Uva, device=dev), torch.tensor(Vva, device=dev)).cpu().numpy().ravel()
            mt = head(torch.tensor(Ute, device=dev), torch.tensor(Vte, device=dev)).cpu().numpy().ravel()

        lam = max(np.round(np.arange(0, 2.01, 0.1), 1),
                  key=lambda L: auc(yva, Zva - L * mv))
        cal = Zte - lam * mt
        out.append({"head": parts, "lam": float(lam),
                    "pair_acc": round(pair_acc(cal), 4),
                    "row_auroc": round(auc(yte, cal), 4)})
        print(out[-1], flush=True)

    base = {"head": "none (cosine)", "lam": 0.0,
            "pair_acc": round(pair_acc(Zte), 4), "row_auroc": round(auc(yte, Zte), 4)}
    df = pd.DataFrame([base] + out)
    df["d_acc_pp"] = ((df.pair_acc - base["pair_acc"]) * 100).round(2)
    df["d_auroc_pp"] = ((df.row_auroc - base["row_auroc"]) * 100).round(2)
    df.to_csv(OUT / "isolate_heads.csv", index=False, encoding="utf-8-sig")
    print()
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
