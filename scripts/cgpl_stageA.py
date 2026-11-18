"""CGPL Stage A: can a global potential explain the pairwise teacher margins?

For one benchmark:
  sample N clean + N noisy rows from the official train split
  s   = cos(image, caption)
  h   = rx + rc  (bank-side CSLS evidence, k=10)
  z   = s - 0.5 h                     (cleanliness score of CSLS)
  m_C = s_i - s_j,  m_R = z_i - z_j   for clean i, noisy j
  t   = max(0, m_C/sigma_C, m_R/sigma_R)
  a*  = argmin sum_ij (a_i - a_j - t_ij)^2   (graph Laplacian least squares)

  rho_global = 1 - SSE / SST
  stability  = Spearman(a* on half of the edges, a* on the other half)

No MLP, no labels at inference; training labels are used to form the graph.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import lsqr
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402

K = 10


def topk_mean(q, bank, k=K, chunk=256):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bt = torch.as_tensor(bank, device=dev)
    out = np.empty(len(q))
    for i in range(0, len(q), chunk):
        sim = torch.as_tensor(q[i:i + chunk], device=dev) @ bt.T
        out[i:i + chunk] = torch.topk(sim, k, dim=1).values.mean(1).cpu().numpy()
    return out


def solve_potential(n, ii, jj, tt, w=None):
    """min sum_e (a_i - a_j - t_e)^2 with sum(a)=0."""
    m = len(ii)
    rows = np.concatenate([np.arange(m), np.arange(m)])
    cols = np.concatenate([ii, jj])
    vals = np.concatenate([np.ones(m), -np.ones(m)])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(m, n))
    if w is not None:
        A = sp.diags(np.sqrt(w)) @ A
        tt = tt * np.sqrt(w)
    # add the sum-to-zero constraint
    A = sp.vstack([A, sp.csr_matrix(np.ones((1, n))) * 10.0]).tocsr()
    rhs = np.concatenate([tt, [0.0]])
    a = lsqr(A, rhs, atol=1e-8, btol=1e-8, iter_lim=3000)[0]
    pred = A[:m] @ a
    sse = float(np.sum((pred - (tt if w is None else tt)) ** 2))
    return a, sse, float(np.sum(tt ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--per-class", type=int, default=500)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    df = load_split(args.bench, "train")
    clean = df[df["label"] == 0].sample(args.per_class, random_state=0)
    noisy = df[df["label"] == 1].sample(args.per_class, random_state=0)
    sub = __import__("pandas").concat([clean, noisy]).reset_index(drop=True)
    y = np.r_[np.zeros(len(clean)), np.ones(len(noisy))]

    vi = enc.encode_images(sub["path"].tolist(), batch_size=64, num_workers=4)
    vt = enc.encode_texts(sub["caption"].astype(str).tolist())
    s = (vi * vt).sum(1)
    with np.load(RESULTS_DIR / "gate" / f"bank_{args.bench}_all.npz") as z:
        bi, bt = z["img"], z["txt"]
    rx = topk_mean(vi, bt)
    rc = topk_mean(vt, bi)
    h = rx + rc
    zc = s - 0.5 * h                                  # cleanliness

    c_idx = np.where(y == 0)[0]
    n_idx = np.where(y == 1)[0]
    ii, jj = [], []
    for i in c_idx:
        for j in n_idx:
            ii.append(i)
            jj.append(j)
    ii, jj = np.asarray(ii), np.asarray(jj)
    mC = s[ii] - s[jj]
    mR = zc[ii] - zc[jj]
    sC, sR = mC.std(), mR.std()
    t = np.maximum(0.0, np.maximum(mC / sC, mR / sR))

    n = len(sub)
    a, sse, sst = solve_potential(n, ii, jj, t)
    rho = 1 - sse / sst

    # control: the ORACLE teacher (any expert correct) is a union of two global
    # predicates, not a scalar; this is the quantity that matters for CGPL.
    t_or = ((mC > 0) | (mR > 0)).astype(float)
    _, sse_o, sst_o = solve_potential(n, ii, jj, t_or)
    rho_or = 1 - sse_o / sst_o

    # is the potential itself a function of (s, h)?  if yes, distilling it into
    # F_theta(s, rI, rc, h) cannot beat a logistic fusion of the same inputs.
    X = np.column_stack([np.ones(n), s, h, s ** 2, h ** 2, s * h])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    pred = X @ coef
    r2_pot = 1 - float(((a - pred) ** 2).sum() / ((a - a.mean()) ** 2).sum())

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ii))
    h1, h2 = perm[: len(ii) // 2], perm[len(ii) // 2:]
    a1, _, _ = solve_potential(n, ii[h1], jj[h1], t[h1])
    a2, _, _ = solve_potential(n, ii[h2], jj[h2], t[h2])
    stab = float(spearmanr(a1, a2).statistic)

    out = {"bench": args.bench, "n_clean": int(len(clean)), "n_noisy": int(len(noisy)),
           "n_edges": int(len(ii)), "rho_global": float(rho),
           "cycle_residual": float(1 - rho), "stability_spearman": stab,
           "s_perfect_acc": float(np.mean(mC > 0)), "z_perfect_acc": float(np.mean(mR > 0)),
           "teacher_pos_frac": float(np.mean(t > 0)),
           "rho_global_oracle_teacher": float(rho_or),
           "oracle_teacher_mean": float(t_or.mean()),
           "potential_r2_on_s_h": r2_pot}
    write_json(RESULTS_DIR / "gate" / f"cgpl_{args.bench}.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
