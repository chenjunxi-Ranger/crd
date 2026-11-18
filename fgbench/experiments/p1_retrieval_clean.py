"""Pilot: cleaning arms -> linear image->text map -> R@1 on held-out COCO 5k."""
import numpy as np
import pandas as pd

ROWS = "results/fgbench/lemon_protocol/rows_frac0.40.parquet"
FEAT = "results/fgbench/lemon_protocol/features.npz"

rows = pd.read_parquet(ROWS)
z = np.load(FEAT, allow_pickle=True)
img = dict(zip(z["img_ids"].astype(str), z["img_vec"]))
txt = dict(zip(z["txt_keys"], z["txt_vec"]))
rows = rows[rows.caption.isin(txt)].copy()
rows["image_id"] = rows.image_id.astype(str)
rows = rows[rows.image_id.isin(img)].copy()
rows["noisy"] = rows.label.astype(bool)

ids = np.array(sorted(rows.image_id.unique()))
rng = np.random.default_rng(0)
rng.shuffle(ids)
tr_ids, ev_ids = set(ids[:4000]), set(ids[4000:])
tr = rows[rows.image_id.isin(tr_ids)].reset_index(drop=True)
ev = rows[rows.image_id.isin(ev_ids)].reset_index(drop=True)


def mat(d):
    U = np.stack([img[i] for i in d.image_id]).astype("float32")
    V = np.stack([txt[c] for c in d.caption]).astype("float32")
    U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-9
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return U, V


Utr, Vtr = mat(tr)
Uev, Vev = mat(ev)
y = tr.noisy.to_numpy()
yev = ev.noisy.to_numpy()
ev_owner = ev.image_id.to_numpy()
ev_uid = np.array(sorted(set(ev_owner)))


def r1(Uq, Vc, owner, uid):
    hit = 0
    for i in range(0, len(Uq), 512):
        s = Uq[i:i + 512] @ Vc.T
        top = s.argmax(1)
        hit += int((owner[top] == uid[i:i + 512]).sum())
    return hit / len(Uq)


def fit_map(U, V, lam=10.0):
    W = np.linalg.solve(U.T @ U + lam * np.eye(U.shape[1]), U.T @ V)
    return W


def crd_score(U, V, noisy, seed=0):
    zc = (U * V).sum(1)
    zs = (zc - zc.mean()) / (zc.std() + 1e-8)
    rng = np.random.default_rng(seed)
    hold = rng.choice(len(zs), min(4000, len(zs) // 5), replace=False)
    m = np.ones(len(zs), bool); m[hold] = False
    X = np.concatenate([U, V, np.ones((len(U), 1), np.float32)], 1)
    A = X[m].T @ X[m] + 1e-2 * np.eye(X.shape[1])
    w = np.linalg.solve(A, X[m].T @ zs[m])
    g = X @ w
    best = (0.0, -1.0)
    for lam in np.arange(0, 2.01, 0.25):
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(noisy[hold], -(zs[hold] - lam * g[hold]))
        if auc > best[1]:
            best = (lam, auc)
    return zs - best[0] * g


# build eval image/caption matrices: one row per unique image, all eval captions
uev = {i: img[i] for i in ev_uid}
Uq = np.stack([uev[i] for i in ev_uid]).astype("float32")
Uq /= np.linalg.norm(Uq, axis=1, keepdims=True) + 1e-9
Vc = Vev
caps_owner = ev_owner

for seed in range(3):
    rng = np.random.default_rng(seed)
    scores = {
        "Cosine": (Utr * Vtr).sum(1),
        "CRD": crd_score(Utr, Vtr, y, seed),
        "None": rng.random(len(Utr)),
        "Oracle": (~y).astype(float),
    }
    for arm, sc in scores.items():
        keep = np.argsort(-sc)[: int(0.5 * len(sc))]
        W = fit_map(Utr[keep], Vtr[keep])
        print(seed, arm, "R@1", round(r1(Uq @ W, Vc, caps_owner, ev_uid), 4),
              "noise_keep", round(float(y[keep].mean()), 4))
