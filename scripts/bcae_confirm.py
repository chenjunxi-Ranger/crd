"""BCAE confirmatory audit (MM-IMDb).

1. donor-source presence: is the noisy sample's donor poster inside the cohort?
2. Gibbs degeneracy: H(M), N_eff = exp(H), MAP probability, correlation with Hungarian
3. multi-cohort stability: 5 cohorts -> mean p and variance p
4. second backbone (FG-CLIP) on the identical rows / cohort manifest
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, load_split, write_json   # noqa: E402
from src.scorer import build_scorer                            # noqa: E402

B = 8
TAUS = (1.0, 3.0, 10.0)
COHORTS = 5


def cv_auc(X, y, seed=0):
    w = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    p = cross_val_predict(w, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip", choices=["clip", "fg"])
    ap.add_argument("--limit", type=int, default=800)
    args = ap.parse_args()

    df = load_split("mmimdb", "val", args.limit).drop_duplicates("image_id").reset_index(drop=True)
    y = df["label"].to_numpy()

    # donor_image_id lives in the derived CSV, not in load_split
    from src.protocol import PROJECT
    raw = pd.read_csv(PROJECT / "数据集" / "derived" / "mmimdb" /
                      "mmimdb__cat__lvl0.40__seed0.csv", encoding="utf-8-sig")
    raw["_iid"] = raw["image_id"].astype("int64").astype(str)
    dmap = dict(zip(raw["_iid"], raw["donor_image_id"].fillna(-1).astype("int64").astype(str)))
    donor = df["image_id"].astype(str).map(dmap).fillna("").to_numpy()
    has_donor = bool((donor != "").any())

    sc = build_scorer(args.backbone)

    # --- reference-bank CSLS (the strongest existing baseline)
    if args.backbone == "clip":
        from rat_gate import topk_mean
        from src.encoders import JointEncoder
        from src.protocol import CLIP_DIR
        enc = JointEncoder(CLIP_DIR)
        vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
        vt = enc.encode_texts(df["caption"].astype(str).tolist())
        zb = np.load(RESULTS_DIR / "gate" / "bank_mmimdb_all.npz")
        h = topk_mean(vi, zb["txt"]) + topk_mean(vt, zb["img"])
    else:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, CLIPImageProcessor, CLIPTokenizerFast
        from PIL import Image
        p = r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\fg-clip-base"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True,
                                                 local_files_only=True).to(dev).eval()
        ip = CLIPImageProcessor.from_pretrained(p, local_files_only=True)
        tk = CLIPTokenizerFast.from_pretrained(p, local_files_only=True)

        @torch.no_grad()
        def enc_i(paths, bs=48):
            out = []
            for i in range(0, len(paths), bs):
                ims = []
                for q in paths[i:i + bs]:
                    with Image.open(q) as im:
                        im = im.convert("RGB")
                        if im.width * im.height > 20_000_000:
                            im.draft("RGB", (1024, 1024))
                        ims.append(im.copy())
                px = ip(images=ims, return_tensors="pt")["pixel_values"].to(dev)
                out.append(F.normalize(m.get_image_features(pixel_values=px).float(), dim=-1).cpu().numpy())
            return np.concatenate(out)

        @torch.no_grad()
        def enc_t(caps, bs=128):
            out = []
            for i in range(0, len(caps), bs):
                t = {k: v.to(dev) for k, v in tk(list(caps[i:i + bs]), return_tensors="pt",
                                                 padding=True, truncation=True, max_length=77).items()}
                out.append(F.normalize(m.get_text_features(**t).float(), dim=-1).cpu().numpy())
            return np.concatenate(out)

        vi = enc_i(df["path"].tolist()); vt = enc_t(df["caption"].astype(str).tolist())
        zb = np.load(RESULTS_DIR / "gate" / "fgbank_mmimdb.npz")
        sim_t = vi @ zb["txt"].T
        sim_i = vt @ zb["img"].T
        k = 10
        h = np.partition(sim_t, -k, axis=1)[:, -k:].mean(1) + \
            np.partition(sim_i, -k, axis=1)[:, -k:].mean(1)

    cos = (vi * vt).sum(1)
    s = -cos

    P = np.array(list(itertools.permutations(range(B))), dtype=np.int16)
    idxv = np.arange(B)[None, :]

    # --- five label-blind cohorts; S is computed once and reused for every tau
    mats = []
    donor_in = {}          # (cohort, sample_id) -> donor present in cohort
    for r in range(COHORTS):
        order = np.random.default_rng(100 + r).permutation(len(df))
        for s0 in range(0, len(df) - B + 1, B):
            idx = order[s0:s0 + B]
            sub = df.iloc[idx]
            imgs = sub["path"].tolist(); caps = sub["caption"].astype(str).tolist()
            if args.backbone == "fg":
                S = vi[idx] @ vt[idx].T                 # reuse cached embeddings
            else:
                S = np.zeros((B, B))
                for i in range(B):
                    S[i] = sc.score_many([imgs[i]] * B, caps)
            mats.append((idx, S))
            ids = set(sub["image_id"].astype(str))
            for k in range(B):
                donor_in[(r, int(idx[k]))] = int(donor[idx[k]] in ids) if has_donor else -1

    res = {}
    for tau in TAUS:
        p_sum = np.zeros(len(df)); p_sq = np.zeros(len(df)); cnt = np.zeros(len(df))
        marg = np.zeros(len(df)); hmap = np.zeros(len(df)); neff = np.zeros(len(df))
        maprob = np.zeros(len(df)); hits = 0
        for idx, S in mats:
            E = S[idxv, P].sum(1)
            w = np.exp(tau * (E - E.max())); w /= w.sum()
            p = np.array([w[P[:, i] == i].sum() for i in range(B)])
            H = float(-(w * np.log(w + 1e-12)).sum())
            neff[idx] = np.exp(H); maprob[idx] = float(w.max())
            for i in range(B):
                has = P[:, i] == i
                marg[idx[i]] = E[has].max() - E[~has].max()
                hmap[idx[i]] = 1.0 if np.argmax(w) in np.where(P[:, i] == i)[0] else 0.0
            p_sum[idx] += p; p_sq[idx] += p ** 2; cnt[idx] += 1
        pbar = p_sum / cnt
        pvar = p_sq / cnt - pbar ** 2
        q = -np.log(pbar + 1e-9)
        res[str(tau)] = {
            "auc_pbar_alone": float(roc_auc_score(y, q)),
            "auc_pvar_alone": float(roc_auc_score(y, pvar)),
            "M1_csls": cv_auc(np.column_stack([s, h]), y),
            "M3_csls_pbar": cv_auc(np.column_stack([s, h, q]), y),
            "M3_csls_pbar_pvar": cv_auc(np.column_stack([s, h, q, pvar]), y),
            "neff_mean": float(neff.mean()), "map_prob_mean": float(maprob.mean()),
            "p_mean": float(pbar.mean()), "p_std": float(pbar.std()),
            "corr_p_hung": float(np.corrcoef(pbar, hmap)[0, 1]),
            "corr_p_margin": float(np.corrcoef(pbar, marg)[0, 1]),
        }
        r = res[str(tau)]
        r["delta_over_M1"] = r["M3_csls_pbar"] - r["M1_csls"]

    # --- donor-source presence split (tau=10)
    tau = "10.0"
    pbar = None
    key = [k for k in res if k.startswith("10")][0]
    # recompute pbar for the split
    p_sum = np.zeros(len(df)); cnt = np.zeros(len(df))
    for idx, S in mats:
        E = S[idxv, P].sum(1)
        w = np.exp(10.0 * (E - E.max())); w /= w.sum()
        p = np.array([w[P[:, i] == i].sum() for i in range(B)])
        p_sum[idx] += p; cnt[idx] += 1
    pbar = p_sum / cnt
    q = -np.log(pbar + 1e-9)
    if has_donor:
        pres = np.zeros(len(df)); cntd = np.zeros(len(df))
        for (r_, sid), v in donor_in.items():
            pres[sid] += v; cntd[sid] += 1
        present = (pres / np.maximum(cntd, 1)) > 0.5
        noisy = y == 1
        res["donor"] = {
            "n_noisy": int(noisy.sum()),
            "frac_source_present": float(present[noisy].mean()) if noisy.any() else None,
            "auc_p_present": float(roc_auc_score(noisy[present], q[present]))
            if (present & noisy).sum() > 20 and (~present & noisy).sum() > 20 else None,
            "auc_p_absent": float(roc_auc_score(noisy[~present], q[~present]))
            if (~present & noisy).sum() > 20 and (present & noisy).sum() > 20 else None,
            "mean_p_present_noisy": float(pbar[present & noisy].mean()) if (present & noisy).any() else None,
            "mean_p_absent_noisy": float(pbar[~present & noisy].mean()) if (~present & noisy).any() else None,
            "mean_p_clean": float(pbar[y == 0].mean()),
        }
    print(args.backbone, res, flush=True)
    write_json(RESULTS_DIR / "gate" / f"bcae_confirm_{args.backbone}.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
