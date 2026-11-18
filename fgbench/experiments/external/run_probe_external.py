"""Zero-shot external generalisation: FGBench-trained CRD vs FGBench-trained probe.

Question.  Section VI compares CRD with a supervised probe *inside* FGBench.  A
reviewer will ask which of the two transfers to a corpus it was not fitted on.
This script answers that with the same protocol for both methods:

    source   FGBench training split only (COCO Entity and VG Attribute)
    target   five external protocols; **no target label, no target mean/std and
             no target refitting** enters either scorer
    metric   row-level AUROC (independent rows) or within-pair accuracy
             (same-image two-caption tasks)

Scorers, all in anomaly space (higher = noisier):

    cosine    -s(I,T)
    CRD       -( (s - mu) / sigma - lambda * (a(u) + b(v) + b0) )
    probe     MLP on [u; v; u*v], trained on FGBench pair labels
    text      MLP on v only, same budget (control for caption-style shortcuts)

mu, sigma, the CRD head, lambda and both probes come from the FGBench source
split.  The probe is a supervised reference, not a method under the same budget.

Run:  python fgbench/experiments/external/run_probe_external.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))
sys.path.insert(0, str(ROOT / "fgbench" / "models"))
sys.path.insert(0, str(ROOT / "fgbench" / "experiments" / "p0_baselines"))

from run_baselines import load_cache  # noqa: E402
from run_probe import MLP, build  # noqa: E402
from fgbench.models.crd_backbone import Off  # noqa: E402

EXT = ROOT / "results" / "fgbench" / "external"
FG = ROOT / "results" / "fgbench"
HEADS = EXT / "probe_heads"
HEADS.mkdir(parents=True, exist_ok=True)
FOIL = ROOT / "results" / "fgbench" / "foil"
DERIVED = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\derived")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT = 1000
SEEDS = (0, 1, 2)
SOURCES = [("COCO", "Entity", 8000), ("VG", "Attribute", 8000)]


# ---------------------------------------------------------------- source heads
def train_head(kind: str, ds: str, facet: str, cycles: int, seed: int):
    """Train (or load) one MLP on the FGBench source split."""
    path = HEADS / f"probe_{kind}_{ds}_{facet}_s{seed}.pt"
    dtr, Utr, Vtr = load_cache(ds, facet, "train", cycles)
    X = build(Utr, Vtr, kind)
    if path.is_file():
        ck = torch.load(path, map_location=DEV, weights_only=False)
        model = MLP(ck["dim"]).to(DEV)
        model.load_state_dict(ck["state"])
        model.eval()
        return model

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X, device=DEV, dtype=torch.float32)
    yt = torch.tensor(dtr.pair_label.to_numpy().astype(np.float32), device=DEV)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(1500):
        idx = torch.tensor(rng.integers(0, len(Xt), 256), device=DEV)
        loss = lossf(model(Xt[idx]), yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    torch.save({"state": model.state_dict(), "dim": int(X.shape[1]),
                "kind": kind, "dataset": ds, "facet": facet, "seed": seed}, path)
    return model


@torch.no_grad()
def infer(models, U: np.ndarray, V: np.ndarray, kind: str) -> np.ndarray:
    X = torch.tensor(build(U, V, kind).astype(np.float32), device=DEV)
    return np.mean([m(X).cpu().numpy() for m in models], axis=0)


def crd_scorer(ds: str, facet: str):
    """Source-standardised CRD score, anomaly space, no target statistics."""
    prov = json.loads((FG / f"crdbb_prov_clip_{ds}_{facet}_both_row_s0.json")
                      .read_text(encoding="utf-8"))
    mu, sd = float(prov["mu"]), float(prov["sd"])
    ck = torch.load(FG / f"crdbb_clip_{ds}_{facet}_both_row_s0_head.pt",
                    map_location=DEV, weights_only=False)
    head = Off(d=int(ck["dim"]), parts="both").to(DEV)
    head.load_state_dict(ck["state"])
    head.eval()
    lam = float(np.load(FG / f"crdbb_scores_clip_{ds}_{facet}_both_row_s0.npz",
                        allow_pickle=True)["lam"][0])

    @torch.no_grad()
    def score(U, V):
        S = np.einsum("ij,ij->i", U, V)
        q = head(torch.tensor(U.astype(np.float32), device=DEV),
                 torch.tensor(V.astype(np.float32), device=DEV)).cpu().numpy().ravel()
        return -(S - mu) / sd + lam * q
    return score


# ---------------------------------------------------------------- metrics
def pair_acc(pair, y, a) -> float:
    df = pd.DataFrame({"p": pair, "y": y, "a": a})
    pos = df[df.y == 0].groupby("p").a.first()
    neg = df[df.y == 1].groupby("p").a.first()
    j = pos.index.intersection(neg.index)
    return float((pos[j] < neg[j]).mean())


def boot_row(y, a_ref, a_alt, rng, n_boot=N_BOOT):
    d, n = [], len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy = y[i]
        if yy.min() == yy.max():
            continue
        d.append(roc_auc_score(yy, a_alt[i]) - roc_auc_score(yy, a_ref[i]))
    return np.asarray(d)


def boot_pair(pair, y, a_ref, a_alt, rng, n_boot=N_BOOT):
    def sides(a):
        df = pd.DataFrame({"p": pair, "y": y, "a": a})
        return df[df.y == 0].groupby("p").a.first(), df[df.y == 1].groupby("p").a.first()
    p0, n0 = sides(a_ref)
    p1, n1 = sides(a_alt)
    ids = np.asarray(p0.index.intersection(n0.index))
    d = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        d.append((p1[pick] < n1[pick]).mean() - (p0[pick] < n0[pick]).mean())
    return np.asarray(d)


def summarize(name, protocol, metric, y, a_pos, pair, rng) -> dict:
    """a_pos maps scorer name -> anomaly score.

    Comparisons of interest: CRD vs cosine, probe vs CRD, text-probe vs CRD.
    """
    rec = {"target": name, "protocol": protocol, "metric": metric, "n": int(len(y))}
    if metric == "auroc":
        for k, a in a_pos.items():
            rec[k] = round(float(roc_auc_score(y, a)), 4)
        boot = lambda ref, alt: boot_row(y, a_pos[ref], a_pos[alt], rng)
    else:
        for k, a in a_pos.items():
            rec[k] = round(pair_acc(pair, y, a), 4)
        boot = lambda ref, alt: boot_pair(pair, y, a_pos[ref], a_pos[alt], rng)

    for key, ref, alt in (("crd_vs_cos", "cosine", "crd"),
                          ("probe_vs_crd", "crd", "probe"),
                          ("text_vs_crd", "crd", "text")):
        d = boot(ref, alt)
        rec[f"d_{key}_pp"] = round(float(np.mean(d)) * 100, 2)
        rec[f"ci_{key}"] = [round(float(np.percentile(d, 2.5)) * 100, 2),
                            round(float(np.percentile(d, 97.5)) * 100, 2)]
        p_two = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
        rec[f"p_{key}"] = round(p_two, 4)
    return rec


# ---------------------------------------------------------------- targets
def load_sugarcrepe():
    p = DERIVED / "sugarcrepe" / "sugarcrepe__comp__lvl0.50__seed0.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    with open(EXT / "feat_sugarcrepe.pkl", "rb") as f:
        U, V = pickle.load(f)
    assert len(df) == len(U)
    return df, U, V


def load_flickr(noise: str):
    p = DERIVED / "flickr30k" / f"flickr30k__{noise}__lvl0.40__seed0.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    with open(EXT / "feat_flickr_images.pkl", "rb") as f:
        U = pickle.load(f)
    if noise == "random":
        with open(EXT / "feat_flickr_random0.40.pkl", "rb") as f:
            U2, V = pickle.load(f)
        assert np.allclose(U, U2)
    else:
        with open(EXT / "feat_flickr_txt_noun0.40.pkl", "rb") as f:
            V = pickle.load(f)
        # the image cache was built from one of the two CSVs; verify row order
        d0 = pd.read_csv(DERIVED / "flickr30k" / "flickr30k__random__lvl0.40__seed0.csv",
                         encoding="utf-8-sig")
        assert (d0.image_id.to_numpy() == df.image_id.to_numpy()).all()
    assert len(df) == len(U) == len(V)
    return df, U, V


def load_mmimdb():
    p = DERIVED / "mmimdb" / "mmimdb__cat__lvl0.40__seed0.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    with open(EXT / "feat_mmimdb.pkl", "rb") as f:
        U, V = pickle.load(f)
    assert len(df) == len(U)
    return df, U, V


def load_foil():
    with open(FOIL / "feat_test.pkl", "rb") as f:
        U, Vp, Vn = pickle.load(f)
    return U, Vp, Vn


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m, out, run = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


# ---------------------------------------------------------------- main
def main() -> int:
    rng = np.random.default_rng(0)
    recs = []
    for ds, facet, cycles in SOURCES:
        print(f"\n== source {ds}/{facet} ==", flush=True)
        models = {k: [train_head(k, ds, facet, cycles, s) for s in SEEDS]
                  for k in ("uvdot", "v")}
        crd = crd_scorer(ds, facet)

        def scores(U, V):
            return {"cosine": -np.einsum("ij,ij->i", U, V),
                    "crd": crd(U, V),
                    "probe": infer(models["uvdot"], U, V, "uvdot"),
                    "text": infer(models["v"], U, V, "v")}

        def emit(name, protocol, metric, y, U, V, pair=None):
            rec = summarize(name, protocol, metric, y, scores(U, V), pair, rng)
            rec["source"] = f"{ds}/{facet}"
            recs.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)

        # 1. SugarCrepe: same-image two-caption, no train split
        df, U, V = load_sugarcrepe()
        emit("SugarCrepe-comp", "same-image 2-way", "pair",
             df.is_mislabel.to_numpy().astype(int), U, V, df.pair_id.to_numpy())

        # 2-3. Flickr30k: external corpus and noise family, test split
        for noise in ("random", "noun"):
            df, U, V = load_flickr(noise)
            m = (df.split == "test").to_numpy()
            emit(f"Flickr30k-{noise}", "row-level", "auroc",
                 df.is_mislabel.to_numpy()[m].astype(int), U[m], V[m])

        # 4. MM-IMDb: posters + plots, test split
        df, U, V = load_mmimdb()
        m = (df.split == "test").to_numpy()
        emit("MM-IMDb-cat", "row-level", "auroc",
             df.is_mislabel.to_numpy()[m].astype(int), U[m], V[m])

        # 5. FOIL: same-image one-word counterfactual (labels implicit)
        Uf, Vp, Vn = load_foil()
        n = len(Uf)
        y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
        pair = np.r_[np.arange(n), np.arange(n)]
        U2, V2 = np.vstack([Uf, Uf]), np.vstack([Vp, Vn])
        emit("FOIL-COCO", "same-image 2-way", "pair", y, U2, V2, pair)

    df = pd.DataFrame(recs)
    for key in ("probe_vs_crd", "text_vs_crd", "crd_vs_cos"):
        col = f"p_{key}"
        df[f"p_holm_{key}"] = np.round(holm(df[col].to_numpy()), 4)
    df.to_csv(EXT / "probe_external.csv", index=False, encoding="utf-8-sig")
    (EXT / "probe_external.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8")
    cols = ["source", "target", "metric", "cosine", "crd", "probe", "text",
            "d_crd_vs_cos_pp", "d_probe_vs_crd_pp", "ci_probe_vs_crd",
            "p_probe_vs_crd", "p_holm_probe_vs_crd",
            "d_text_vs_crd_pp", "p_holm_text_vs_crd"]
    print("\n== summary ==")
    print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
