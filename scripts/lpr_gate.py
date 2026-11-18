"""LPR admission gate: does the score trajectory across training checkpoints
carry information beyond the final score?

Trains one LoRA run, snapshots the adapter at 6 phases, scores the SAME val rows
with each snapshot, and compares
    M0 = [s_T]
    M1 = [s_T, early_gain, late_gain]
    M2 = M1 + [middle_gain, path_variation, late_early_contrast]
All features are per-checkpoint standardised with a train-subsample mean/std.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lora_stage1 import Data, inject, BLOCKS                      # noqa: E402
from src.encoders import JointEncoder                              # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402

FRACS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def cv_auc(X, y, seed=0):
    w = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    p = cross_val_predict(w, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco", choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--val-limit", type=int, default=1000)
    ap.add_argument("--train-cap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    enc = JointEncoder(CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device
    for p in model.parameters():
        p.requires_grad = False
    inject(model)
    model.to(dev).eval()

    def prep_img(p):
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.width * im.height > 20_000_000:
                im.draft("RGB", (1024, 1024))
            return proc(images=[im], return_tensors="pt")["pixel_values"][0]

    def prep_txt(c):
        t = proc(text=[c], return_tensors="pt", padding="max_length",
                 truncation=True, max_length=77)
        return t["input_ids"][0], t["attention_mask"][0]

    def encode(px, ids, mask, grad=False):
        with (torch.enable_grad() if grad else torch.no_grad()):
            vo = model.vision_model(pixel_values=px)
            to = model.text_model(input_ids=ids, attention_mask=mask)
            v = F.normalize(model.visual_projection(vo.pooler_output).float(), dim=-1)
            t = F.normalize(model.text_projection(to.pooler_output).float(), dim=-1)
        return v, t

    @torch.no_grad()
    def score_rows(df):
        out = []
        for i in range(len(df)):
            px = prep_img(df["path"].iloc[i]).unsqueeze(0).to(dev)
            ids, msk = prep_txt(str(df["caption"].iloc[i]))
            v, t = encode(px, ids.unsqueeze(0).to(dev), msk.unsqueeze(0).to(dev))
            out.append(float((v * t).sum()))
        return np.array(out)

    # ---- training data (same construction as the LoRA stage)
    data = Data(enc, 400, 200, 100, 100, seed=args.seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    marks = {max(1, int(round(f * args.steps))): f for f in FRACS}
    marks[0] = 0.0
    snaps = {}
    model.train()
    for step in range(0, args.steps + 1):
        if step in marks:
            snaps[marks[step]] = {k: v.detach().clone() for k, v in model.state_dict().items()
                                  if any(f"layers.{li}.self_attn.{pj}.{n}" in k
                                         for li in BLOCKS for pj in ("q_proj", "v_proj")
                                         for n in ("A", "B"))}
            if step == args.steps:
                break
        idx = rng.choice(len(data.rows), 24, replace=False)
        pxs, pi, pm, ni, nm = [], [], [], [], []
        for k in idx:
            path, a, b = data.rows[k]
            pxs.append(prep_img(path))
            i1, m1 = prep_txt(a); i2, m2 = prep_txt(b)
            pi.append(i1); pm.append(m1); ni.append(i2); nm.append(m2)
        px = torch.stack(pxs).to(dev)
        with torch.no_grad():
            v0, t0p = encode(px, torch.stack(pi).to(dev), torch.stack(pm).to(dev))
            _, t0n = encode(px, torch.stack(ni).to(dev), torch.stack(nm).to(dev))
        v, tp = encode(px, torch.stack(pi).to(dev), torch.stack(pm).to(dev), grad=True)
        _, tn = encode(px, torch.stack(ni).to(dev), torch.stack(nm).to(dev), grad=True)
        loss = F.softplus(0.1 - ((v * tp).sum(1) - (v * tn).sum(1))).mean()
        logits = (v @ tp.t()) / 0.07
        tgt = torch.arange(len(v), device=dev)
        loss = loss + 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
        anchor = (1 - (v * v0).sum(1)).mean() + 0.5 * (
            (1 - (tp * t0p).sum(1)).mean() + (1 - (tn * t0n).sum(1)).mean())
        loss = loss + 5.0 * anchor
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    # ---- score the same val rows at every checkpoint
    df = load_split(args.bench, "val", args.val_limit)
    y = df["label"].to_numpy()
    traj = []
    for f in FRACS:
        model.load_state_dict(snaps[f], strict=False)
        traj.append(score_rows(df))
        print(f"  scored checkpoint {f}", flush=True)
    traj = np.stack(traj)                       # (T, n)

    # per-checkpoint standardisation with a train subsample
    tr = load_split(args.bench, "train", args.train_cap)
    norm = []
    for f in FRACS:
        model.load_state_dict(snaps[f], strict=False)
        smp = score_rows(tr.rename(columns={tr.columns[tr.columns.get_loc("caption")]: "caption"})
                         if "caption" in tr else tr)
        norm.append((float(smp.mean()), float(smp.std()) + 1e-6))

    Z = np.stack([(traj[i] - norm[i][0]) / norm[i][1] for i in range(len(FRACS))])
    sT = Z[-1]
    feat = {"init": Z[0], "early": Z[2] - Z[0], "mid": Z[4] - Z[2],
            "late": Z[5] - Z[4], "path": np.abs(np.diff(Z, axis=0)).sum(0),
            "contrast": (Z[5] - Z[4]) - (Z[2] - Z[0])}
    s = -sT
    m0 = float(roc_auc_score(y, s))
    m1 = cv_auc(np.column_stack([s, feat["early"], feat["late"]]), y)
    m2 = cv_auc(np.column_stack([s, feat["early"], feat["late"],
                                 feat["mid"], feat["path"], feat["contrast"]]), y)
    out = {"bench": args.bench, "steps": args.steps, "n": int(len(y)),
           "auc_cos_final": m0, "auc_M1": m1, "auc_M2": m2,
           "delta_traj": m2 - m1,
           "mean_early_clean": float(feat["early"][y == 0].mean()),
           "mean_early_noisy": float(feat["early"][y == 1].mean()),
           "mean_path_clean": float(feat["path"][y == 0].mean()),
           "mean_path_noisy": float(feat["path"][y == 1].mean())}
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"lpr_{args.bench}.json", out)
    np.savez_compressed(RESULTS_DIR / "gate" / f"lpr_{args.bench}.npz",
                        traj=traj, label=y, fracs=np.array(FRACS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
