"""Per-dataset VES under relaxed C1 (same formula, per-dataset constants).

For one benchmark:
  train  : official train split, donor-swap negatives, pixel-level teacher
  val    : choose lambda on val (per-dataset now allowed)
  test   : evaluated ONCE with the frozen lambda

Formula (same for all datasets):
    A(I,c) = -cos(I,c) - lambda * E_theta(I,c)
    E(I,c) = TopMean_K( g_theta([v_j, t, v_j*t, |v_j-t|]) )
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402
from ves_train import P, Selector, mask_batch             # noqa: E402

K_TOP = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--train-images", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--test-limit", type=int, default=None)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    dev = enc.device
    proc, model = enc.processor, enc.model

    def prep(path):
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width * im.height > 20_000_000:
                im.draft("RGB", (1024, 1024))
            return proc(images=[im], return_tensors="pt")["pixel_values"].to(dev)

    @torch.no_grad()
    def emb_img(px):
        vis = model.vision_model(pixel_values=px)
        pooled = F.normalize(model.visual_projection(vis.pooler_output).float(), dim=-1)
        patch = F.normalize(model.visual_projection(vis.last_hidden_state[:, 1:, :]).float(), dim=-1)
        return pooled, patch

    @torch.no_grad()
    def emb_txt(cap):
        t = proc(text=[cap], return_tensors="pt", padding=True, truncation=True, max_length=77)
        t = {k: v.to(dev) for k, v in t.items()}
        o = model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
        return F.normalize(model.text_projection(o.pooler_output).float(), dim=-1)

    # ---------------- teacher on the official train split
    tr = load_split(args.bench, "train")
    tr = tr.sample(min(args.train_images, len(tr)), random_state=0).reset_index(drop=True)
    caps = tr["caption"].astype(str).tolist()
    perm = np.random.default_rng(0).permutation(len(caps))
    V, Tp, Tn, Dp, Dn = [], [], [], [], []
    for i in range(len(tr)):
        px = prep(tr["path"].iloc[i])
        p0, patch = emb_img(px)
        pos = caps[i]
        neg = caps[perm[i]] if perm[i] != i else caps[(i + 1) % len(caps)]
        tp, tn_ = emb_txt(pos), emb_txt(neg)
        s0p, s0n = float(p0 @ tp.T), float(p0 @ tn_.T)
        with torch.no_grad():
            occ = emb_img(mask_batch(px, [[j] for j in range(P)]))[0]
            dp = s0p - (occ @ tp.T).squeeze(-1).cpu().numpy()
            dn = s0n - (occ @ tn_.T).squeeze(-1).cpu().numpy()
        V.append(patch[0].cpu()); Tp.append(tp[0].cpu()); Tn.append(tn_[0].cpu())
        Dp.append(dp); Dn.append(dn)
        if (i + 1) % 200 == 0:
            print(f"teacher {i + 1}/{len(tr)}", flush=True)

    V = torch.stack(V).to(dev); Tp = torch.stack(Tp).to(dev); Tn = torch.stack(Tn).to(dev)
    Dp = torch.tensor(np.stack(Dp), dtype=torch.float32, device=dev)
    Dn = torch.tensor(np.stack(Dn), dtype=torch.float32, device=dev)

    sel = Selector().to(dev)
    opt = torch.optim.Adam(sel.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        p_ = torch.randperm(len(V), device=dev)
        for b in range(0, len(V), 32):
            idx = p_[b:b + 32]
            pp_, pn_ = sel(V[idx], Tp[idx]), sel(V[idx], Tn[idx])
            loss = F.huber_loss(pp_, Dp[idx], delta=0.01) + F.huber_loss(pn_, Dn[idx], delta=0.01)
            ep_ = pp_.topk(K_TOP, dim=1).values.mean(1)
            en_ = pn_.topk(K_TOP, dim=1).values.mean(1)
            loss = loss + F.softplus(0.1 - (ep_ - en_)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(sel.state_dict(), RESULTS_DIR / "gate" / f"selector_{args.bench}.pt")

    # ---------------- score val / test
    def score_split(split: str, limit):
        df = load_split(args.bench, split, limit)
        cos, en = [], []
        for i in range(0, len(df), 500):
            st = enc.encode(df["path"].iloc[i:i + 500].tolist(),
                            df["caption"].iloc[i:i + 500].tolist(),
                            batch_size=64, num_workers=4)
            cos.append((st.pooled_img * st.pooled_txt).sum(1))
            with torch.no_grad():
                e = sel(torch.as_tensor(st.patch, device=dev),
                        torch.as_tensor(st.pooled_txt, device=dev))
                en.append(e.topk(K_TOP, dim=1).values.mean(1).cpu().numpy())
        return df["label"].to_numpy(), np.concatenate(cos), np.concatenate(en)

    yv, cv, ev = score_split("val", args.val_limit)
    best = {"lam": 0.0, "auc": float(roc_auc_score(yv, -cv))}
    for lam in (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        a = float(roc_auc_score(yv, -cv - lam * ev))
        if a > best["auc"]:
            best = {"lam": lam, "auc": a}
    yt, ct, et = score_split("test", args.test_limit)
    test_auc = float(roc_auc_score(yt, -ct - best["lam"] * et))
    payload = {"bench": args.bench, "train_images": int(len(tr)),
               "val": {"n": int(len(yv)), "auc_cos": float(roc_auc_score(yv, -cv)),
                       "best_lambda": best["lam"], "auc": best["auc"]},
               "test": {"n": int(len(yt)), "auc_cos": float(roc_auc_score(yt, -ct)),
                        "auc": test_auc, "passed": bool(test_auc >= 0.96)}}
    if args.bench == "mmimdb":
        payload["test"]["passed"] = bool(test_auc >= 0.88)
    write_json(RESULTS_DIR / "gate" / f"ves_{args.bench}.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
