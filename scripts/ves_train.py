"""VES-0 / VES-1 / VES-2 under the revised C2 (shared, frozen-CLIP selector).

Teacher signals are PIXEL-LEVEL (occlusion on the 224x224 input), computed once
and cached; the MLP trains on cached patch/text features only, so CLIP never
enters the backward graph.

  VES-0 : L_patch on d+ only                (matches the earlier failing baseline)
  VES-1 : L_patch on d+ and d- + L_pair     (does positive/negative supervision help?)
  VES-2 : VES-1 + L_group                   (does group-level counterfactual ranking help?)

d_j(I,c) = s(I,c) - s(I \\ j, c)
z_j      = d_j(I,c+) - d_j(I,c-)
Group Z_M = D_M+ - D_M- for an equal-area mask M.

Evaluation on held-out FOIL val pairs: direction accuracy of E, complementarity
with CLIP, and a pixel-level check of the selected patches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import (CLIP_DIR, PROJECT, RESULTS_DIR, load_foil_val,  # noqa: E402
                          write_json)

GRID = 7
P = GRID * GRID


class Selector(nn.Module):
    def __init__(self, d: int = 512, h: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * d, h), nn.GELU(),
                                 nn.Linear(h, h // 2), nn.GELU(),
                                 nn.Linear(h // 2, 1))

    def forward(self, v, t):
        if v.dim() == 2:
            v, t = v.unsqueeze(0), t.unsqueeze(0)
        t = t.unsqueeze(-2).expand_as(v)
        q = torch.cat([v, t, v * t, (v - t).abs()], dim=-1)
        return self.net(q).squeeze(-1)          # (...,P)


def mask_batch(px: torch.Tensor, idx_list) -> torch.Tensor:
    _, c, h, w = px.shape
    ph, pw = h // GRID, w // GRID
    out = px.repeat(len(idx_list), 1, 1, 1).clone()
    for b, idx in enumerate(idx_list):
        for j in idx:
            r, cc = divmod(j, GRID)
            out[b, :, r * ph:(r + 1) * ph, cc * pw:(cc + 1) * pw] = 0.5
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", type=int, default=600)
    ap.add_argument("--val-pairs", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--k-top", type=int, default=10)
    ap.add_argument("--group-k", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixed", action="store_true",
                    help="also train on donor-swap pairs from the other three train splits")
    ap.add_argument("--per-bench", type=int, default=300)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    enc = JointEncoder(CLIP_DIR)
    model, proc = enc.model, enc.processor
    dev = enc.device

    # ---------------- data (image-disjoint from the frozen val split)
    val = load_foil_val(10000)
    val_names = set(val["path"].str.replace("\\", "/").str.split("/").str[-1])
    train = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                        encoding="utf-8-sig")
    tn = train["path"].str.replace("\\", "/").str.split("/").str[-1]
    train = train[~tn.isin(val_names)].drop_duplicates("path").head(args.train_images)
    rows = [{"path": p, "pos": str(a), "neg": str(b)}
            for p, a, b in zip(train["path"], train["positive_caption"],
                               train["negative_caption"])]
    if args.mixed:
        from src.protocol import load_split
        for bench in ("coco", "flickr", "mmimdb"):
            d = load_split(bench, "train")
            d = d.sample(min(args.per_bench, len(d)), random_state=0)
            caps = d["caption"].astype(str).tolist()
            perm = np.random.default_rng(0).permutation(len(caps))
            for k, (p, c) in enumerate(zip(d["path"], caps)):
                donor = caps[perm[k]] if perm[k] != k else caps[(k + 1) % len(caps)]
                rows.append({"path": p, "pos": c, "neg": donor})
        np.random.default_rng(0).shuffle(rows)
    print(f"train images={len(rows)} (foil={min(len(train), args.train_images)}, "
          f"mixed={args.mixed}) val pairs={args.val_pairs}")

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
        t = proc(text=[cap], return_tensors="pt", padding=True, truncation=True,
                 max_length=77)
        t = {k: v.to(dev) for k, v in t.items()}
        o = model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
        return F.normalize(model.text_projection(o.pooler_output).float(), dim=-1)

    # ---------------- teacher generation
    V, Tp, Tn, Dp, Dn = [], [], [], [], []
    groups: list[list[tuple[list[int], float]]] = []
    rng = np.random.default_rng(args.seed)
    for i in range(len(rows)):
        px = prep(rows[i]["path"])
        p0, patch = emb_img(px)
        tp = emb_txt(rows[i]["pos"])
        tn_ = emb_txt(rows[i]["neg"])
        s0p, s0n = float(p0 @ tp.T), float(p0 @ tn_.T)
        with torch.no_grad():
            occ = emb_img(mask_batch(px, [[j] for j in range(P)]))[0]
            dp = s0p - (occ @ tp.T).squeeze(-1).cpu().numpy()
            dn = s0n - (occ @ tn_.T).squeeze(-1).cpu().numpy()
            ks = [args.group_k, max(2, args.group_k // 2)]
            masks = [rng.choice(P, k, replace=False).tolist()
                     for k in ks for _ in range(3)]
            gp = emb_img(mask_batch(px, masks))[0]
            gs_p = (gp @ tp.T).squeeze(-1).cpu().numpy()
            gs_n = (gp @ tn_.T).squeeze(-1).cpu().numpy()
        gvals = [(m, float((s0p - gs_p[j]) - (s0n - gs_n[j])))
                 for j, m in enumerate(masks)]
        V.append(patch[0].cpu()); Tp.append(tp[0].cpu()); Tn.append(tn_[0].cpu())
        Dp.append(dp); Dn.append(dn); groups.append(gvals)
        if (i + 1) % 100 == 0:
            print(f"teacher {i + 1}/{len(rows)}", flush=True)

    V = torch.stack(V).to(dev)
    Tp = torch.stack(Tp).to(dev)
    Tn = torch.stack(Tn).to(dev)
    Dp = torch.tensor(np.stack(Dp), dtype=torch.float32, device=dev)
    Dn = torch.tensor(np.stack(Dn), dtype=torch.float32, device=dev)

    # ---------------- train the three variants
    def train_variant(variant: str):
        sel = Selector().to(dev)
        opt = torch.optim.Adam(sel.parameters(), lr=1e-3)
        n = len(V)
        for ep in range(args.epochs):
            perm = torch.randperm(n, device=dev)
            for b in range(0, n, 32):
                idx = perm[b:b + 32]
                pp = sel(V[idx], Tp[idx])
                pn = sel(V[idx], Tn[idx])
                if ep == 0 and b == 0:
                    print("shapes", tuple(V[idx].shape), tuple(Tp[idx].shape),
                          tuple(pp.shape), tuple(Dp[idx].shape), flush=True)
                loss = F.huber_loss(pp, Dp[idx], delta=0.01)
                if variant != "VES-0":
                    loss = loss + F.huber_loss(pn, Dn[idx], delta=0.01)
                    ep_ = pp.topk(args.k_top, dim=1).values.mean(1)
                    en_ = pn.topk(args.k_top, dim=1).values.mean(1)
                    loss = loss + args.alpha * F.softplus(args.gamma - (ep_ - en_)).mean()
                if variant == "VES-2":
                    diff = pp - pn                       # (B,P) predicted z
                    za, zb, sgn = [], [], []
                    for r, i_img in enumerate(idx.tolist()):
                        g = groups[i_img][:3]              # same-area masks only
                        a = max(g, key=lambda x: x[1])
                        b2 = min(g, key=lambda x: x[1])
                        if a[1] == b2[1]:
                            continue
                        za.append(diff[r, torch.tensor(a[0], device=dev)].sum())
                        zb.append(diff[r, torch.tensor(b2[0], device=dev)].sum())
                        sgn.append(1.0 if a[1] > b2[1] else -1.0)
                    if not za:
                        continue
                    za = torch.stack(za); zb = torch.stack(zb)
                    sg = torch.tensor(sgn, device=dev)
                    loss = loss + args.beta * F.softplus(-sg * (za - zb)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return sel

    # ---------------- evaluation
    def eval_variant(sel, limit: int, variant_tag: str = "ves"):
        sp0, sn0, Ep, En = [], [], [], []
        for i in range(limit):
            px = prep(val["path"].iloc[i])
            p0, patch = emb_img(px)
            tp = emb_txt(val["positive"].iloc[i]); tn_ = emb_txt(val["negative"].iloc[i])
            s0p, s0n = float(p0 @ tp.T), float(p0 @ tn_.T)
            with torch.no_grad():
                e_p = sel(patch[0], tp[0]).topk(args.k_top).values.mean().item()
                e_n = sel(patch[0], tn_[0]).topk(args.k_top).values.mean().item()
            sp0.append(s0p); sn0.append(s0n); Ep.append(e_p); En.append(e_n)
        sp0, sn0 = np.array(sp0), np.array(sn0)
        Ep, En = np.array(Ep), np.array(En)
        np.savez_compressed(RESULTS_DIR / "gate" / f"ves_pairs_{variant_tag}.npz",
                            s_pos=sp0, s_neg=sn0, e_pos=Ep, e_neg=En)
        clip_ok = sn0 < sp0
        ves_ok = Ep > En
        accs = {}
        # E lives at scale ~5, s0 at ~0.3: lambda must be O(1e-2), not O(1)
        for lam in (0.0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05, 0.1, 0.5):
            a = (-sn0 - lam * En) > (-sp0 - lam * Ep)
            accs[str(lam)] = float(a.mean())
        # gated fusion: use VES only where the CLIP margin is small
        margin = np.abs(sn0 - sp0)
        gated = {}
        for tau in (0.002, 0.005, 0.01, 0.02, 0.05, 10.0):
            use_ves = margin < tau
            pred_clip = -sn0 > -sp0
            pred_ves = Ep > En          # higher evidence for c+ => c- is the wrong one
            pred = np.where(use_ves, pred_ves, pred_clip)
            gated[str(tau)] = {"acc": float(pred.mean()),
                               "coverage": float(use_ves.mean())}
        return {
            "clip_acc": float(clip_ok.mean()),
            "ves_dir_acc": float(np.mean(np.where(Ep > En, 1.0,
                                                 np.where(Ep < En, 0.0, 0.5)))),
            "lambda_sweep": accs,
            "gated_fusion": gated,
            "clip_wrong_ves_right": int(np.sum(~clip_ok & ves_ok)),
            "clip_right_ves_wrong": int(np.sum(clip_ok & ~ves_ok)),
            "mean_E_pos": float(Ep.mean()), "mean_E_neg": float(En.mean()),
        }

    out = {"train_images": len(train), "val_pairs": args.val_pairs,
           "k_top": args.k_top, "group_k": args.group_k,
           "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma}
    for variant in ("VES-0", "VES-1", "VES-2"):
        sel = train_variant(variant)
        torch.save(sel.state_dict(), RESULTS_DIR / "gate" / f"selector_{variant}.pt")
        out[variant] = eval_variant(sel, args.val_pairs, variant)
        print(variant, out[variant], flush=True)

    write_json(RESULTS_DIR / "gate" / "ves_train.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
