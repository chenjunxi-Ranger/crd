"""VES pre-experiment: can a LEARNED evidence selector beat a heuristic one?

Pipeline (all CLIP parameters frozen)
------------------------------------
1. for N training images, occlude each of the 49 patches in turn at pixel level
   and record the drop in cos(image, caption)  -> per-patch effect (ground truth)
2. fit a tiny MLP g_theta on q_j = [v_j, t, v_j*t, |v_j-t|] -> effect_j
3. on held-out pairs, occlude the top-rho patches chosen by
      (a) CLIP patch-text similarity (heuristic)
      (b) learned selector g_theta
      (c) random (M draws, control)
   and compute E = mean(s_random) - s_erase, then test the within-pair direction.

This is pixel-level, so the selector is evaluated on the same quantity it is
trained to predict; no feature-space proxy is used for the final metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_foil_val, write_json  # noqa: E402


class Selector(nn.Module):
    def __init__(self, d: int = 512, h: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * d, h), nn.GELU(),
                                 nn.Linear(h, h // 2), nn.GELU(),
                                 nn.Linear(h // 2, 1))

    def forward(self, v, t):                       # v (...,P,d), t (...,d)
        if v.dim() == 2:
            v, t = v.unsqueeze(0), t.unsqueeze(0)
        t = t.unsqueeze(-2).expand_as(v)
        q = torch.cat([v, t, v * t, (v - t).abs()], dim=-1)
        return self.net(q).squeeze(-1)             # (...,P)


def patch_grid() -> int:
    return 7


def make_mask_batch(px: torch.Tensor, idx_list, grid: int = 7) -> torch.Tensor:
    """px: (1,3,H,W); idx_list: list of lists of patch ids -> (B,3,H,W)."""
    _, c, h, w = px.shape
    ph, pw = h // grid, w // grid
    out = px.repeat(len(idx_list), 1, 1, 1).clone()
    for b, idx in enumerate(idx_list):
        for j in idx:
            r, cc = divmod(j, grid)
            out[b, :, r * ph:(r + 1) * ph, cc * pw:(cc + 1) * pw] = 0.5
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pairs", type=int, default=200)
    ap.add_argument("--test-pairs", type=int, default=100)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--m-random", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    model, proc = enc.model, enc.processor
    grid = patch_grid()
    P = grid * grid
    k = max(1, int(round(args.rho * P)))

    val = load_foil_val(args.train_pairs + args.test_pairs)

    def prep(path):
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width * im.height > 20_000_000:
                im.draft("RGB", (1024, 1024))
            x = proc(images=[im], return_tensors="pt")["pixel_values"]
        return x.to(enc.device)

    @torch.no_grad()
    def embed_image(px):
        vis = model.vision_model(pixel_values=px)
        pooled = F.normalize(model.visual_projection(vis.pooler_output).float(), dim=-1)
        patch = F.normalize(model.visual_projection(vis.last_hidden_state[:, 1:, :]).float(), dim=-1)
        return pooled, patch

    @torch.no_grad()
    def embed_text(cap):
        t = proc(text=[cap], return_tensors="pt", padding=True, truncation=True,
                 max_length=77)
        t = {kk: v.to(enc.device) for kk, v in t.items()}
        out = model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
        return F.normalize(model.text_projection(out.pooler_output).float(), dim=-1)

    # ---------- stage 1: per-patch occlusion effects on the training pairs
    effects, feats, texts = [], [], []
    for i in range(args.train_pairs):
        px = prep(val["path"].iloc[i])
        _, patch = embed_image(px)
        tp = embed_text(val["positive"].iloc[i])
        with torch.no_grad():
            base = float(embed_image(px)[0] @ tp.T)
            idxs = [[j] for j in range(P)]
            batch = make_mask_batch(px, idxs, grid)
            vis = model.vision_model(pixel_values=batch)
            pooled = F.normalize(model.visual_projection(vis.pooler_output).float(), dim=-1)
            occ = (pooled @ tp.T).squeeze(-1)
        effects.append((base - occ.cpu()).numpy())
        feats.append(patch[0].cpu())
        texts.append(tp[0].cpu())
        if (i + 1) % 50 == 0:
            print(f"occlusion {i + 1}/{args.train_pairs}", flush=True)

    V = torch.stack(feats).to(enc.device)     # (N,P,d)
    T = torch.stack(texts).to(enc.device)     # (N,d)
    Y = torch.tensor(np.stack(effects), dtype=torch.float32).to(enc.device)  # (N,P)
    mu, sd = Y.mean(), Y.std().clamp_min(1e-6)

    sel = Selector(V.shape[-1]).to(enc.device)
    opt = torch.optim.Adam(sel.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        perm = torch.randperm(len(V), device=enc.device)
        total = 0.0
        for b in range(0, len(V), 32):
            idx = perm[b:b + 32]
            pred = sel(V[idx], T[idx])
            loss = F.mse_loss(pred, (Y[idx] - mu) / sd)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
    print("selector trained, last loss", total / max(1, len(V) // 32))

    # ---------- stage 2: pixel-level evaluation on held-out pairs
    results = {"rho": args.rho, "k_patches": k, "n_train": args.train_pairs,
               "n_test": args.test_pairs, "m_random": args.m_random}
    for policy in ("clip_sim", "learned", "random"):
        sp0, sn0, sp_e, sn_e, sp_r, sn_r = [], [], [], [], [], []
        for i in range(args.train_pairs, args.train_pairs + args.test_pairs):
            px = prep(val["path"].iloc[i])
            tp = embed_text(val["positive"].iloc[i])
            tn = embed_text(val["negative"].iloc[i])
            with torch.no_grad():
                p0, patch = embed_image(px)
                s0p, s0n = float(p0 @ tp.T), float(p0 @ tn.T)
                if policy == "clip_sim":
                    sim = (patch[0] @ tp[0])
                    idx = sim.topk(k).indices.tolist()
                elif policy == "learned":
                    a = sel(patch[0], tp[0]).flatten()
                    idx = a.topk(k).indices.tolist()
                else:
                    idx = torch.randperm(P)[:k].tolist()
                # erase selected
                er = make_mask_batch(px, [idx], grid)
                pe, _ = embed_image(er)
                sep, sen = float(pe @ tp.T), float(pe @ tn.T)
                # random controls
                rnd = [torch.randperm(P)[:k].tolist() for _ in range(args.m_random)]
                rb = make_mask_batch(px, rnd, grid)
                pr, _ = embed_image(rb)
                srp = float((pr @ tp.T).mean())
                srn = float((pr @ tn.T).mean())
            sp0.append(s0p); sn0.append(s0n); sp_e.append(sep); sn_e.append(sen)
            sp_r.append(srp); sn_r.append(srn)

        sp0, sn0 = np.array(sp0), np.array(sn0)
        Ep = np.array(sp_r) - np.array(sp_e)
        En = np.array(sn_r) - np.array(sn_e)
        clip_ok = sn0 < sp0
        ves_ok = En > Ep
        results[policy] = {
            "clip_acc": float(clip_ok.mean()),
            "ves_dir_acc": float(np.mean(np.where(En > Ep, 1.0,
                                                  np.where(En < Ep, 0.0, 0.5)))),
            "union_acc": float(np.mean(clip_ok | ves_ok)),
            "clip_wrong_ves_right": int(np.sum(~clip_ok & ves_ok)),
            "clip_right_ves_wrong": int(np.sum(clip_ok & ~ves_ok)),
            "mean_E_pos": float(Ep.mean()),
            "mean_E_neg": float(En.mean()),
        }
        print(policy, results[policy], flush=True)

    write_json(RESULTS_DIR / "gate" / "ves_pre.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
