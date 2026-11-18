"""FMP-LoRA: function-space margin projection.

Start from the R200 source teacher theta_T. Only the FOIL (semantic counterfactual)
objective is optimized. A linearized constraint protects the teacher's source
margins d_i = s(x+) - s(x-):

    delta0 = -eta * g_F
    if  mean(d^theta) + <J, delta0> <  mean(d^T) - eps:
        delta = delta0 + lam * J,   lam from the single hyperplane projection

M1 (control) = identical init / data / budget, FOIL-only, no projection.
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import lora_stage1 as _lora                                   # noqa: E402
from src.encoders import JointEncoder                          # noqa: E402
from src.protocol import (CLIP_DIR, PROJECT, RESULTS_DIR, load_foil_val,  # noqa: E402
                          load_split, write_json)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fmp", choices=["fmp", "foil", "r2r"])
    ap.add_argument("--teacher", default="results/gate/r200_teacher_coco.pt")
    ap.add_argument("--bench", default="coco")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eta", type=float, default=1e-4)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--pairs", type=int, default=256)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--foil-eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    enc = JointEncoder(CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device
    for p in model.parameters():
        p.requires_grad = False
    nv = len(model.vision_model.encoder.layers)
    nt = len(model.text_model.encoder.layers)
    _lora.BLOCKS = (nv - 2, nt - 2) if False else (min(nv, nt) - 2, min(nv, nt) - 1)
    _lora.inject(model)
    model.to(dev)

    sd = torch.load(ROOT / args.teacher, map_location=dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"teacher loaded {len(sd)} tensors", flush=True)

    # frozen teacher copy for margin targets
    teacher = {k: v.detach().clone() for k, v in model.state_dict().items()}

    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]

    tr = load_split("coco", "train", 6000).reset_index(drop=True)
    cl = tr[tr.label == 0].head(args.pairs).reset_index(drop=True)
    no = tr[tr.label == 1].head(args.pairs).reset_index(drop=True)
    foil = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                       encoding="utf-8-sig").sample(8000, random_state=0).reset_index(drop=True)

    def prep(paths, caps):
        px = proc(images=[Image.open(p).convert("RGB") for p in paths],
                  return_tensors="pt")["pixel_values"].to(dev)
        tk = proc(text=list(caps), return_tensors="pt", padding=True,
                  truncation=True, max_length=77).to(dev)
        return px, tk

    def s_of(px, tk):
        vo = model.vision_model(pixel_values=px)
        to = model.text_model(**tk)
        v = F.normalize(model.visual_projection(vo.pooler_output).float(), dim=-1)
        t = F.normalize(model.text_projection(to.pooler_output).float(), dim=-1)
        return (v * t).sum(1)

    def source_margin(idx):
        c = cl.iloc[idx]; n = no.iloc[idx]
        px, tk = prep(c["path"].tolist(), c["caption"].astype(str).tolist())
        sp = s_of(px, tk)
        px2, tk2 = prep(n["path"].tolist(), n["caption"].astype(str).tolist())
        sn = s_of(px2, tk2)
        return (sp - sn).mean()

    POOL = np.arange(min(64, len(cl)))
    diag = {"normJ": [], "D": [], "corr": [], "trigger": 0, "skip": 0, "accept": 0}

    # teacher margins (fixed targets)
    model.eval()
    with torch.no_grad():
        d_T = float(source_margin(POOL))
    print(f"teacher mean source margin {d_T:.4f}", flush=True)
    model.train()

    for step in range(args.steps):
        f = foil.sample(args.batch, random_state=None)
        px, tk = prep(f["path"].tolist(), f["positive_caption"].astype(str).tolist())
        sp = s_of(px, tk)
        px2, tk2 = prep(f["path"].tolist(), f["negative_caption"].astype(str).tolist())
        sn = s_of(px2, tk2)
        L_foil = F.softplus(0.1 - sp + sn).mean()

        if args.mode == "r2r":
            # source-relation replay on the same LoRA parameters
            sc = cl.sample(min(64, len(cl)), random_state=None)
            sn_ = no.sample(min(64, len(no)), random_state=None)
            qx, qk = prep(sc["path"].tolist(), sc["caption"].astype(str).tolist())
            s_src_p = s_of(qx, qk)
            qx2, qk2 = prep(sn_["path"].tolist(), sn_["caption"].astype(str).tolist())
            s_src_n = s_of(qx2, qk2)
            L_src = F.softplus(0.1 - s_src_p + s_src_n).mean()
            loss = 0.5 * L_src + 0.5 * L_foil
            g = torch.autograd.grad(loss, params, allow_unused=True)
            g = [torch.zeros_like(p) if gg is None else gg for gg, p in zip(g, params)]
            with torch.no_grad():
                for p, gg in zip(params, g):
                    p -= args.eta * gg
            if (step + 1) % 50 == 0:
                print(f"  step {step+1} Lsrc {float(L_src):.4f} Lfoil {float(L_foil):.4f}",
                      flush=True)
            continue

        g_F = torch.autograd.grad(L_foil, params, allow_unused=True)
        g_F = [torch.zeros_like(p) if g is None else g for g, p in zip(g_F, params)]

        if args.mode == "foil":
            with torch.no_grad():
                for p, g in zip(params, g_F):
                    p -= args.eta * g
        else:
            # fixed source pool -> deterministic D(theta), J and post-check
            dS = source_margin(POOL)
            J = torch.autograd.grad(dS, params, allow_unused=True)
            J = [torch.zeros_like(p) if g is None else g for g, p in zip(J, params)]
            jj = sum(float((g * g).sum()) for g in J) + 1e-12
            jd0 = sum(float((g * (-args.eta * gF)).sum()) for g, gF in zip(J, g_F))
            b = d_T - args.eps - float(dS)
            diag["normJ"].append(jj ** 0.5)
            diag["D"].append(float(dS))
            viol = b - jd0
            if viol > 0:                      # project onto the linearized half-space
                lam = viol / jj
                diag["trigger"] += 1
            else:
                lam = 0.0
            delta = [(-args.eta * gF) + lam * g for gF, g in zip(g_F, J)]
            diag["corr"].append(sum(float((lam * g).norm() ** 2) for g in J) ** 0.5)
            # apply with post-hoc feasibility check (halve up to 4 times, else revert)
            backup = [p.detach().clone() for p in params]
            ok = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                with torch.no_grad():
                    for p, s, d in zip(params, backup, delta):
                        p.copy_(s + scale * d)
                with torch.no_grad():
                    real = float(source_margin(POOL))
                if lam == 0.0 or real >= d_T - args.eps - args.tol:
                    ok = True
                    break
            if not ok:
                with torch.no_grad():
                    for p, s in zip(params, backup):
                        p.copy_(s)
                diag["skip"] += 1
            else:
                diag["accept"] += 1
        if (step + 1) % 50 == 0:
            print(f"  step {step+1} L_foil {float(L_foil):.4f} "
                  f"lam_trig {diag['trigger']} skip {diag['skip']}", flush=True)

    # ---- evaluation
    model.eval()

    @torch.no_grad()
    def enc_i(paths, bs=64):
        out = []
        for i in range(0, len(paths), bs):
            px = proc(images=[Image.open(p).convert("RGB") for p in paths[i:i + bs]],
                      return_tensors="pt")["pixel_values"].to(dev)
            vo = model.vision_model(pixel_values=px)
            out.append(F.normalize(model.visual_projection(vo.pooler_output).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def enc_t(caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            tk = proc(text=list(caps[i:i + bs]), return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(dev)
            to = model.text_model(**tk)
            out.append(F.normalize(model.text_projection(to.pooler_output).float(),
                                   dim=-1).cpu().numpy())
        return np.concatenate(out)

    va = load_split(args.bench, "val")
    vi, vv = enc_i(va["path"].tolist()), enc_t(va["caption"].astype(str).tolist())
    bank = load_split(args.bench, "train", 15000)
    vb, tb = enc_i(bank["path"].tolist()), enc_t(bank["caption"].astype(str).tolist())
    cosv = (vi * vv).sum(1)
    k = 10
    h = np.partition(vi @ tb.T, -k, axis=1)[:, -k:].mean(1) + \
        np.partition(vv @ vb.T, -k, axis=1)[:, -k:].mean(1)
    yv = va["label"].to_numpy()
    res = {"row_cos": float(roc_auc_score(yv, -cosv)),
           "row_csls": float(roc_auc_score(yv, -cosv + 0.8 * h))}

    fv = load_foil_val(args.foil_eval)
    fi = enc_i(fv["path"].tolist())
    fp = enc_t(fv["positive"].astype(str).tolist())
    fn = enc_t(fv["negative"].astype(str).tolist())
    spn, snn = (fi * fp).sum(1), (fi * fn).sum(1)
    res["foil_cos"] = float(np.mean(snn < spn))
    fb = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                     encoding="utf-8-sig").sample(8000, random_state=0)
    fbi = enc_i(fb["path"].tolist()); fbt = enc_t(fb["positive_caption"].astype(str).tolist())
    hi = np.partition(fi @ fbt.T, -k, axis=1)[:, -k:].mean(1)
    hp = np.partition(fp @ fbi.T, -k, axis=1)[:, -k:].mean(1)
    hn = np.partition(fn @ fbi.T, -k, axis=1)[:, -k:].mean(1)
    res["foil_csls"] = float(np.mean((-snn + 0.8 * (hi + hn)) > (-spn + 0.8 * (hi + hp))))

    out = {"mode": args.mode, "bench": args.bench, "seed": args.seed,
           "steps": args.steps, "eps": args.eps, "result": res,
           "teacher_margin": float(d_T),
           "diag": {"trigger_rate": diag["trigger"] / max(1, args.steps),
                    "accept_rate": diag["accept"] / max(1, args.steps),
                    "skip_rate": diag["skip"] / max(1, args.steps),
                    "normJ_last": diag["normJ"][-1] if diag["normJ"] else None,
                    "corr_last": diag["corr"][-1] if diag["corr"] else None,
                    "D_last": diag["D"][-1] if diag["D"] else None}}
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"fmp_{args.mode}_s{args.seed}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
