"""RC-LoRA: relation-constrained LoRA.

Two supervision relations share ONE scorer s_theta(I,c) = cos(f_theta(I), g_theta(c)):

    A (source mismatch) : clean rows vs noisy rows        -> L_src
    B (counterfactual)  : FOIL positive vs foil caption   -> L_sem

    min L_src   s.t.  L_sem <= tau_sem
    L = L_src + lam * relu(L_sem - tau),  lam <- max(0, lam + rho * (L_sem - tau))

M0 = source only (R200 recipe), M1 = fixed-weight combination, M2 = dual ascent.
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

from lora_stage1 import inject                                  # noqa: E402
from src.encoders import JointEncoder                           # noqa: E402
from src.protocol import (CLIP_DIR, PROJECT, RESULTS_DIR, load_foil_val,  # noqa: E402
                          load_split, write_json)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dual", choices=["src", "fixed", "dual"])
    ap.add_argument("--bench", default="coco", choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor", type=float, default=5.0)
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--foil-eval", type=int, default=2000)
    ap.add_argument("--calib", action="store_true",
                    help="compute L_sem on FOIL val and report it as tau")
    ap.add_argument("--tau-file", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    enc = JointEncoder(CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device
    for p in model.parameters():
        p.requires_grad = False
    inject(model)
    model.to(dev)

    tr = load_split(args.bench, "train", 6000).reset_index(drop=True)
    cl = tr[tr.label == 0]; no = tr[tr.label == 1]
    foil = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                       encoding="utf-8-sig").sample(8000, random_state=0)

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
        return (v * t).sum(1), v, t

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    lam = args.lam0
    if args.tau_file:
        import json as _json
        args.tau = float(_json.load(open(args.tau_file, encoding="utf-8"))["result"]["tau_suggested"])
        print(f"tau from file: {args.tau:.4f}", flush=True)
    model.train()

    for step in range(args.steps):
        k = args.batch
        c = cl.sample(k, random_state=None); n = no.sample(k, random_state=None)
        px, tk = prep(c["path"].tolist(), c["caption"].astype(str).tolist())
        sp, v1, t1 = s_of(px, tk)
        px2, tk2 = prep(n["path"].tolist(), n["caption"].astype(str).tolist())
        sn, _, _ = s_of(px2, tk2)
        L_src = F.softplus(0.1 - sp.mean() + sn.mean())

        if args.mode == "src":
            loss = L_src + args.anchor * ((1 - v1 * v1).sum() * 0)
            loss = L_src
        else:
            f = foil.sample(k, random_state=None)
            ppx, ptk = prep(f["path"].tolist(), f["positive_caption"].astype(str).tolist())
            fps, _, _ = s_of(ppx, ptk)
            np_px, np_tk = prep(f["path"].tolist(), f["negative_caption"].astype(str).tolist())
            fns, _, _ = s_of(np_px, np_tk)
            L_sem = F.softplus(0.1 - fps.mean() + fns.mean())
            if args.mode == "fixed":
                loss = L_src + args.lam0 * L_sem
            else:
                viol = torch.clamp(L_sem - args.tau, min=0.0)
                loss = L_src + lam * viol
                lam = max(0.0, lam + args.rho * float(viol.detach()))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1} loss {float(loss):.3f} lam {lam:.2f}", flush=True)

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
    kk = 10
    h = np.partition(vi @ tb.T, -kk, axis=1)[:, -kk:].mean(1) + \
        np.partition(vv @ vb.T, -kk, axis=1)[:, -kk:].mean(1)
    yv = va["label"].to_numpy()
    res = {"row_auc_cos": float(roc_auc_score(yv, -cosv)),
           "row_auc_csls": float(roc_auc_score(yv, -cosv + 0.8 * h))}

    fv = load_foil_val(args.foil_eval)
    fi = enc_i(fv["path"].tolist())
    fp = enc_t(fv["positive"].astype(str).tolist())
    fn = enc_t(fv["negative"].astype(str).tolist())
    spn, snn = (fi * fp).sum(1), (fi * fn).sum(1)
    res["foil_acc_cos"] = float(np.mean(snn < spn))
    fb = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                     encoding="utf-8-sig").sample(8000, random_state=0)
    fbi = enc_i(fb["path"].tolist())
    fbt = enc_t(fb["positive_caption"].astype(str).tolist())
    h_img = np.partition(fi @ fbt.T, -kk, axis=1)[:, -kk:].mean(1)
    hp = np.partition(fp @ fbi.T, -kk, axis=1)[:, -kk:].mean(1)
    hn = np.partition(fn @ fbi.T, -kk, axis=1)[:, -kk:].mean(1)
    res["foil_acc_csls"] = float(np.mean((-snn + 0.8 * (h_img + hn)) >
                                         (-spn + 0.8 * (h_img + hp))))
    res["lam_final"] = float(lam)
    if args.calib:
        Lval = float(F.softplus(0.1 - torch.tensor(spn) + torch.tensor(snn)).mean())
        res["L_sem_val"] = Lval
        res["tau_suggested"] = Lval
    out = {"mode": args.mode, "bench": args.bench, "seed": args.seed,
           "steps": args.steps, "result": res}
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"rc_{args.mode}_s{args.seed}_{args.bench}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
