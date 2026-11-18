"""SCSA-B: train the LoRA adapter against the FINAL deployment score.

    m(I,c) = s_theta(I,c) - lam * (r_I + r_c),  lam = 0.8, k = 10
    L = softplus(gamma - (m+ - m-)) + nce * contrastive + beta * anchor

Reference bank: encoded with the current adapter, stop-gradient on the bank side;
top-k indices are discrete, the selected similarities keep query-side gradients.
The bank is refreshed `--refresh` times during training.
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

from lora_stage1 import Data, inject                       # noqa: E402
from rat_gate import topk_mean                             # noqa: E402
from src.encoders import JointEncoder                      # noqa: E402
from src.protocol import (CLIP_DIR, PROJECT, RESULTS_DIR, load_foil_val,  # noqa: E402
                          load_split, write_json)

LAM, K = 0.8, 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bank-train", type=int, default=3000)
    ap.add_argument("--refresh", type=int, default=2)
    ap.add_argument("--bank", type=int, default=15000)
    ap.add_argument("--foil-eval", type=int, default=5000)
    args = ap.parse_args()

    outdir = RESULTS_DIR / "scsa_paired" / f"seed_{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    enc = JointEncoder(CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device
    for p in model.parameters():
        p.requires_grad = False
    inject(model)
    model.to(dev).eval()

    # same initial adapter for A and B
    torch.manual_seed(args.seed)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if any(f"self_attn.{p}.{n}" in k for p in ("q_proj", "v_proj")
                         for n in ("A", "B"))}
    torch.save(init_state, outdir / "init_adapter.pt")

    def proc_img(p):
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.width * im.height > 20_000_000:
                im.draft("RGB", (1024, 1024))
            return proc(images=[im], return_tensors="pt")["pixel_values"][0]

    def proc_txt(c):
        t = proc(text=[c], return_tensors="pt", padding="max_length",
                 truncation=True, max_length=77)
        return t["input_ids"][0], t["attention_mask"][0]

    def encode(px, ids, mask, no_grad=False):
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            vo = model.vision_model(pixel_values=px)
            to = model.text_model(input_ids=ids, attention_mask=mask)
            v = F.normalize(model.visual_projection(vo.pooler_output).float(), dim=-1)
            t = F.normalize(model.text_projection(to.pooler_output).float(), dim=-1)
        return v, t

    # ---------------- frozen-bank features for the training queries
    tr = load_split("coco", "train").sample(args.bank_train, random_state=args.seed)
    bank_paths = tr["path"].tolist()
    bank_txts = tr["caption"].astype(str).tolist()

    def build_bank():
        with torch.no_grad():
            bi = enc.encode_images(bank_paths, batch_size=64, num_workers=4)
            bt = enc.encode_texts(bank_txts)
        return torch.as_tensor(bi, device=dev), torch.as_tensor(bt, device=dev)

    bank_img, bank_txt = build_bank()

    def r_terms(v, t):
        si = v @ bank_txt.t()                     # image -> bank captions
        st = t @ bank_img.t()                     # caption -> bank images
        ri = torch.topk(si, K, dim=1).values.mean(1)
        rc = torch.topk(st, K, dim=1).values.mean(1)
        return ri, rc

    data = Data(enc, 800, 400, 200, 200, seed=args.seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    model.train()
    refresh_at = {int(args.steps * (i + 1) / (args.refresh + 1)) for i in range(args.refresh)}
    gradcheck_done = False
    for step in range(1, args.steps + 1):
        if step in refresh_at:
            bank_img, bank_txt = build_bank()
            print(f"seed{args.seed} bank refreshed at {step}", flush=True)
        idx = rng.choice(len(data.rows), 24, replace=False)
        pxs, pi, pm, ni, nm = [], [], [], [], []
        for k in idx:
            path, a, b = data.rows[k]
            pxs.append(proc_img(path))
            i1, m1 = proc_txt(a); i2, m2 = proc_txt(b)
            pi.append(i1); pm.append(m1); ni.append(i2); nm.append(m2)
        px = torch.stack(pxs).to(dev)
        PI, PM = torch.stack(pi).to(dev), torch.stack(pm).to(dev)
        NI, NM = torch.stack(ni).to(dev), torch.stack(nm).to(dev)
        with torch.no_grad():
            v0, t0p = encode(px, PI, PM, no_grad=True)
            _, t0n = encode(px, NI, NM, no_grad=True)
        v, tp = encode(px, PI, PM)
        _, tn = encode(px, NI, NM)
        sp, sn = (v * tp).sum(1), (v * tn).sum(1)
        rip, rcp = r_terms(v, tp)
        rin, rcn = r_terms(v, tn)
        mp = sp - LAM * (rip + rcp)
        mn = sn - LAM * (rin + rcn)
        loss = F.softplus(0.1 - (mp - mn)).mean()
        logits = (v @ tp.t()) / 0.07
        tgt = torch.arange(len(v), device=dev)
        loss = loss + 0.5 * (F.cross_entropy(logits, tgt) +
                             F.cross_entropy(logits.t(), tgt))
        anchor = (1 - (v * v0).sum(1)).mean() + \
                 0.5 * ((1 - (tp * t0p).sum(1)).mean() + (1 - (tn * t0n).sum(1)).mean())
        loss = loss + 5.0 * anchor
        opt.zero_grad(); loss.backward()
        if not gradcheck_done:
            gv = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                     if "vision_model" in n and p.grad is not None)
            gt = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                     if "text_model" in n and p.grad is not None)
            print(f"gradcheck: vision {gv:.2e} text {gt:.2e} "
                  f"(image-side term active: {gv > 0})", flush=True)
            gradcheck_done = True
        opt.step()
        if step % 100 == 0:
            print(f"seed{args.seed} step {step} loss {float(loss):.4f}", flush=True)

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if any(f"self_attn.{p}.{n}" in k for p in ("q_proj", "v_proj")
                    for n in ("A", "B"))}
    torch.save(state, outdir / "B_adapter.pt")
    model.eval()

    # ---------------- evaluation (same protocol as the seed audit)
    summary = {}
    for bench in ("coco", "flickr", "mmimdb"):
        path = outdir / f"bank_{bench}.npz"
        if not path.is_file():
            df = load_split(bench, "train").sample(
                min(args.bank, len(load_split(bench, "train"))), random_state=args.seed)
            bi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
            bt = enc.encode_texts(df["caption"].astype(str).tolist())
            np.savez_compressed(path, img=bi, txt=bt)
        df = load_split(bench, "val")
        vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
        vt = enc.encode_texts(df["caption"].tolist())
        with np.load(path) as z:
            bi, bt = z["img"], z["txt"]
        cos = (vi * vt).sum(1)
        s = -cos + LAM * (topk_mean(vi, bt) + topk_mean(vt, bi))
        y = df["label"].to_numpy()
        np.savez_compressed(outdir / f"B_scores_{bench}.npz", score=s, label=y,
                            image_id=df["image_id"].astype(str).to_numpy())
        summary[bench] = float(roc_auc_score(y, s))
        print(f"seed{args.seed} B {bench} {summary[bench]:.4f}", flush=True)

    foil = load_foil_val(args.foil_eval)
    path = outdir / "bank_foil.npz"
    if not path.is_file():
        val_ids = set(load_foil_val(10000)["image_id"].astype(str))
        df = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                         encoding="utf-8-sig")
        df = df[~df["image_id"].astype(str).isin(val_ids)]
        df = df.sample(min(args.bank, len(df)), random_state=args.seed)
        bi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
        bt = enc.encode_texts(df["positive_caption"].astype(str).tolist())
        np.savez_compressed(path, img=bi, txt=bt)
    vi = enc.encode_images(foil["path"].tolist(), batch_size=64, num_workers=4)
    tp = enc.encode_texts(foil["positive"].astype(str).tolist())
    tn = enc.encode_texts(foil["negative"].astype(str).tolist())
    with np.load(path) as z:
        bi, bt = z["img"], z["txt"]
    sp, sn = (vi * tp).sum(1), (vi * tn).sum(1)
    hp = topk_mean(vi, bt) + topk_mean(tp, bi)
    hn = topk_mean(vi, bt) + topk_mean(tn, bi)
    Ap, An = -sp + LAM * hp, -sn + LAM * hn
    np.savez_compressed(outdir / "B_scores_foil.npz", a_pos=Ap, a_neg=An,
                        image_id=foil["image_id"].astype(str).to_numpy())
    summary["foil"] = float(np.mean(An > Ap))
    print(f"seed{args.seed} B foil {summary['foil']:.4f}", flush=True)
    write_json(outdir / "B_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
