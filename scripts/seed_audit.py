"""Three-seed reproducibility audit for plan A (cosine ranking + anchor).

For one seed: train the LoRA adapter -> save it -> rebuild the four reference
banks with THAT adapter -> score the four validation splits and save per-sample
predictions (sample_id, image_id, label, cosine, r_I, r_c, final score).

Everything is resumable: existing artifacts are skipped.
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
    ap.add_argument("--bank", type=int, default=15000)
    ap.add_argument("--eval-n", type=int, default=0)
    ap.add_argument("--foil-eval", type=int, default=5000)
    args = ap.parse_args()

    outdir = RESULTS_DIR / "seed_audit" / f"seed_{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    enc = JointEncoder(CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device
    for p in model.parameters():
        p.requires_grad = False

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

    def encode(px, ids, mask):
        vo = model.vision_model(pixel_values=px)
        to = model.text_model(input_ids=ids, attention_mask=mask)
        v = F.normalize(model.visual_projection(vo.pooler_output).float(), dim=-1)
        t = F.normalize(model.text_projection(to.pooler_output).float(), dim=-1)
        return v, t

    adapter = outdir / "adapter.pt"
    if not adapter.is_file():
        inject(model)
        data = Data(enc, 800, 400, 200, 200, seed=args.seed)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=1e-4, weight_decay=1e-4)
        rng = np.random.default_rng(args.seed)
        model.train()
        for step in range(1, args.steps + 1):
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
                v0, t0p = encode(px, PI, PM)
                _, t0n = encode(px, NI, NM)
            v, tp = encode(px, PI, PM)
            _, tn = encode(px, NI, NM)
            sp, sn = (v * tp).sum(1), (v * tn).sum(1)
            loss = F.softplus(0.1 - (sp - sn)).mean()
            logits = (v @ tp.t()) / 0.07
            tgt = torch.arange(len(v), device=dev)
            loss = loss + 0.5 * (F.cross_entropy(logits, tgt) +
                                 F.cross_entropy(logits.t(), tgt))
            anchor = (1 - (v * v0).sum(1)).mean() + \
                     0.5 * ((1 - (tp * t0p).sum(1)).mean() +
                            (1 - (tn * t0n).sum(1)).mean())
            loss = loss + 5.0 * anchor
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 0:
                print(f"seed{args.seed} step {step} loss {float(loss):.4f}", flush=True)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()
                 if any(f"self_attn.{p}.{n}" in k for p in ("q_proj", "v_proj")
                        for n in ("A", "B"))}
        torch.save(state, adapter)
        print(f"seed{args.seed}: adapter saved ({len(state)} tensors)", flush=True)
    else:
        inject(model)
        model.load_state_dict(torch.load(adapter, map_location="cpu"), strict=False)
        print(f"seed{args.seed}: adapter loaded", flush=True)
    model.to(dev).eval()

    # ---------------- banks (this seed's representation)
    for bench, limit in (("coco", args.bank), ("flickr", args.bank),
                         ("mmimdb", args.bank), ("foil", args.bank)):
        path = outdir / f"bank_{bench}.npz"
        if path.is_file():
            continue
        if bench == "foil":
            val = load_foil_val(10000)
            val_ids = set(val["image_id"].astype(str))
            df = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                             encoding="utf-8-sig")
            df = df[~df["image_id"].astype(str).isin(val_ids)]
            df = df.sample(min(limit, len(df)), random_state=args.seed)
            paths, caps = df["path"].tolist(), df["positive_caption"].astype(str).tolist()
        else:
            df = load_split(bench, "train")
            df = df.sample(min(limit, len(df)), random_state=args.seed)
            paths, caps = df["path"].tolist(), df["caption"].astype(str).tolist()
        img = enc.encode_images(paths, batch_size=64, num_workers=4)
        txt = enc.encode_texts(caps)
        np.savez_compressed(path, img=img, txt=txt)
        print(f"seed{args.seed}: bank {bench} {len(img)} rows", flush=True)

    # ---------------- score val / test-free evaluation
    summary = {}
    for bench in ("coco", "flickr", "mmimdb"):
        df = load_split(bench, "val", args.eval_n or None)
        cos, hid = [], []
        with torch.no_grad():
            for i in range(0, len(df), 32):
                px = torch.stack([proc_img(p) for p in df["path"].iloc[i:i + 32]]).to(dev)
                ids, mask = zip(*[proc_txt(c) for c in df["caption"].iloc[i:i + 32]])
                v, t = encode(px, torch.stack(ids).to(dev), torch.stack(mask).to(dev))
                cos.append((v * t).sum(1).cpu().numpy())
        vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
        vt = enc.encode_texts(df["caption"].tolist())
        with np.load(outdir / f"bank_{bench}.npz") as z:
            bi, bt = z["img"], z["txt"]
        rx = topk_mean(vi, bt); rc = topk_mean(vt, bi)
        cos = (vi * vt).sum(1)
        s = -cos + LAM * (rx + rc)
        y = df["label"].to_numpy()
        np.savez_compressed(outdir / f"val_scores_{bench}.npz",
                            sample_id=np.arange(len(df)),
                            image_id=df["image_id"].astype(str).to_numpy(),
                            label=y, cos=cos, rx=rx, rc=rc, score=s)
        summary[bench] = float(roc_auc_score(y, s))
        print(f"seed{args.seed} {bench} {summary[bench]:.4f}", flush=True)

    foil = load_foil_val(args.foil_eval)
    vi = enc.encode_images(foil["path"].tolist(), batch_size=64, num_workers=4)
    tp = enc.encode_texts(foil["positive"].astype(str).tolist())
    tn = enc.encode_texts(foil["negative"].astype(str).tolist())
    with np.load(outdir / "bank_foil.npz") as z:
        bi, bt = z["img"], z["txt"]
    sp, sn = (vi * tp).sum(1), (vi * tn).sum(1)
    hp = topk_mean(vi, bt) + topk_mean(tp, bi)
    hn = topk_mean(vi, bt) + topk_mean(tn, bi)
    Ap, An = -sp + LAM * hp, -sn + LAM * hn
    np.savez_compressed(outdir / "val_scores_foil.npz",
                        sample_id=np.arange(len(foil)),
                        image_id=foil["image_id"].astype(str).to_numpy(),
                        s_pos=sp, s_neg=sn, r_pos=hp, r_neg=hn,
                        a_pos=Ap, a_neg=An)
    acc = float(np.mean(An > Ap))
    summary["foil"] = acc
    print(f"seed{args.seed} foil {acc:.4f}", flush=True)
    write_json(outdir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
