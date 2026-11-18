"""Stage 1 for the revised C3: LoRA on the last two blocks of one frozen CLIP.

M0  frozen CLIP                       (baseline)
M1  LoRA + unified ranking loss       (cross-image hard negatives + FOIL counterfactuals)
M2  M1 + teacher-anchor loss          (keeps the adapted space close to the frozen one)

Only q_proj / v_proj of blocks 10-11 are trainable in both towers (rank 4).
No second encoder, no external corpus, no dataset id.

Reports row-level AUROC on COCO / Flickr / MM-IMDb and within-pair accuracy on
FOIL, before and after adaptation, on identical rows.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import (CLIP_DIR, PROJECT, RESULTS_DIR, load_foil_val,  # noqa: E402
                          load_split, write_json)

BLOCKS = (10, 11)
RANK, ALPHA = 4, 8.0


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = RANK, alpha: float = ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scaling


def inject(model):
    n = 0
    for tower in (model.vision_model, model.text_model):
        for li in BLOCKS:
            attn = tower.encoder.layers[li].self_attn
            dev = attn.q_proj.weight.device
            attn.q_proj = LoRALinear(attn.q_proj).to(dev)
            attn.v_proj = LoRALinear(attn.v_proj).to(dev)
            n += 2
    return n


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


class Data:
    def __init__(self, enc: JointEncoder, n_foil: int, n_coco: int, n_flickr: int,
                 n_mmimdb: int, seed: int = 0):
        self.rows = []                                   # (path, pos, neg)
        val = load_foil_val(10000)
        val_ids = set(val["image_id"].astype(str))
        foil = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                           encoding="utf-8-sig")
        foil = foil[~foil["image_id"].astype(str).isin(val_ids)]
        foil = foil.sample(min(n_foil, len(foil)), random_state=seed)
        for p, a, b in zip(foil["path"], foil["positive_caption"],
                           foil["negative_caption"]):
            self.rows.append((str(p), str(a), str(b)))

        for bench, n in (("coco", n_coco), ("flickr", n_flickr), ("mmimdb", n_mmimdb)):
            d = load_split(bench, "train").sample(n, random_state=seed)
            caps = d["caption"].astype(str).tolist()
            paths = d["path"].tolist()
            emb = enc.encode_texts(caps)                 # frozen embeddings for hard mining
            sim = emb @ emb.T
            np.fill_diagonal(sim, -1e9)
            hard = sim.argmax(1)                         # hardest caption of another row
            for i, p in enumerate(paths):
                self.rows.append((p, caps[i], caps[hard[i]]))
        np.random.default_rng(seed).shuffle(self.rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-foil", type=int, default=800)
    ap.add_argument("--n-coco", type=int, default=400)
    ap.add_argument("--n-flickr", type=int, default=200)
    ap.add_argument("--n-mmimdb", type=int, default=200)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--eval-n", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=0.5)
    ap.add_argument("--variant", default="M2", choices=["M1", "M2"])
    ap.add_argument("--nce", type=float, default=0.0,
                    help="weight of the in-batch contrastive term (0 disables)")
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    model, proc, dev = enc.model, enc.processor, enc.device
    for p in model.parameters():
        p.requires_grad = False

    data = Data(enc, args.n_foil, args.n_coco, args.n_flickr, args.n_mmimdb)
    print(f"train pairs: {len(data.rows)}", flush=True)

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

    # ---------------- baseline (M0) metrics on identical subsets
    def evaluate(tag: str):
        res = {}
        for bench in ("coco", "flickr", "mmimdb"):
            d = load_split(bench, "val", args.eval_n)
            scores, labels = [], []
            with torch.no_grad():
                for i in range(0, len(d), 32):
                    px = torch.stack([proc_img(p) for p in
                                      d["path"].iloc[i:i + 32]]).to(dev)
                    ids, mask = zip(*[proc_txt(c) for c in
                                      d["caption"].iloc[i:i + 32]])
                    v, t = encode(px, torch.stack(ids).to(dev), torch.stack(mask).to(dev))
                    scores.append((v * t).sum(1).cpu().numpy())
                    labels.append(d["label"].iloc[i:i + 32].to_numpy())
            from sklearn.metrics import roc_auc_score
            res[bench] = float(roc_auc_score(np.concatenate(labels),
                                             -np.concatenate(scores)))
        foil = load_foil_val(args.eval_n)
        correct = []
        with torch.no_grad():
            for i in range(0, len(foil), 24):
                px = torch.stack([proc_img(p) for p in
                                  foil["path"].iloc[i:i + 24]]).to(dev)
                for col, bucket in (("positive", "pos"), ("negative", "neg")):
                    ids, mask = zip(*[proc_txt(c) for c in foil[col].iloc[i:i + 24]])
                    v, t = encode(px, torch.stack(ids).to(dev), torch.stack(mask).to(dev))
                    correct.append(((v * t).sum(1).cpu().numpy(), bucket, i))
        pos = np.concatenate([c[0] for c in correct if c[1] == "pos"])
        neg = np.concatenate([c[0] for c in correct if c[1] == "neg"])
        res["foil"] = float(np.mean(np.where(neg < pos, 1.0,
                                             np.where(neg > pos, 0.0, 0.5))))
        write_json(RESULTS_DIR / "gate" / f"lora_{tag}.json", res)
        print(tag, {k: round(v, 4) for k, v in res.items()}, flush=True)
        return res

    m0 = evaluate("M0_before")

    # ---------------- LoRA training
    n_lora = inject(model)
    params = trainable(model)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    print(f"LoRA modules: {n_lora}, trainable tensors: {len(params)}", flush=True)

    rng = np.random.default_rng(0)
    step = 0
    model.train()
    while step < args.steps:
        idx = rng.choice(len(data.rows), args.batch, replace=False)
        pxs, pos_ids, pos_m, neg_ids, neg_m = [], [], [], [], []
        for k in idx:
            path, a, b = data.rows[k]
            pxs.append(proc_img(path))
            i1, m1 = proc_txt(a)
            i2, m2 = proc_txt(b)
            pos_ids.append(i1); pos_m.append(m1)
            neg_ids.append(i2); neg_m.append(m2)
        px = torch.stack(pxs).to(dev)
        pi = torch.stack(pos_ids).to(dev); pm = torch.stack(pos_m).to(dev)
        ni = torch.stack(neg_ids).to(dev); nm = torch.stack(neg_m).to(dev)
        with torch.no_grad():
            v0, t0p = encode(px, pi, pm)
            _, t0n = encode(px, ni, nm)
        v, tp = encode(px, pi, pm)
        _, tn = encode(px, ni, nm)
        sp = (v * tp).sum(1)
        sn = (v * tn).sum(1)
        loss = F.softplus(0.1 - (sp - sn)).mean()
        if args.nce > 0:
            # in-batch contrastive term: keeps the row-level (cross-image) ordering
            logits = (v @ tp.t()) / 0.07
            tgt = torch.arange(len(v), device=dev)
            loss = loss + args.nce * 0.5 * (
                F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
        if args.variant == "M2":
            anchor = (1 - (v * v0).sum(1)).mean() + \
                     0.5 * ((1 - (tp * t0p).sum(1)).mean() + (1 - (tn * t0n).sum(1)).mean())
            loss = loss + args.anchor * anchor
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 200 == 0:
            print(f"step {step} loss {float(loss):.4f}", flush=True)

    model.eval()
    # persist only the LoRA weights so the adapted encoder can be rebuilt later
    lora_state = {k: v.detach().cpu() for k, v in model.state_dict().items()
                  if any(f"layers.{li}.self_attn.{proj}.{n}" in k
                         for li in BLOCKS for proj in ("q_proj", "v_proj")
                         for n in ("A", "B"))}
    torch.save({"state": lora_state,
                "config": {"blocks": list(BLOCKS), "rank": RANK, "alpha": ALPHA,
                           "n_train": len(data.rows), "steps": args.steps,
                           "lr": args.lr, "anchor": args.anchor, "nce": args.nce,
                           "variant": args.variant}},
               RESULTS_DIR / "gate" / "lora_weights.pt")
    print(f"saved {len(lora_state)} LoRA tensors", flush=True)
    m1 = evaluate(f"{args.variant}_after")
    write_json(RESULTS_DIR / "gate" / "lora_stage1.json",
               {"variant": args.variant, "n_train": len(data.rows),
                "before": m0, "after": m1,
                "delta": {k: m1[k] - m0[k] for k in m0}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
