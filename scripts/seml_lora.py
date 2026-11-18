"""SEML-LoRA: semantic-environment meta-LoRA.

Environments = KMeans over frozen CLIP text embeddings of the official train split.
Each environment holds both clean and noisy rows; pairs are cosine-quantile matched.

    inner : phi_e = phi - eta * grad L_e(phi)          (first order)
    outer : minimize  mean_{e != e'} L_e'(phi_e)

M1 (control) = the same LoRA + same data + same steps with plain ERM ranking.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lora_stage1 import inject                        # noqa: E402
from src.encoders import JointEncoder                 # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json   # noqa: E402


def lora_params(model):
    return [p for k, p in model.named_parameters()
            if p.requires_grad and ("lora" in k.lower() or k.endswith(("A", "B")))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco", choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--mode", default="seml", choices=["seml", "erm"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eta", type=float, default=5e-4)
    ap.add_argument("--anchor", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match", default="quantile", choices=["quantile", "none"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--sample", default="env", choices=["env", "global"])
    ap.add_argument("--eval-foil", type=int, default=0)
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--save-adapter", default="")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import lora_stage1 as _lora
    mdir = args.model_dir or CLIP_DIR
    enc = JointEncoder(mdir)
    model, proc, dev = enc.model, enc.processor, enc.device
    for p in model.parameters():
        p.requires_grad = False
    def inject_last2(md):
        """Last two blocks of EACH tower (text and vision depths can differ)."""
        n = 0
        for tower in (md.vision_model, md.text_model):
            L = len(tower.encoder.layers)
            for li in range(L - 2, L):
                attn = tower.encoder.layers[li].self_attn
                dv = attn.q_proj.weight.device
                attn.q_proj = _lora.LoRALinear(attn.q_proj).to(dv)
                attn.v_proj = _lora.LoRALinear(attn.v_proj).to(dv)
                n += 2
        return n

    nv = len(model.vision_model.encoder.layers)
    nt = len(model.text_model.encoder.layers)
    print(f"model={mdir} vision={nv} text={nt}", flush=True)
    inject_last2(model)
    model.to(dev)

    tr = load_split(args.bench, "train", args.train).reset_index(drop=True)
    # environments from frozen text embeddings
    with torch.no_grad():
        vt = enc.encode_texts(tr["caption"].astype(str).tolist())
    km = KMeans(n_clusters=args.K, n_init=10, random_state=0).fit(vt)
    tr["env"] = km.labels_
    print("env sizes", np.bincount(tr["env"]).tolist(), flush=True)

    # frozen cosine for every train row (used only to match pairs by difficulty)
    with torch.no_grad():
        vi0 = enc.encode_images(tr["path"].tolist(), batch_size=64, num_workers=4)
    tr["cos0"] = (vi0 * vt).sum(1)

    def batch_from(env, n):
        # the ONLY difference between arms: env-stratified vs global sampling
        sub = tr if args.sample == "global" else tr[tr.env == env]
        cl = sub[sub.label == 0]; no = sub[sub.label == 1]
        k = min(len(cl), len(no), n)
        if k < 4:
            return None
        ci = cl.sample(k, random_state=None).index.to_numpy()
        ni = no.sample(k, random_state=None).index.to_numpy()
        return tr.loc[ci], tr.loc[ni]

    def prep(df):
        ims, caps = df["path"].tolist(), df["caption"].astype(str).tolist()
        px = proc(images=[__import__("PIL.Image", fromlist=["x"]).open(p).convert("RGB") for p in ims],
                  return_tensors="pt")["pixel_values"].to(dev)
        tk = proc(text=caps, return_tensors="pt", padding=True, truncation=True,
                  max_length=77).to(dev)
        return px, tk

    def s_of(px, tk):
        vo = model.vision_model(pixel_values=px)
        to = model.text_model(**tk)
        v = F.normalize(model.visual_projection(vo.pooler_output).float(), dim=-1)
        t = F.normalize(model.text_projection(to.pooler_output).float(), dim=-1)
        return (v * t).sum(1), v, t

    def loss_env(env, ref=None):
        b = batch_from(env, args.batch)
        if b is None:
            return None
        cl, no = b
        fc = cl["cos0"].to_numpy(); fn = no["cos0"].to_numpy()
        px, tk = prep(cl)
        sp, v1, t1 = s_of(px, tk)
        px2, tk2 = prep(no)
        sn, v2, t2 = s_of(px2, tk2)
        if args.match == "quantile":
            # order-statistics matching on the FROZEN cosine
            oc = np.argsort(fc); on = np.argsort(fn)
            k = min(len(oc), len(on))
            sp = sp[torch.tensor(oc[:k], device=dev)]
            sn = sn[torch.tensor(on[:k], device=dev)]
            loss = F.softplus(args.margin - sp + sn).mean()
        else:
            loss = F.softplus(args.margin - sp.mean() + sn.mean())
        if ref is not None and args.anchor > 0:
            v0, t0 = ref
            anchor = (1 - (v1 * v0).sum(1)).mean() + (1 - (t1 * t0).sum(1)).mean()
            loss = loss + args.anchor * anchor
        return loss

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    model.train()

    for step in range(args.steps):
        e1, e2 = np.random.choice(args.K, 2, replace=False)
        if args.mode == "seml":
            # reference features for the anchor (frozen)
            b = batch_from(e1, args.batch)
            if b is None:
                continue
            px, tk = prep(b[0])
            with torch.no_grad():
                _, v0, t0 = s_of(px, tk)
            L1 = loss_env(e1, ref=(v0, t0))
            if L1 is None:
                continue
            g1 = torch.autograd.grad(L1, params, create_graph=False, allow_unused=True)
            g1 = [torch.zeros_like(p) if g is None else g for g, p in zip(g1, params)]
            backup = [p.detach().clone() for p in params]
            with torch.no_grad():
                for p, g in zip(params, g1):
                    p -= args.eta * g
            L2 = loss_env(e2)                    # meta objective on the *other* environment
            if L2 is None:
                with torch.no_grad():
                    for p, s in zip(params, backup):
                        p.copy_(s)
                continue
            g2 = torch.autograd.grad(L2, params, allow_unused=True)
            g2 = [torch.zeros_like(p) if g is None else g for g, p in zip(g2, params)]
            with torch.no_grad():               # restore then apply meta gradient
                for p, s, g in zip(params, backup, g2):
                    p.copy_(s - args.lr * g)
        else:
            L = loss_env(e1)
            if L is None:
                continue
            opt.zero_grad(); L.backward(); opt.step()
        if (step + 1) % 20 == 0:
            print(f"  step {step+1} loss {float(L2 if args.mode=='seml' else L):.4f}", flush=True)

    if args.save_adapter:
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
              if k.endswith((".A", ".B")) and ("q_proj" in k or "v_proj" in k)}
        torch.save(sd, args.save_adapter)
        print(f"saved adapter {len(sd)} tensors -> {args.save_adapter}", flush=True)
        if args.no_eval:
            return 0

    # ---- evaluation: adapted bank + CSLS
    model.eval()
    print("eval", flush=True)

    @torch.no_grad()
    def enc_i(paths, bs=64):
        out = []
        for i in range(0, len(paths), bs):
            px = proc(images=[__import__("PIL.Image", fromlist=["x"]).open(p).convert("RGB")
                              for p in paths[i:i + bs]], return_tensors="pt")["pixel_values"].to(dev)
            vo = model.vision_model(pixel_values=px)
            out.append(F.normalize(model.visual_projection(vo.pooler_output).float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def enc_t(caps, bs=128):
        out = []
        for i in range(0, len(caps), bs):
            tk = proc(text=list(caps[i:i + bs]), return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(dev)
            to = model.text_model(**tk)
            out.append(F.normalize(model.text_projection(to.pooler_output).float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    bank = load_split(args.bench, "train", 15000)
    vb, tb = enc_i(bank["path"].tolist()), enc_t(bank["caption"].astype(str).tolist())
    va = load_split(args.bench, "val")
    vi, vt2 = enc_i(va["path"].tolist()), enc_t(va["caption"].astype(str).tolist())
    cosv = (vi * vt2).sum(1)
    k = 10
    h = np.partition(vi @ tb.T, -k, axis=1)[:, -k:].mean(1) + \
        np.partition(vt2 @ vb.T, -k, axis=1)[:, -k:].mean(1)
    from sklearn.metrics import roc_auc_score
    yv = va["label"].to_numpy()
    res = {"auc_cos": float(roc_auc_score(yv, -cosv)),
           "auc_csls": float(roc_auc_score(yv, -cosv + 0.8 * h))}
    if args.eval_foil:
        from src.protocol import load_foil_val
        fv = load_foil_val(args.eval_foil)
        fi = enc_i(fv["path"].tolist())
        fp = enc_t(fv["positive"].astype(str).tolist())
        fn_ = enc_t(fv["negative"].astype(str).tolist())
        sp, sn = (fi * fp).sum(1), (fi * fn_).sum(1)
        acc_cos = float(np.mean(sn < sp))
        import pandas as pd
        from src.protocol import PROJECT
        trf = pd.read_csv(PROJECT / "数据集" / "FOIL-COCO" / "pairs_train.csv",
                          encoding="utf-8-sig").sample(15000, random_state=0)
        fbi = enc_i(trf["path"].tolist())
        fbt = enc_t(trf["positive_caption"].astype(str).tolist())
        k = 10
        h_img = np.partition(fi @ fbt.T, -k, axis=1)[:, -k:].mean(1)      # cancels within a pair
        h_pos = np.partition(fp @ fbi.T, -k, axis=1)[:, -k:].mean(1)
        h_neg = np.partition(fn_ @ fbi.T, -k, axis=1)[:, -k:].mean(1)
        # noisy candidate must score HIGHER (more anomalous)
        acc_csls = float(np.mean((-sn + 0.8 * (h_img + h_neg)) >
                                 (-sp + 0.8 * (h_img + h_pos))))
        res["foil_acc_cos"] = acc_cos
        res["foil_acc_csls"] = acc_csls
        res["foil_n"] = int(len(fv))

    out = {"bench": args.bench, "mode": args.mode, "match": args.match,
           "sample": args.sample, "seed": args.seed, "K": args.K,
           "steps": args.steps, "result": res}
    print(out, flush=True)
    tag = f"_{args.tag}" if args.tag else ""
    write_json(RESULTS_DIR / "gate" /
               f"eslora_{args.sample}_{args.match}_s{args.seed}{tag}_{args.bench}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
