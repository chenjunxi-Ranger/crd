"""RAT admission gate.

M0: [s, h]
M1: [s, h, S_IT]                 (cross-modal similarity structure only)
M2: [s, h, S_IT, A_I, A_T]       (+ image-internal and text-internal relations)

Delta_rel = AUC(M2) - AUC(M1) must be positive on all three row benchmarks,
with a paired bootstrap interval excluding 0, for RAT to be worth training.

Everything is read from a frozen CLIP forward with output_attentions=True; no
parameter is updated. Attention matrices are summarised by PCA fitted inside
each CV fold (no leakage).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder                     # noqa: E402
from src.protocol import CLIP_DIR, RESULTS_DIR, load_split, write_json  # noqa: E402

NTOK = 12
K = 10


def topk_mean(q, bank, k=K, chunk=256):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bt = torch.as_tensor(bank, device=dev)
    out = np.empty(len(q))
    for i in range(0, len(q), chunk):
        sim = torch.as_tensor(q[i:i + chunk], device=dev) @ bt.T
        out[i:i + chunk] = torch.topk(sim, k, dim=1).values.mean(1).cpu().numpy()
    return out


@torch.no_grad()
def forward_feats(model, proc, dev, paths, caps, batch=32):
    from PIL import Image
    s_all, sit_all, ai_all, at_all = [], [], [], []
    for st0 in range(0, len(paths), batch):
        imgs = []
        for p in paths[st0:st0 + batch]:
            with Image.open(p) as im:
                im2 = im.convert("RGB")
                if im2.width * im2.height > 20_000_000:
                    im2.draft("RGB", (1024, 1024))
                imgs.append(im2.copy())
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
        tx = proc(text=caps[st0:st0 + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=77)
        tx = {k: v.to(dev) for k, v in tx.items()}
        vo = model.vision_model(pixel_values=px, output_attentions=True)
        to = model.text_model(input_ids=tx["input_ids"],
                              attention_mask=tx["attention_mask"],
                              output_attentions=True)
        p_img = F.normalize(model.visual_projection(vo.pooler_output).float(), dim=-1)
        patch = F.normalize(model.visual_projection(vo.last_hidden_state[:, 1:, :]).float(), dim=-1)
        tok = F.normalize(model.text_projection(to.last_hidden_state).float(), dim=-1)
        p_txt = F.normalize(model.text_projection(to.pooler_output).float(), dim=-1)
        s = (p_img * p_txt).sum(1)

        # image-internal: last-layer attention, mean over heads, patch block
        ai = vo.attentions[-1].mean(1)[:, 1:, 1:]          # (B,49,49)
        # text-internal: last-layer attention over the first NTOK content tokens
        at_full = to.attentions[-1].mean(1)                # (B,77,77)
        ids = tx["input_ids"]
        eos = ids.argmax(-1)
        at_b = torch.zeros(len(imgs), NTOK, NTOK, device=dev)
        for b in range(len(imgs)):
            q = min(int(eos[b]) - 1, NTOK)                 # skip SOT at position 0
            at_b[b, :q, :q] = at_full[b, 1:q + 1, 1:q + 1]
        # cross-modal similarity: patch x token (first NTOK content tokens)
        tok_c = tok[:, 1:NTOK + 1, :]                      # (B,NTOK,d)
        sit = torch.einsum("bpd,bqd->bpq", patch, tok_c)   # (B,49,NTOK)
        s_all.append(s.cpu().numpy())
        ai_all.append(ai.cpu().numpy().reshape(len(imgs), -1))
        at_all.append(at_b.cpu().numpy().reshape(len(imgs), -1))
        # compact S_IT summary: per-token max over patches + mean/std
        mx = sit.max(1).values
        sm = sit.mean(1)
        sit_all.append(torch.cat([mx, sm, mx.std(1, keepdim=True)], 1).cpu().numpy())
    return (np.concatenate(s_all), np.concatenate(sit_all),
            np.concatenate(ai_all), np.concatenate(at_all))


def cv_auc(cols, y, n_pca=16):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    vals = []
    for tr, te in skf.split(cols[0], y):
        parts_tr, parts_te = [], []
        for c in cols:
            if c.shape[1] > 64:                            # matrix -> PCA inside fold
                p = PCA(n_components=min(n_pca, len(tr) - 1, c.shape[1]), random_state=0)
                parts_tr.append(p.fit_transform(c[tr]))
                parts_te.append(p.transform(c[te]))
            else:
                parts_tr.append(c[tr])
                parts_te.append(c[te])
        xtr = np.column_stack(parts_tr)
        xte = np.column_stack(parts_te)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        m.fit(xtr, y[tr])
        vals.append(roc_auc_score(y[te], m.predict_proba(xte)[:, 1]))
    return float(np.mean(vals)), np.asarray(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--limit", type=int, default=800)
    args = ap.parse_args()

    enc = JointEncoder(CLIP_DIR)
    df = load_split(args.bench, "val", args.limit)
    s, sit, ai, at = forward_feats(enc.model, enc.processor, enc.device,
                                   df["path"].tolist(), df["caption"].tolist())
    with np.load(RESULTS_DIR / "gate" / f"bank_{args.bench}_all.npz") as z:
        bi, bt = z["img"], z["txt"]
    vi = enc.encode_images(df["path"].tolist(), batch_size=64, num_workers=4)
    vt = enc.encode_texts(df["caption"].tolist())
    h = topk_mean(vi, bt) + topk_mean(vt, bi)
    y = df["label"].to_numpy()

    base = np.column_stack([s, h])
    a0, v0 = cv_auc([base], y)
    a1, v1 = cv_auc([base, sit], y)
    a2, v2 = cv_auc([base, sit, ai, at], y)
    d = v2 - v1
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]

    out = {"bench": args.bench, "n": int(len(df)),
           "auc_M0": a0, "auc_M1": a1, "auc_M2": a2,
           "delta_rel_M2_minus_M1": float(a2 - a1),
           "delta_M1_minus_M0": float(a1 - a0),
           "delta_ci95": [float(np.percentile(boot, 2.5)),
                          float(np.percentile(boot, 97.5))]}
    write_json(RESULTS_DIR / "gate" / f"rat_{args.bench}.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
