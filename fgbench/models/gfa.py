"""GFA: Grounded Fact Alignment (v1).

Frozen CLIP B/16. Per (image, caption):
    patches P (196x512)  <- visual_projection(last_hidden_state[:,1:])
    tokens  H (L x512)   <- text_projection(last_hidden_state)
    A = softmax_j( (H P^T)/tau )                # token -> patch attention
    v_fact = A^T H  (196x512) -> pooled by A    # fact-grounded visual evidence
    q = MLP([pooled_fact, pooled_text])         # fact support logit
    s = alpha * cos_clip + (1-alpha) * (2q-1)

Loss: BCE(pair labels) + lam_g * L1(pred_box, gt_box) on CLEAN rows only
      (boxes come from VG region descriptions).
Inference: single (image, caption); no partner, no reference bank, no boxes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fgbench" / "audit"))

from fgbench.common import BENCH, INTER, jdump  # noqa: E402

VG_ROOT = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\VisualGenome")
VG_INDEX = ROOT / "fgbench" / "data" / "intermediate" / "vg" / "image_index.parquet"
OUT = ROOT / "results" / "fgbench"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 14


class Head(nn.Module):
    def __init__(self, d=512, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.q = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(),
                               nn.Linear(256, 1))
        self.box = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 4))

    def forward(self, P, H, cos):
        att = (H @ P.transpose(0, 1)) / (P.shape[-1] ** 0.5)   # (L,196)
        A = att.softmax(-1)
        pooled = (A.transpose(0, 1) @ H).sum(0)                # (512,)
        tpool = H.mean(0)
        q = self.q(torch.cat([pooled, tpool])).squeeze(-1)
        box = torch.sigmoid(self.box(pooled))
        s = self.alpha * cos + (1 - self.alpha) * (2 * torch.sigmoid(q) - 1)
        return s, box, A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--cycles", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-g", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--variant", default="gfa", choices=["gfa", "m2b16"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from src.encoders import JointEncoder
    from src.protocol import CLIP_DIR
    from PIL import Image

    enc = JointEncoder(args.model_dir or CLIP_DIR)
    model, proc = enc.model, enc.processor
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    vgidx = pd.read_parquet(VG_INDEX).set_index("image_id").rel_path.to_dict()
    gp = pd.read_parquet(INTER / "vg_grounding_pairs.parquet")
    box_of = {(r.target_image_id, r.caption): (r.x, r.y, r.w, r.h) for r in gp.itertuples()}

    d = pd.read_parquet(BENCH / "train.parquet")
    d = d[(d.dataset == "VG") & (d.facet == "Attribute")]
    cids = d.cycle_id.unique(); rng.shuffle(cids)
    d = d[d.cycle_id.isin(cids[: args.cycles])]
    d = d[d.groupby("cycle_id").target_image_id.transform("nunique") == 2]
    rows = d.reset_index(drop=True)
    print(f"train rows {len(rows)} cycles {rows.cycle_id.nunique()}", flush=True)

    def img_path(tid):
        return str(VG_ROOT / vgidx[int(str(tid).split(":")[1])])

    @torch.no_grad()
    def feats(paths, caps):
        px = proc(images=[Image.open(p).convert("RGB") for p in paths],
                  return_tensors="pt")["pixel_values"].to(DEV)
        vo = model.vision_model(pixel_values=px)
        P = F.normalize(model.visual_projection(vo.last_hidden_state[:, 1:, :].float()), dim=-1)
        u = F.normalize(model.visual_projection(vo.pooler_output.float()), dim=-1)
        tk = proc(text=caps, return_tensors="pt", padding=True, truncation=True,
                  max_length=77).to(DEV)
        to = model.text_model(**tk)
        H = F.normalize(model.text_projection(to.last_hidden_state.float()), dim=-1)
        v = F.normalize(model.text_projection(to.pooler_output.float()), dim=-1)
        ids = tk["input_ids"]
        eos = model.config.text_config.eos_token_id
        return P, u, H, v, ids, eos

    head = Head(alpha=args.alpha if args.variant == "gfa" else 1.0).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    idxs = np.arange(len(rows))
    for step in range(args.steps):
        pick = rng.choice(idxs, args.batch, replace=False)
        sub = rows.iloc[pick]
        P, u, H, v, ids, eos = feats([img_path(t) for t in sub.target_image_id],
                                     sub.caption.astype(str).tolist())
        cos = (u * v).sum(1)
        loss = torch.zeros((), device=DEV)
        for b in range(len(sub)):
            pos = int((ids[b] == eos).nonzero()[0]) if (ids[b] == eos).any() else H.shape[1] - 1
            Hb = H[b, 1:max(2, pos)]
            s, box, _ = head(P[b], Hb, cos[b])
            y = torch.tensor(1.0 - float(sub.pair_label.iat[b]), device=DEV)
            loss = loss + F.binary_cross_entropy_with_logits(s, y)
            if args.variant == "gfa":
                key = (sub.target_image_id.iat[b], str(sub.caption.iat[b]))
                gt = box_of.get(key)
                if gt is not None and sub.pair_label.iat[b] == 0:
                    w = sub_img_wh(sub.target_image_id.iat[b], vgidx)
                    if w is not None:
                        gx, gy, gw, gh = gt
                        tgt = torch.tensor([(gx + gw / 2) / w[0], (gy + gh / 2) / w[1],
                                            gw / w[0], gh / w[1]], device=DEV)
                        loss = loss + args.lam_g * F.l1_loss(box, tgt)
        loss = loss / len(sub)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1} loss {float(loss):.4f}", flush=True)

    # ---- eval on Attribute / Relation / Quantity test ----
    te = pd.read_parquet(BENCH / "test.parquet")
    res = []
    with torch.no_grad():
        for facet in ("Attribute", "Relation"):
            sub = te[(te.dataset == "VG") & (te.facet == facet)].reset_index(drop=True)
            if len(sub) == 0:
                continue
            scores = []
            for i in range(0, len(sub), 64):
                ch = sub.iloc[i:i + 64]
                P, u, H, v, ids, eos = feats([img_path(t) for t in ch.target_image_id],
                                             ch.caption.astype(str).tolist())
                cos = (u * v).sum(1)
                for b in range(len(ch)):
                    pos = int((ids[b] == eos).nonzero()[0]) if (ids[b] == eos).any() else H.shape[1] - 1
                    s, _, _ = head(P[b], H[b, 1:max(2, pos)], cos[b])
                    scores.append(float(s))
            sub["s"] = -np.array(scores)
            from sklearn.metrics import roc_auc_score
            y = sub.pair_label.to_numpy()
            auc = float(roc_auc_score(y, sub.s))
            accs = []
            for cid, g in sub.groupby("cycle_id"):
                if len(g) != 4:
                    continue
                accs.append(g[g.pair_label == 0].s.sum() < g[g.pair_label == 1].s.sum())
            res.append({"facet": facet, "variant": args.variant, "n_noisy": int((y == 1).sum()),
                        "auc": auc, "acc2afc": float(np.mean(accs)) if accs else None})
            print(f"{facet} AUC {auc:.4f} 2AFC {res[-1]['acc2afc']}", flush=True)

    out = pd.DataFrame(res)
    p = OUT / f"gfa_{args.variant}.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    jdump({"variant": args.variant, "steps": args.steps, "cycles": args.cycles,
           "lam_g": args.lam_g, "alpha": args.alpha, "rows": res}, p.with_suffix(".json"))
    print(out.to_string(index=False), flush=True)
    return 0


def sub_img_wh(tid, vgidx) -> tuple[float, float] | None:
    """Image width/height from VG image_data (needed to normalise boxes)."""
    global _WH
    try:
        _WH
    except NameError:
        from fgbench.common import INTER as _I
        meta = pd.read_parquet(_I / "vg" / "image_data.parquet")
        _WH = {int(r.image_id): (float(r.width), float(r.height)) for r in meta.itertuples()}
    return _WH.get(int(str(tid).split(":")[1]))


if __name__ == "__main__":
    raise SystemExit(main())
