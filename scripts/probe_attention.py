"""Stage-3 iteration 2 probe: attention-weighted support.

Addresses reviewer R3 ("the family ceiling may be an implementation artefact") by
adding the encoders' own CLS attention:
* text CLS attention over tokens  -> which words the sentence encoder relies on
* vision CLS attention over patches -> which regions the image encoder relies on

New statistics (all forward-only, no bank, no labels):
* coverage_attn  : attention-weighted soft coverage
* support_attn   : attention-weighted max patch support
* salient_patch  : support restricted to the top-m attended patches

Compared against cosine and the frozen coverage on COCO/FOIL val subsets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoders import JointEncoder, _identity, _PathDataset   # noqa: E402
from src.features import STOPWORDS, content_mask, pair_features   # noqa: E402
from src.protocol import CLIP_DIR, load_foil_val, load_split      # noqa: E402


@torch.no_grad()
def encode_att(enc: JointEncoder, paths, captions, batch_size=32, workers=4):
    loader = DataLoader(_PathDataset(paths), batch_size=batch_size,
                        num_workers=workers, collate_fn=_identity, shuffle=False)
    tok_all, mask_all, patch_all, pimg_all, ptxt_all = [], [], [], [], []
    at_all, av_all, ids_all, wid_all = [], [], [], []
    cursor = 0
    for pil in loader:
        caps = list(captions[cursor:cursor + len(pil)])
        cursor += len(pil)
        images = enc.processor(images=list(pil), return_tensors="pt")
        images = {k: v.to(enc.device) for k, v in images.items()}
        text = enc.processor(text=caps, return_tensors="pt", padding=True,
                             truncation=True, max_length=77)
        text = {k: v.to(enc.device) for k, v in text.items()}

        vision = enc.model.vision_model(pixel_values=images["pixel_values"],
                                        output_attentions=True)
        patch = F.normalize(enc.model.visual_projection(
            vision.last_hidden_state[:, 1:, :]).float(), dim=-1)
        # last-layer CLS attention over patches, averaged over heads
        av = vision.attentions[-1][:, :, 0, 1:].mean(dim=1)
        av = av / av.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        t_out = enc.model.text_model(input_ids=text["input_ids"],
                                     attention_mask=text["attention_mask"],
                                     output_attentions=True)
        tok = F.normalize(enc.model.text_projection(t_out.last_hidden_state).float(), dim=-1)
        at = t_out.attentions[-1][:, :, 0, :].mean(dim=1)

        pimg = F.normalize(enc.model.visual_projection(vision.pooler_output).float(), dim=-1)
        ptxt = F.normalize(enc.model.text_projection(t_out.pooler_output).float(), dim=-1)

        mask = text["attention_mask"].bool().clone()
        for sid in enc.special_ids:
            mask &= ~(text["input_ids"] == sid)
        b, cur = tok.shape[0], tok.shape[1]
        if cur < 77:
            tok = torch.cat([tok, tok.new_zeros(b, 77 - cur, tok.shape[2])], dim=1)
            mask = torch.cat([mask, mask.new_zeros(b, 77 - cur)], dim=1)
            at = torch.cat([at, at.new_zeros(b, 77 - cur)], dim=1)

        tok_all.append(tok.cpu().numpy())
        mask_all.append(mask.cpu().numpy())
        patch_all.append(patch.cpu().numpy())
        pimg_all.append(pimg.cpu().numpy())
        ptxt_all.append(ptxt.cpu().numpy())
        at_all.append(at.cpu().numpy())
        av_all.append(av.cpu().numpy())
        ids_np = text["input_ids"].cpu().numpy()
        if ids_np.shape[1] < 77:
            ids_np = np.pad(ids_np, ((0, 0), (0, 77 - ids_np.shape[1])))
        ids_all.append(ids_np)
        for i in range(len(caps)):
            wid_all.append(enc.tokenizer(caps[i], truncation=True, max_length=77).word_ids())

    return dict(tok=np.concatenate(tok_all), mask=np.concatenate(mask_all),
                patch=np.concatenate(patch_all), pimg=np.concatenate(pimg_all),
                ptxt=np.concatenate(ptxt_all), at=np.concatenate(at_all),
                av=np.concatenate(av_all), ids=np.concatenate(ids_all), wid=wid_all)


def features(st: dict, tokenizer, tau=0.2, temp=0.05, top_m=10):
    # content mask reuse: build a lightweight namespace with the same fields
    class S:  # noqa: D101
        pass
    ns = S()
    ns.token, ns.token_mask = st["tok"], st["mask"]
    ns.word_ids, ns.input_ids = st["wid"], st["ids"]
    mask = content_mask(ns, tokenizer)

    out = {k: [] for k in ("coverage_attn", "support_attn", "salient_patch", "coverage")}
    for i in range(len(st["tok"])):
        t, p, m = st["tok"][i][mask[i]], st["patch"][i], mask[i]
        s = t @ p.T
        mx, mn = s.max(1), s.mean(1)
        cov = 1 / (1 + np.exp(-(mx - tau) / temp))
        a = st["at"][i][m]
        a = np.clip(a, 0, None)
        a = a / max(a.sum(), 1e-8)
        out["coverage_attn"].append(float((a * cov).sum()))
        out["support_attn"].append(float((a * mx).sum()))
        out["coverage"].append(float(cov.mean()))
        top = np.argsort(-st["av"][i])[:top_m]
        out["salient_patch"].append(float(s[:, top].max(axis=1).mean()))
    return {k: np.asarray(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco", choices=["coco", "foil"])
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()
    enc = JointEncoder(CLIP_DIR)

    if args.bench == "coco":
        df = load_split("coco", "val", args.limit)
        st = encode_att(enc, df["path"].tolist(), df["caption"].tolist())
        f = features(st, enc.tokenizer)
        y = df["label"].to_numpy()
        cos = (st["pimg"] * st["ptxt"]).sum(1)
        rows = {"cosine": roc_auc_score(y, -cos)}
        for k, v in f.items():
            rows[k] = roc_auc_score(y, -v)
            rows[k + "+cos(0.5)"] = roc_auc_score(y, -cos + 0.5 * (1 - v))
        for k, v in rows.items():
            print(f"{k:28s} {v:.4f}")
    else:
        df = load_foil_val(args.limit)
        st = encode_att(enc, df["path"].tolist() * 2,
                        df["positive"].tolist() + df["negative"].tolist())
        f = features(st, enc.tokenizer)
        n = len(df)
        cos = (st["pimg"] * st["ptxt"]).sum(1)

        def acc(d):
            return float(np.mean(np.where(d < 0, 1.0, np.where(d > 0, 0.0, 0.5))))

        rows = {"cosine": acc(cos[n:] - cos[:n])}
        for k, v in f.items():
            rows[k] = acc(v[n:] - v[:n])
            rows[k + "+cos"] = acc((v[n:] - 0.5 * cos[n:]) - (v[:n] - 0.5 * cos[:n]))
        for k, v in rows.items():
            print(f"{k:28s} {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
