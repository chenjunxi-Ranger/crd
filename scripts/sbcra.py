"""SBCRA: source-blind correspondence recovery audit.

For each noisy row (target image a, transplanted caption c_b) we ask, using only
the unlabeled candidate bank and the caption:

    rank of the *donor* image b among all bank images by s(., c_b)

donor identity / gold caption are used ONLY for evaluation, never for scoring.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import PROJECT, RESULTS_DIR, write_json   # noqa: E402

DERIVED = PROJECT / "数据集" / "derived"
FILES = {"coco": "mscoco/mscoco__cat__lvl0.40__seed1.csv",
         "flickr": "flickr30k/flickr30k__noun__lvl0.40__seed1.csv"}


def donor_path(p: str, f: str) -> str:
    parent = Path(p).parent
    if "train2014" in str(f) and parent.name != "train2014":
        parent = parent.parent / "train2014"
    return str(parent / str(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco", choices=["coco", "flickr"])
    ap.add_argument("--backbone", default="clip", choices=["clip", "fg"])
    ap.add_argument("--bank-size", type=int, default=8000)
    args = ap.parse_args()

    d = pd.read_csv(DERIVED / FILES[args.bench], encoding="utf-8-sig")
    va = d[d.split == "val"]
    noisy = d[(d.split == "val") & (d.is_mislabel == 1)].dropna(subset=["donor_filename"])
    noisy = noisy.copy()
    noisy["donor_path"] = [donor_path(p, f) for p, f in zip(noisy["path"], noisy["donor_filename"])]
    noisy = noisy[[Path(p).is_file() for p in noisy["donor_path"]]]

    # candidate bank = all val images + every donor image (unlabeled for the scorer)
    bank_paths = list(dict.fromkeys(list(va["path"].astype(str)) + list(noisy["donor_path"])))
    if len(bank_paths) > args.bank_size:
        rng = np.random.default_rng(0)
        keep = set(noisy["donor_path"]) | set(noisy["path"].astype(str))
        others = [p for p in bank_paths if p not in keep]
        extra = list(rng.choice(others, max(0, args.bank_size - len(keep)), replace=False))
        bank_paths = list(dict.fromkeys(list(keep) + extra))

    if args.backbone == "clip":
        from src.encoders import JointEncoder
        from src.protocol import CLIP_DIR
        enc = JointEncoder(CLIP_DIR)
        ei = lambda ps: enc.encode_images(ps, batch_size=64, num_workers=4)
        et = lambda cs: enc.encode_texts(cs)
    else:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from transformers import AutoModelForCausalLM, CLIPImageProcessor, CLIPTokenizerFast
        p = r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\fg-clip-base"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True,
                                                 local_files_only=True).to(dev).eval()
        ip = CLIPImageProcessor.from_pretrained(p, local_files_only=True)
        tk = CLIPTokenizerFast.from_pretrained(p, local_files_only=True)

        @torch.no_grad()
        def ei(paths, bs=48):
            o = []
            for i in range(0, len(paths), bs):
                ims = []
                for q in paths[i:i + bs]:
                    with Image.open(q) as im:
                        im = im.convert("RGB")
                        if im.width * im.height > 20_000_000:
                            im.draft("RGB", (1024, 1024))
                        ims.append(im.copy())
                px = ip(images=ims, return_tensors="pt")["pixel_values"].to(dev)
                o.append(F.normalize(m.get_image_features(pixel_values=px).float(), dim=-1).cpu().numpy())
            return np.concatenate(o)

        @torch.no_grad()
        def et(caps, bs=128):
            o = []
            for i in range(0, len(caps), bs):
                t = {k: v.to(dev) for k, v in tk(list(caps[i:i + bs]), return_tensors="pt",
                                                 padding=True, truncation=True, max_length=77).items()}
                o.append(F.normalize(m.get_text_features(**t).float(), dim=-1).cpu().numpy())
            return np.concatenate(o)

    print(f"bank {len(bank_paths)} images", flush=True)
    B = ei(bank_paths)
    idx = {p: k for k, p in enumerate(bank_paths)}
    T = et(noisy["donor_sentence"].astype(str).tolist())

    S = T @ B.T                                   # (n_query, n_bank)
    donor_rank, target_rank, margin = [], [], []
    for i, (tp, dp) in enumerate(zip(noisy["path"].astype(str), noisy["donor_path"])):
        row = S[i]
        order = np.argsort(-row)
        rank_of = np.empty(len(row), dtype=np.int32)
        rank_of[order] = np.arange(len(row))
        dr = int(rank_of[idx[dp]]) + 1
        tr = int(rank_of[idx[tp]]) + 1
        donor_rank.append(dr); target_rank.append(tr)
        margin.append(float(row[idx[tp]] - row[idx[dp]]))

    noisy = noisy.assign(donor_rank=donor_rank, target_rank=target_rank, margin=margin)
    noisy["image_id"] = noisy["image_id"].astype(str)
    g = pd.read_csv(RESULTS_DIR / "audit" / f"{args.bench}_all_groups.csv", encoding="utf-8-sig")
    g["image_id"] = g["image_id"].astype(str)
    noisy = noisy.merge(g[["image_id", "group"]], on="image_id", how="left")
    noisy["group"] = noisy["group"].fillna("unknown")

    def stats(sub):
        if len(sub) < 10:
            return {"n": int(len(sub))}
        r = sub["donor_rank"].to_numpy()
        return {"n": int(len(r)),
                "recall_at1": float((r <= 1).mean()),
                "recall_at5": float((r <= 5).mean()),
                "mrr": float((1.0 / r).mean()),
                "median_rank": float(np.median(r)),
                "median_target_rank": float(np.median(sub["target_rank"])),
                "mean_margin": float(sub["margin"].mean())}

    out = {"bench": args.bench, "backbone": args.backbone, "bank": len(bank_paths),
           "n_noisy": int(len(noisy)),
           "coverage": float(len(noisy) / max(1, len(d[(d.split == 'val') & (d.is_mislabel == 1)]))),
           "all": stats(noisy),
           "hard": stats(noisy[noisy.group == "persistent_hard"]),
           "easy": stats(noisy[noisy.group == "both_low_burden"])}
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"sbcra_{args.backbone}_{args.bench}.json", out)
    noisy[["image_id", "donor_image_id", "donor_rank", "target_rank", "margin", "group"]].to_csv(
        RESULTS_DIR / "gate" / f"sbcra_{args.backbone}_{args.bench}.csv", index=False,
        encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
