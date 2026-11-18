"""Real-donor pairing cycle diagnostic.

For each noisy row we know: target image A, its gold caption c_a, the donor image B
and the transplanted caption c_b.  Build the 2x2:

    cycle = s(A,c_a) + s(B,c_b) - s(A,c_b) - s(B,c_a)

cycle > 0  -> the scorer prefers the true pairing
cycle ~ 0  -> additive image/text effects only (no pairing information)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="coco", choices=["coco", "flickr"])
    ap.add_argument("--backbone", default="clip", choices=["clip", "fg"])
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()

    d = pd.read_csv(DERIVED / FILES[args.bench], encoding="utf-8-sig")
    d = d[(d.split == "val") & (d.is_mislabel == 1)]
    d = d.dropna(subset=["donor_filename"]).sample(min(args.n, len(d)), random_state=0)
    d = d.reset_index(drop=True)

    # donor path: same directory, unless the donor filename says it lives in train2014
    def _donor_path(p: str, f: str) -> str:
        parent = Path(p).parent
        if "train2014" in str(f) and parent.name != "train2014":
            parent = parent.parent / "train2014"
        return str(parent / str(f))

    d["target_path"] = d["path"].astype(str)
    d["donor_path"] = [_donor_path(p, f) for p, f in zip(d["target_path"], d["donor_filename"])]
    exist = [Path(p).is_file() for p in d["donor_path"]]
    d = d[exist].reset_index(drop=True)

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

    va = ei(d["target_path"].tolist()); vb = ei(d["donor_path"].tolist())
    ca = et(d["gold_sentence"].astype(str).tolist())
    cb = et(d["donor_sentence"].astype(str).tolist())

    s_aa = (va * ca).sum(1)
    s_bb = (vb * cb).sum(1)
    s_ab = (va * cb).sum(1)
    s_ba = (vb * ca).sum(1)
    cyc = s_aa + s_bb - s_ab - s_ba

    # groups
    g = pd.read_csv(RESULTS_DIR / "audit" / f"{args.bench}_all_groups.csv", encoding="utf-8-sig")
    g["image_id"] = g["image_id"].astype(str)
    d["image_id"] = d["image_id"].astype(str)
    d = d.merge(g[["image_id", "group"]], on="image_id", how="left")
    d["cycle"] = cyc
    d["s_aa"], d["s_bb"], d["s_ab"], d["s_ba"] = s_aa, s_bb, s_ab, s_ba
    d["group"] = d["group"].fillna("unknown")

    def stat(mask):
        v = d.loc[mask, "cycle"].to_numpy()
        if len(v) < 10:
            return {"n": int(len(v))}
        return {"n": int(len(v)), "acc": float((v > 0).mean()),
                "median": float(np.median(v)), "mean": float(v.mean())}

    out = {"bench": args.bench, "backbone": args.backbone, "n_pairs": int(len(d)),
           "overall": stat(np.ones(len(d), bool)),
           "hard": stat((d.group == "persistent_hard").to_numpy()),
           "easy": stat((d.group == "both_low_burden").to_numpy())}

    # cluster bootstrap by donor image (a donor can be reused)
    rng = np.random.default_rng(0)
    uq = d["donor_filename"].astype(str).unique()
    idxs = {k: np.where(d["donor_filename"].astype(str).to_numpy() == k)[0] for k in uq}
    accs = []
    for _ in range(500):
        pick = rng.choice(uq, len(uq), replace=True)
        idx = np.concatenate([idxs[k] for k in pick])
        accs.append(float((cyc[idx] > 0).mean()))
    out["acc_ci"] = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]
    print(out, flush=True)
    write_json(RESULTS_DIR / "gate" / f"cycle_{args.backbone}_{args.bench}.json", out)
    d[["image_id", "group", "cycle", "s_aa", "s_bb", "s_ab", "s_ba"]].to_csv(
        RESULTS_DIR / "gate" / f"cycle_{args.backbone}_{args.bench}.csv", index=False,
        encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
