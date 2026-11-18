"""Cache global embeddings (M1' interface) for BCPR: train + val, per backbone."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import RESULTS_DIR, load_split   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["coco", "flickr", "mmimdb"])
    ap.add_argument("--backbone", default="clip", choices=["clip", "fg"])
    ap.add_argument("--train-cap", type=int, default=8000)
    args = ap.parse_args()

    out = RESULTS_DIR / "gate"
    out.mkdir(parents=True, exist_ok=True)

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

    for split, cap in (("train", args.train_cap), ("val", None)):
        p = out / f"emb_{args.backbone}_{args.bench}_{split}.npz"
        if p.is_file():
            print("cached", p.name); continue
        df = load_split(args.bench, split, cap)
        vi = ei(df["path"].tolist())
        vt = et(df["caption"].astype(str).tolist())
        np.savez_compressed(p, img=vi, txt=vt, label=df["label"].to_numpy(),
                            image_id=df["image_id"].astype(str).to_numpy())
        print("saved", p.name, vi.shape, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
