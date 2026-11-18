"""Count parameters of every frozen backbone used in the paper.

Reads tensor shapes only (safetensors headers / mmap), so nothing is loaded into
memory. Reports total, vision-tower and text-tower counts where the checkpoint
exposes separate prefixes.

Run:  python fgbench/experiments/p0_baselines/count_params.py
"""
from __future__ import annotations

import json
from pathlib import Path

WEIGHTS = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights")
DIRS = ["clip-vit-base-patch16", "clip-vit-base-patch32", "clip-vit-large-patch14",
        "siglip2-base-patch16-224", "blip-itm-base-coco", "flava-full"]

VISION_PREFIX = ("visual", "vision_model", "vision_tower", "image_encoder")
TEXT_PREFIX = ("text_model", "text_encoder", "text_transformer", "language_model")


def shapes_from_safetensors(p: Path) -> dict[str, tuple]:
    from safetensors import safe_open
    out = {}
    with safe_open(str(p), framework="pt") as f:
        for k in f.keys():
            out[k] = tuple(f.get_slice(k).get_shape())
    return out


def shapes_from_bin(p: Path) -> dict[str, tuple]:
    import torch
    sd = torch.load(str(p), map_location="meta", mmap=True, weights_only=True)
    return {k: tuple(v.shape) for k, v in sd.items()}


def main() -> int:
    rows = []
    for name in DIRS:
        d = WEIGHTS / name
        if not d.is_dir():
            print(f"missing {name}")
            continue
        files = sorted(d.glob("*.safetensors")) or sorted(d.glob("pytorch_model.bin"))
        if not files:
            print(f"no weights in {name}")
            continue
        shapes = {}
        for f in files:
            shapes.update(shapes_from_safetensors(f) if f.suffix == ".safetensors"
                          else shapes_from_bin(f))
        total = sum(int(__import__("math").prod(s)) for s in shapes.values())
        vis = sum(int(__import__("math").prod(s)) for k, s in shapes.items()
                  if k.lower().startswith(VISION_PREFIX))
        txt = sum(int(__import__("math").prod(s)) for k, s in shapes.items()
                  if k.lower().startswith(TEXT_PREFIX))
        cfg = {}
        if (d / "config.json").is_file():
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        rows.append({"model": name, "total_M": round(total / 1e6, 1),
                     "vision_M": round(vis / 1e6, 1), "text_M": round(txt / 1e6, 1),
                     "arch": cfg.get("architectures", cfg.get("model_type", ""))})
        print(rows[-1], flush=True)

    base = next(r for r in rows if r["model"] == "clip-vit-base-patch16")["total_M"]
    large = next(r for r in rows if r["model"] == "clip-vit-large-patch14")["total_M"]
    print(f"\nL/14 : B/16 parameter ratio = {large / base:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
