"""Convert pytorch_model.bin -> model.safetensors (transformers refuses .bin on torch 2.5)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

D = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights\blip-itm-base-coco")


def main() -> int:
    src, dst = D / "pytorch_model.bin", D / "model.safetensors"
    if dst.is_file():
        print("already exists", dst.stat().st_size)
        return 0
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    sd = {k: v.contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    save_file(sd, dst, metadata={"format": "pt"})
    print("wrote", dst, round(dst.stat().st_size / 1e6, 1), "MB", len(sd), "tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
