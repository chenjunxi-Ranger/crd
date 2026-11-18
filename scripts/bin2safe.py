"""Convert <dir>/pytorch_model.bin -> model.safetensors (transformers blocks .bin on torch<2.6)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> int:
    d = Path(sys.argv[1])
    src, dst = d / "pytorch_model.bin", d / "model.safetensors"
    if dst.is_file() and dst.stat().st_size > 1e8:
        print("already exists", round(dst.stat().st_size / 1e6, 1), "MB")
        return 0
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    out, seen = {}, set()
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        key = (v.data_ptr(), tuple(v.shape), tuple(v.stride()))
        if key in seen:                 # tied weights must be cloned before saving
            v = v.clone()
        seen.add(key)
        out[k] = v.contiguous()
    save_file(out, dst, metadata={"format": "pt"})
    print("wrote", dst, round(dst.stat().st_size / 1e6, 1), "MB", len(sd), "tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
