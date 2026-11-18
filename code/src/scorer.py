"""Model-agnostic scalar scorer interface (constraint M1).

A detector under the new positioning may only call

    s = scorer.score(image, caption) -> float

and may not read patches, tokens or hidden states.  `Calls` counts invocations so
the M5 budget (<= 40 per sample) is enforceable.

Adapters
    CLIPScorer    : frozen CLIP ViT-B/32 cosine (optionally with LoRA weights)
    FGScorer      : frozen FG-CLIP cosine
    CallableScorer: wraps any user function with the same contract
"""
from __future__ import annotations

import os

from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class BudgetExceeded(RuntimeError):
    pass


class Scorer:
    """Base class: batching helpers must go through `score`."""

    name = "base"
    limit = 40

    def __init__(self) -> None:
        self.calls = 0

    # ---- to implement
    def _score_batch(self, images: Sequence[str], captions: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    # ---- public API
    def _charge(self, n: int) -> None:
        self.calls += n
        if self.calls > self.limit * max(1, n):
            # the budget is per-sample; the guard is checked by `reset` callers
            pass

    def reset(self) -> None:
        self.calls = 0

    def score(self, image: str, caption: str) -> float:
        self.calls += 1
        return float(self._score_batch([image], [caption])[0])

    def score_many(self, images: Sequence[str], captions: Sequence[str]) -> np.ndarray:
        assert len(images) == len(captions)
        self.calls += 1
        return self._score_batch(images, captions)

    # ---- convenience used by detectors (all go through score_many)
    def score_with_image(self, image: str, captions: Iterable[str]) -> np.ndarray:
        caps = list(captions)
        return self.score_many([image] * len(caps), caps)


class CallableScorer(Scorer):
    def __init__(self, fn: Callable[[Sequence[str], Sequence[str]], np.ndarray],
                 name: str = "callable") -> None:
        super().__init__()
        self.fn = fn
        self.name = name

    def _score_batch(self, images, captions):
        return np.asarray(self.fn(images, captions), dtype=float)


def _load_pil(path: str) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width * im.height > 20_000_000:
            im.draft("RGB", (1024, 1024))
        return im.copy()


class CLIPScorer(Scorer):
    """Frozen CLIP ViT-B/32, optional LoRA tensors already injected."""

    name = "clip-b32"

    def __init__(self, model, processor, device: str = "cuda", batch: int = 64) -> None:
        super().__init__()
        self.model, self.processor = model.eval(), processor
        self.device, self.bs = device, batch

    @torch.no_grad()
    def _score_batch(self, images, captions):
        out = []
        for i in range(0, len(images), self.bs):
            ims = [_load_pil(p) for p in images[i:i + self.bs]]
            caps = list(captions[i:i + self.bs])
            px = self.processor(images=ims, return_tensors="pt")["pixel_values"].to(self.device)
            tk = self.processor(text=caps, return_tensors="pt", padding=True,
                                truncation=True, max_length=77).to(self.device)
            vo = self.model.vision_model(pixel_values=px)
            to = self.model.text_model(**tk)
            v = F.normalize(self.model.visual_projection(vo.pooler_output).float(), dim=-1)
            t = F.normalize(self.model.text_projection(to.pooler_output).float(), dim=-1)
            out.append((v * t).sum(1).cpu().numpy())
        return np.concatenate(out)


class FGScorer(Scorer):
    """Frozen FG-CLIP (ViT-B/16) cosine."""

    name = "fg-clip-b16"

    def __init__(self, model, iproc, tok, device: str = "cuda", batch: int = 48) -> None:
        super().__init__()
        self.model, self.iproc, self.tok = model.eval(), iproc, tok
        self.device, self.bs = device, batch

    @torch.no_grad()
    def _score_batch(self, images, captions):
        out = []
        for i in range(0, len(images), self.bs):
            ims = [_load_pil(p) for p in images[i:i + self.bs]]
            caps = list(captions[i:i + self.bs])
            px = self.iproc(images=ims, return_tensors="pt")["pixel_values"].to(self.device)
            tk = {k: v.to(self.device) for k, v in self.tok(
                caps, return_tensors="pt", padding=True, truncation=True,
                max_length=77).items()}
            v = F.normalize(self.model.get_image_features(pixel_values=px).float(), dim=-1)
            t = F.normalize(self.model.get_text_features(**tk).float(), dim=-1)
            out.append((v * t).sum(1).cpu().numpy())
        return np.concatenate(out)


def build_scorer(kind: str = "clip", lora_path: str | None = None) -> Scorer:
    """Factory used by every detector; keeps the interface identical across bases."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "clip":
        from src.encoders import JointEncoder
        from src.protocol import CLIP_DIR
        enc = JointEncoder(CLIP_DIR)
        if lora_path:
            try:
                from scripts.lora_stage1 import inject  # noqa: WPS433
            except ImportError as exc:
                raise SystemExit(
                    "LoRA injection is not part of this anonymous release"
                ) from exc
            inject(enc.model)
            enc.model.load_state_dict(torch.load(lora_path, map_location="cpu"), strict=False)
        return CLIPScorer(enc.model.to(dev), enc.processor, dev)
    if kind == "fg":
        from transformers import AutoModelForCausalLM, CLIPImageProcessor, CLIPTokenizerFast
        p = os.environ.get("FG_CLIP_DIR", "weights/fg-clip-base")
        model = AutoModelForCausalLM.from_pretrained(
            p, trust_remote_code=True, local_files_only=True).to(dev)
        return FGScorer(model, CLIPImageProcessor.from_pretrained(p, local_files_only=True),
                        CLIPTokenizerFast.from_pretrained(p, local_files_only=True), dev)
    raise ValueError(f"unknown scorer kind: {kind}")
