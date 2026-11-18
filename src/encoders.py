"""Frozen CLIP ViT-B/32 wrapper exposing token and patch states.

Why this exists
---------------
The pooled 512-d embedding is what every recorded baseline uses.  The Gate needs
the *unpooled* states, i.e. the 77 text-token states and the 49 patch states,
both projected into the joint space.  No parameter is updated anywhere.

Projection convention (verified against the official pooled embeddings):
* image patches: ``visual_projection(patch_hidden)``
* text tokens:  ``text_projection(token_hidden)``
* pooled image:  ``visual_projection(cls_hidden)``  == ``get_image_features``
* pooled text:   ``text_projection(eot_hidden)``    == ``get_text_features``
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# MM-IMDb has 75 MP posters; allow them but down-shift before decoding (see below)
Image.MAX_IMAGE_PIXELS = 300_000_000


@dataclass
class BatchStates:
    token: np.ndarray          # (B, L, D) float32, L2-normalised
    token_mask: np.ndarray     # (B, L) bool, special/pad tokens already False
    patch: np.ndarray          # (B, P, D) float32, L2-normalised
    pooled_img: np.ndarray     # (B, D)
    pooled_txt: np.ndarray     # (B, D)
    word_ids: list[list[int | None]]
    input_ids: np.ndarray      # (B, L) int64, for subword decoding


class _PathDataset(Dataset):
    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as im:
            # only down-shift huge posters (MM-IMDb); otherwise keep the exact
            # pixels the recorded CLIP cache was built from
            if im.width * im.height > 20_000_000:
                im.draft("RGB", (1024, 1024))
            return im.convert("RGB")


def _identity(batch):
    return batch


class JointEncoder:
    def __init__(self, model_dir: Path | str, device: str = "auto",
                 logger: logging.Logger | None = None) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.model_dir = str(model_dir)
        self.logger = logger or logging.getLogger("mymodel7")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(self.model_dir).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_dir)
        self.tokenizer = self.processor.tokenizer
        cfg = self.model.config
        self.dim = int(cfg.projection_dim)
        self.grid = int(round((cfg.vision_config.image_size // cfg.vision_config.patch_size)))
        self.max_len = int(cfg.text_config.max_position_embeddings)
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.logger.info("CLIP ready: dim=%d grid=%dx%d max_len=%d device=%s",
                         self.dim, self.grid, self.grid, self.max_len, self.device)

    # -------------------------------------------------------------- image only
    @torch.no_grad()
    def encode_images(self, paths: Sequence[str], batch_size: int = 64,
                      num_workers: int = 4) -> np.ndarray:
        loader = DataLoader(_PathDataset(paths), batch_size=batch_size,
                            num_workers=num_workers, collate_fn=_identity,
                            shuffle=False)
        out: list[np.ndarray] = []
        for pil in loader:
            images = self.processor(images=list(pil), return_tensors="pt")
            images = {k: v.to(self.device) for k, v in images.items()}
            vision = self.model.vision_model(pixel_values=images["pixel_values"])
            emb = F.normalize(
                self.model.visual_projection(vision.pooler_output).float(), dim=-1)
            out.append(emb.cpu().numpy())
        return np.concatenate(out, 0)

    # --------------------------------------------------------------- text only
    @torch.no_grad()
    def encode_texts(self, captions: Sequence[str], batch_size: int = 256,
                     max_length: int | None = None) -> np.ndarray:
        """Pooled, normalised text embeddings (no image forward)."""
        max_length = int(max_length or self.max_len)
        out: list[np.ndarray] = []
        for start in range(0, len(captions), batch_size):
            chunk = list(captions[start:start + batch_size])
            text = self.processor(text=chunk, return_tensors="pt", padding=True,
                                  truncation=True, max_length=max_length)
            text = {k: v.to(self.device) for k, v in text.items()}
            t_out = self.model.text_model(input_ids=text["input_ids"],
                                          attention_mask=text["attention_mask"])
            emb = F.normalize(self.model.text_projection(t_out.pooler_output).float(), dim=-1)
            out.append(emb.cpu().numpy())
        return np.concatenate(out, 0)

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(self, paths: Sequence[str], captions: Sequence[str],
               batch_size: int = 32, num_workers: int = 4,
               max_length: int | None = None,
               project_text: bool = True) -> BatchStates:
        max_length = int(max_length or self.max_len)
        loader = DataLoader(_PathDataset(paths), batch_size=batch_size,
                            num_workers=num_workers, collate_fn=_identity,
                            shuffle=False)
        tokens, masks, patches, pimgs, ptxts, word_ids, ids = [], [], [], [], [], [], []
        cursor = 0
        for pil in loader:
            caps = list(captions[cursor:cursor + len(pil)])
            cursor += len(pil)
            images = self.processor(images=list(pil), return_tensors="pt")
            images = {k: v.to(self.device) for k, v in images.items()}
            text = self.processor(text=caps, return_tensors="pt", padding=True,
                                  truncation=True, max_length=max_length)
            text = {k: v.to(self.device) for k, v in text.items()}

            vision = self.model.vision_model(pixel_values=images["pixel_values"])
            hidden = vision.last_hidden_state
            # NOTE: ``pooler_output`` is *not* ``last_hidden_state[:, 0]`` in this
            # checkpoint (cosine 0.90); only the pooler reproduces the recorded
            # official embeddings.  Patches use the same visual_projection.
            patch = self.model.visual_projection(hidden[:, 1:, :])
            patch = F.normalize(patch.float(), dim=-1)

            t_out = self.model.text_model(input_ids=text["input_ids"],
                                          attention_mask=text["attention_mask"])
            tok = t_out.last_hidden_state
            if project_text:
                tok = self.model.text_projection(tok)
            tok = F.normalize(tok.float(), dim=-1)

            p_img = F.normalize(
                self.model.visual_projection(vision.pooler_output).float(), dim=-1)
            p_txt = F.normalize(self.model.text_projection(t_out.pooler_output).float(), dim=-1)

            mask = text["attention_mask"].bool().clone()
            for sid in self.special_ids:
                mask &= ~(text["input_ids"] == sid)

            # pad to the global text length so batches can be concatenated
            b, cur = tok.shape[0], tok.shape[1]
            if cur < max_length:
                pad = max_length - cur
                tok = torch.cat([tok, tok.new_zeros(b, pad, tok.shape[2])], dim=1)
                mask = torch.cat([mask, mask.new_zeros(b, pad)], dim=1)
            tok_np = tok.cpu().numpy()
            mask_np = mask.cpu().numpy()
            ids_np = text["input_ids"].cpu().numpy().astype(np.int64)
            if ids_np.shape[1] < max_length:
                ids_np = np.pad(ids_np, ((0, 0), (0, max_length - ids_np.shape[1])))

            tokens.append(tok_np)
            masks.append(mask_np)
            patches.append(patch.cpu().numpy())
            pimgs.append(p_img.cpu().numpy())
            ptxts.append(p_txt.cpu().numpy())
            ids.append(ids_np)

            enc = self.tokenizer(list(caps), padding=True, truncation=True,
                                 max_length=max_length)
            for i in range(len(caps)):
                word_ids.append(enc.word_ids(batch_index=i))

        return BatchStates(
            token=np.concatenate(tokens, 0),
            token_mask=np.concatenate(masks, 0),
            patch=np.concatenate(patches, 0),
            pooled_img=np.concatenate(pimgs, 0),
            pooled_txt=np.concatenate(ptxts, 0),
            word_ids=word_ids,
            input_ids=np.concatenate(ids, 0),
        )
