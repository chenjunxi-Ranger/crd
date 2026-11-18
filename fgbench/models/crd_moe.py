"""Modality-specific gated mixture-of-experts for CRD v2.

Each modality gets its own bank of experts and its own gate; nothing is shared
between the two sides. The additive form is preserved:

    q_I(u) = w_I' u + sum_k pi_{I,k}(u) e_{I,k}(u)
    q_T(v) = w_T' v + sum_k pi_{T,k}(v) e_{T,k}(v)
    z_hat  = b0 + q_I(u) + q_T(v)

The linear branch keeps the simple feature-direction effect that a plain MLP has
to rediscover; the experts model several non-linear regimes; the gate decides per
sample which regime applies.

Why a separate bank per modality: u and v are both 512-d but live in different
spaces, so sharing experts would force one parameter set to serve two different
statistics. Nothing in this module couples the two sides except the summed loss.

Reference parameter counts (d=512, width=64):

    plain MLP    2 x (512*64 + 64 + 64 + 1)              =  65,794
    MoE k=2      2 x (2*(512*64+64+65) + 1026 + 513)     = 134,690
    wide MLP     2 x (512*128 + 128 + 129)               = 131,330   <- control
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityMoE(nn.Module):
    """One modality: linear path + K gated experts."""

    def __init__(self, d: int = 512, k: int = 2, width: int = 64,
                 use_linear: bool = True):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(d, 1) if use_linear else None
        self.gate = nn.Linear(d, k)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
            for _ in range(k)])

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        logits = self.gate(x)                       # (N, k)
        pi = torch.softmax(logits, dim=-1)
        e = torch.cat([ex(x) for ex in self.experts], dim=-1)   # (N, k)
        out = (pi * e).sum(-1)
        if self.linear is not None:
            out = out + self.linear(x).squeeze(-1)
        return (out, pi) if return_gate else out


class MoEAdditive(nn.Module):
    """Two independent modality banks plus a shared bias."""

    def __init__(self, d: int = 512, k: int = 2, width: int = 64,
                 parts: str = "both"):
        super().__init__()
        self.wa = parts in ("both", "image")
        self.wb = parts in ("both", "text")
        self.a = ModalityMoE(d, k, width) if self.wa else None
        self.b = ModalityMoE(d, k, width) if self.wb else None
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward_q(self, u: torch.Tensor, v: torch.Tensor):
        qI = self.a(u) if self.wa else torch.zeros(len(u), device=u.device)
        qT = self.b(v) if self.wb else torch.zeros(len(v), device=v.device)
        return qI, qT, qI + qT + self.b0

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        return self.forward_q(u, v)[2].squeeze(-1)

    @torch.no_grad()
    def gate_stats(self, u: torch.Tensor, v: torch.Tensor) -> dict:
        """Mean gate weight per expert, per modality.

        A collapsed gate puts ~1.0 on one expert; a used bank spreads mass. This
        is the difference between a real mixture and an expensive single MLP.
        """
        out = {}
        if self.wa:
            _, pi = self.a(u, return_gate=True)
            out["image_gate"] = pi.mean(0).cpu().numpy().round(3).tolist()
            out["image_entropy"] = float(
                -(pi * pi.clamp_min(1e-9).log()).sum(-1).mean().cpu())
        if self.wb:
            _, pi = self.b(v, return_gate=True)
            out["text_gate"] = pi.mean(0).cpu().numpy().round(3).tolist()
            out["text_entropy"] = float(
                -(pi * pi.clamp_min(1e-9).log()).sum(-1).mean().cpu())
        return out


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    from fgbench.models.crd_backbone import Off
    for name, mod in (("mlp64", Off(d=512, width=64)),
                      ("mlp128", Off(d=512, width=128)),
                      ("moe2", MoEAdditive(d=512, k=2, width=64))):
        print(f"{name:8s} {param_count(mod):,} params")
