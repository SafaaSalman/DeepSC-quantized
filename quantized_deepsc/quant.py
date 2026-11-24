"""Uniform scalar quantiser with Gray-labelled indices (torch, any tensor shape).

    idx  = clip(round((t + A)/Delta), 0, 2^b - 1)
    val  = -A + idx*Delta
    bits = gray(idx) = idx XOR (idx >> 1)      # b bits, MSB first
"""
from __future__ import annotations

import torch


def gray(idx: torch.Tensor) -> torch.Tensor:
    return idx ^ (idx >> 1)


def ungray(g: torch.Tensor) -> torch.Tensor:
    idx = g.clone()
    shift = 1
    while shift < 64:
        idx = idx ^ (idx >> shift)
        shift *= 2
        if shift > 32:
            break
    return idx


def int_to_bits(x: torch.Tensor, b: int) -> torch.Tensor:
    """[...] ints -> [..., b] bits, MSB first."""
    shifts = torch.arange(b - 1, -1, -1, device=x.device)
    return ((x.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """[..., b] bits (MSB first) -> [...] ints."""
    b = bits.shape[-1]
    weights = (1 << torch.arange(b - 1, -1, -1, device=bits.device)).long()
    return (bits.long() * weights).sum(-1)


class ScalarQuantizer:
    def __init__(self, A: float, b: int = 4):
        self.A = float(A)
        self.b = int(b)
        self.M = 2 ** self.b
        self.delta = 2.0 * self.A / (self.M - 1)
        self.q_rms = self.delta / (12 ** 0.5)

    # ---- values <-> indices
    def quantize(self, t: torch.Tensor) -> torch.Tensor:
        idx = torch.round((t + self.A) / self.delta)
        return torch.clamp(idx, 0, self.M - 1).long()

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        return -self.A + idx.float() * self.delta

    def levels(self, device=None) -> torch.Tensor:
        return self.dequantize(torch.arange(self.M, device=device))

    # ---- indices <-> Gray bits
    def to_bits(self, idx: torch.Tensor) -> torch.Tensor:
        """[...] -> [..., b] Gray bits, MSB first."""
        return int_to_bits(gray(idx), self.b)

    def from_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """[..., b] Gray bits -> [...] indices."""
        return ungray(bits_to_int(bits))

    def bit_labels(self, device=None) -> torch.Tensor:
        """[M, b] Gray label of every level index."""
        return self.to_bits(torch.arange(self.M, device=device))

    def __repr__(self):
        return f"ScalarQuantizer(A={self.A:.4f}, b={self.b}, delta={self.delta:.4f})"
