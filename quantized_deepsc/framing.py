"""Per-sentence framing of latent bits into LDPC information blocks.

Within a batch every sentence has the same padded length L, so the bit stream of
each sentence has the same length Nb = L*D*b and the same number of blocks.
"""
from __future__ import annotations

import torch


def frame(bits: torch.Tensor, k: int):
    """bits [B, Nb] -> (blocks [B*nblk, k], nblk, n_pad_bits) with zero padding at the end."""
    B, Nb = bits.shape
    nblk = -(-Nb // k)
    n_pad = nblk * k - Nb
    if n_pad:
        bits = torch.cat([bits, torch.zeros(B, n_pad, dtype=bits.dtype, device=bits.device)], dim=1)
    return bits.reshape(B * nblk, k), nblk, n_pad


def unframe(blocks: torch.Tensor, B: int, Nb: int) -> torch.Tensor:
    """blocks [B*nblk, k] -> bits [B, Nb] (padding stripped)."""
    return blocks.reshape(B, -1)[:, :Nb]
