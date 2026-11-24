"""M-PAM with unit average energy per real dimension and Gray labelling.

Symbol index k in 0..M-1 has level (M-1-2k)*d with d = sqrt(3/(M^2-1)) and label
gray(k) (MSB first).  For M = 2 this is BPSK with bit 0 -> +1, bit 1 -> -1, so the
LLR sign convention is  L > 0 <=> bit 0  and the BPSK LLR is 2*h2*y/sigma^2.
"""
from __future__ import annotations

import math

import torch

from .quant import bits_to_int, gray, int_to_bits, ungray


class PAM:
    def __init__(self, M: int, device=None):
        assert M >= 2 and (M & (M - 1)) == 0, "M must be a power of two"
        self.M = int(M)
        self.m = int(round(math.log2(M)))          # bits per real channel use
        self.d = math.sqrt(3.0 / (M * M - 1.0))   # half the level spacing; E[x^2] = 1
        self.device = torch.device(device) if device is not None else None
        k = torch.arange(self.M, device=self.device)
        self.levels = ((self.M - 1) - 2 * k).float() * self.d           # [M], descending
        self.labels = int_to_bits(gray(k), self.m)                     # [M, m]
        # per bit position j: masks of symbols whose j-th bit is 0 / 1
        self.mask0 = (self.labels == 0).T.contiguous()                  # [m, M]
        self.mask1 = (self.labels == 1).T.contiguous()

    @property
    def bits_per_use(self) -> int:
        return self.m

    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        """bits [B, Nbits] (Nbits multiple of m) -> x [B, Nbits/m]."""
        B = bits.shape[0]
        g = bits_to_int(bits.reshape(B, -1, self.m))
        k = ungray(g)
        return self.levels[k]

    def nearest_index(self, y: torch.Tensor) -> torch.Tensor:
        """Hard decision: symbol index of the nearest level."""
        k = torch.round(((self.M - 1) * self.d - y) / (2 * self.d))
        return torch.clamp(k, 0, self.M - 1).long()

    def hard_demod(self, y: torch.Tensor) -> torch.Tensor:
        """y [B, Ns] -> bits [B, Ns*m]."""
        k = self.nearest_index(y)
        return self.labels[k].reshape(y.shape[0], -1)

    def llr_maxlog(self, y: torch.Tensor, sigma2: float, h2=None) -> torch.Tensor:
        """Max-log per-bit LLRs (L > 0 <=> bit 0).

        y  [B, Ns]   ZF-equalised received samples
        h2 |h|^2 broadcastable to [B, Ns] (None -> 1, i.e. AWGN); the effective noise
           variance after ZF is sigma2/h2 conditional on h.
        Returns [B, Ns*m].
        """
        sigma2 = max(float(sigma2), 1e-12)   # noiseless input must give finite (huge) LLRs, not NaN
        diff2 = (y.unsqueeze(-1) - self.levels) ** 2                     # [B, Ns, M]
        if h2 is None:
            metric = -diff2 / (2.0 * sigma2)
        else:
            h2 = torch.as_tensor(h2, device=y.device, dtype=y.dtype)
            metric = -(h2.unsqueeze(-1) * diff2) / (2.0 * sigma2)
        neg = torch.tensor(-float("inf"), device=y.device, dtype=y.dtype)
        out = []
        for j in range(self.m):
            l0 = torch.where(self.mask0[j], metric, neg).max(-1).values
            l1 = torch.where(self.mask1[j], metric, neg).max(-1).values
            out.append(l0 - l1)
        return torch.stack(out, -1).reshape(y.shape[0], -1)

    def __repr__(self):
        return f"PAM(M={self.M}, m={self.m}, d={self.d:.4f})"
