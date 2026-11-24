"""AWGN and Rayleigh channels with zero-forcing, mirroring the repo's conventions.

* sigma is the noise standard deviation PER REAL DIMENSION (utils.SNR_to_noise).
* Consecutive real pairs form one complex symbol (utils.Channels.Rayleigh views the
  tensor as [B, -1, 2]).
* h ~ CN(0, 1) (E|h|^2 = 1); r = h*x + n; ZF: y = r/h.  Conditional on h the
  effective noise per real dimension is N(0, sigma^2/|h|^2).
"""
from __future__ import annotations

import math

import torch


def awgn(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x + sigma * torch.randn_like(x)


def rayleigh_zf(x: torch.Tensor, sigma: float, h_draw: str = "sentence"):
    """x [B, N] real (N even) -> (y [B, N] after ZF, h2 = |h|^2 broadcastable to [B, N]).

    h_draw: 'sentence' (one h per row), 'batch' (one h for the whole batch),
            'symbol' (independent h per complex symbol).
    """
    B, N = x.shape
    assert N % 2 == 0, "need an even number of real values (pairs form complex symbols)"
    xc = torch.view_as_complex(x.reshape(B, N // 2, 2).contiguous())
    if h_draw == "sentence":
        shape = (B, 1)
    elif h_draw == "batch":
        shape = (1, 1)
    elif h_draw == "symbol":
        shape = (B, N // 2)
    else:
        raise ValueError(f"unknown h_draw {h_draw!r}")
    h = torch.complex(torch.randn(shape, device=x.device), torch.randn(shape, device=x.device)) / math.sqrt(2.0)
    n = torch.complex(torch.randn(B, N // 2, device=x.device), torch.randn(B, N // 2, device=x.device)) * sigma
    y = (h * xc + n) / h
    yr = torch.view_as_real(y).reshape(B, N)
    h2 = (h.real ** 2 + h.imag ** 2)
    if h_draw == "symbol":
        h2 = h2.repeat_interleave(2, dim=1)  # per real dimension [B, N]
    return yr, h2
