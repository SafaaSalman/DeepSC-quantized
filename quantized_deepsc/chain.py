"""The link variants between DeepSC encode() and decode().

    c0       analog: t -> channel -> t_hat
    q0       quantise -> dequantise (no channel): the digital ceiling
    u_bpsk   quantise -> Gray bits -> BPSK   -> channel -> hard decision -> dequantise
    u_pam16  quantise -> Gray bits -> 16-PAM -> channel -> hard decision -> dequantise
    c_bpsk   quantise -> Gray bits -> LDPC -> BPSK   -> channel -> LLR -> BP -> dequantise
    c_pam16  quantise -> Gray bits -> LDPC -> 16-PAM -> channel -> max-log LLR -> BP -> dequantise

No interleaver is used: the fading gain is constant over a sentence and the AWGN channel is
memoryless, so there is nothing to spread.
"""
from __future__ import annotations

import torch

from .channel import awgn, rayleigh_zf
from .framing import frame, unframe

VARIANTS = ["c0", "q0", "u_bpsk", "u_pam16", "c_bpsk", "c_pam16"]


def apply_channel(x: torch.Tensor, sigma: float, channel: str, h_draw: str, link=None):
    """x [B, N] -> (y [B, N], h2 or None).  h_draw='repo' uses utils.Channels.Rayleigh (h per batch, no h2)."""
    if channel == "none":
        return x, None
    if channel == "awgn":
        return awgn(x, sigma), None
    if channel == "rayleigh":
        if h_draw == "repo":
            return link.Channels().Rayleigh(x, sigma), None
        return rayleigh_zf(x, sigma, h_draw)
    raise ValueError(f"unknown channel {channel!r}")


class DigitalChain:
    def __init__(self, link, quantizer, pam_bpsk, pam16, ldpc=None):
        self.link = link
        self.Q = quantizer
        self.pam = {"bpsk": pam_bpsk, "pam16": pam16}
        self.ldpc = ldpc

    @torch.no_grad()
    def run(self, variant: str, t: torch.Tensor, sigma: float, channel: str, h_draw: str):
        """t [B, L, D] -> (t_hat [B, L, D], stats dict of sums/counts for this batch)."""
        B, L, D = t.shape
        st = {"tokens_all": B * L, "real_uses": 0}
        if variant == "c0":
            y, _ = apply_channel(t.reshape(B, L * D), sigma, channel, h_draw, self.link)
            st["real_uses"] = B * L * D
            return y.reshape(B, L, D), st

        idx = self.Q.quantize(t)
        if variant == "q0":
            return self.Q.dequantize(idx), st

        bits = self.Q.to_bits(idx).reshape(B, -1)              # [B, L*D*b]
        Nb = bits.shape[1]
        pam = self.pam["bpsk" if variant.endswith("bpsk") else "pam16"]

        if variant.startswith("u_"):
            x = pam.modulate(bits)
            y, h2 = apply_channel(x, sigma, channel, h_draw, self.link)
            bits_hat = pam.hard_demod(y)
        elif variant.startswith("c_"):
            if self.ldpc is None:
                raise ValueError("coded variant needs an LDPC code")
            if h_draw == "repo":
                raise ValueError("coded variants need |h|^2 for the LLR; use h_draw=sentence/batch/symbol")
            blocks, nblk, n_pad = frame(bits, self.ldpc.k)     # [B*nblk, k]
            cw = self.ldpc.encode(blocks)                        # [B*nblk, n]
            x = pam.modulate(cw.reshape(B, -1))                  # [B, nblk*n/m]
            y, h2 = apply_channel(x, sigma, channel, h_draw, self.link)
            llr = pam.llr_maxlog(y, sigma * sigma, h2)           # [B, nblk*n]
            dec, conv, iters = self.ldpc.decode(llr.reshape(B * nblk, -1))
            info_hat = dec[:, :self.ldpc.k]
            bits_hat = unframe(info_hat, B, Nb)
            st["n_blocks"] = B * nblk
            st["block_errors"] = int((info_hat != blocks).any(-1).sum())
            st["blocks_converged"] = int(conv.sum())
            st["iters_sum"] = float(iters.sum())
            st["pad_bits"] = B * n_pad
        else:
            raise ValueError(f"unknown variant {variant!r}")

        st["real_uses"] = int(x.numel())
        st["n_bits"] = int(bits.numel())
        st["bit_errors"] = int((bits != bits_hat).sum())
        idx_hat = self.Q.from_bits(bits_hat.reshape(B, L, D, self.Q.b))
        st["n_levels"] = int(idx.numel())
        st["level_errors"] = int((idx != idx_hat).sum())
        return self.Q.dequantize(idx_hat), st
