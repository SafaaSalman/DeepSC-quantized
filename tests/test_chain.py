"""End-to-end checks of the digital chain on synthetic latents (no DeepSC model needed).

For every variant the chain must (i) reproduce the quantised latent exactly when the
channel is noiseless, (ii) count channel uses as the variants table in README.md says, (iii) keep the
LDPC framing/padding invisible, and (iv) recover the analog latent exactly for c0.
"""
import math

import pytest
import torch

from quantized_deepsc.chain import DigitalChain
from quantized_deepsc.ldpc import LDPC
from quantized_deepsc.modulation import PAM
from quantized_deepsc.quant import ScalarQuantizer


class _NoLink:  # apply_channel only touches link for h_draw='repo'
    Channels = None


@pytest.fixture(scope="module")
def chain(device):
    Q = ScalarQuantizer(4.54, 4)
    code = LDPC(n=1008, dv=3, dc=6, k=504, seed=1234, cache_dir=None, device=device)
    return DigitalChain(_NoLink(), Q, PAM(2, device), PAM(16, device), code)


@pytest.fixture(scope="module")
def latent(device):
    torch.manual_seed(3)
    B, L, D = 4, 23, 16
    t = 1.2 * torch.randn(B, L, D, device=device)
    t[0, 0, 0] = 9.0   # beyond +A: must clip, not overflow
    t[1, 2, 3] = -9.0
    return t


@pytest.mark.parametrize("variant", ["q0", "u_bpsk", "u_pam16", "c_bpsk", "c_pam16"])
def test_noiseless_channel_reproduces_quantised_latent(chain, latent, variant):
    t_hat, st = chain.run(variant, latent, 0.0, "none", "sentence")
    ref = chain.Q.dequantize(chain.Q.quantize(latent))
    assert torch.equal(t_hat, ref)
    if variant != "q0":
        assert st["bit_errors"] == 0 and st["level_errors"] == 0
    if variant.startswith("c_"):
        assert st["block_errors"] == 0 and st["blocks_converged"] == st["n_blocks"]


@pytest.mark.parametrize("variant", ["c_bpsk", "c_pam16"])
def test_coded_noiseless_fading_is_exact(chain, latent, variant):
    """With sigma = 0 the ZF output equals x exactly, whatever h; the LLRs then have the right sign."""
    t_hat, st = chain.run(variant, latent, 0.0, "rayleigh", "sentence")
    assert torch.equal(t_hat, chain.Q.dequantize(chain.Q.quantize(latent)))
    assert st["bit_errors"] == 0


def test_channel_uses_per_token(chain, latent):
    B, L, D = latent.shape
    expect = {"u_bpsk": 64.0, "u_pam16": 16.0}
    for v, u in expect.items():
        _, st = chain.run(v, latent, 0.0, "none", "sentence")
        assert st["real_uses"] / st["tokens_all"] == u
    nblk = -(-(L * 64) // 504)
    for v, bits_per_use in (("c_bpsk", 1), ("c_pam16", 4)):
        _, st = chain.run(v, latent, 0.0, "none", "sentence")
        assert st["n_blocks"] == B * nblk
        assert st["pad_bits"] == B * (nblk * 504 - L * 64)
        assert st["real_uses"] == B * nblk * 1008 // bits_per_use
    _, st = chain.run("c0", latent, 0.0, "none", "sentence")
    assert st["real_uses"] / st["tokens_all"] == 16.0


def test_analog_noiseless_and_awgn_scale(chain, latent, device):
    t_hat, _ = chain.run("c0", latent, 0.0, "none", "sentence")
    assert torch.equal(t_hat, latent)
    torch.manual_seed(0)
    big = torch.zeros(8, 250, 16, device=device)
    sigma = 0.3
    y, _ = chain.run("c0", big, sigma, "awgn", "sentence")
    assert abs(float(y.std()) - sigma) < 0.01
    y, _ = chain.run("c0", big, sigma, "rayleigh", "symbol")
    assert float(y.std()) > sigma  # ZF amplifies noise in fades


def test_hard_decision_matches_llr_sign(chain, latent, device):
    """Uncoded hard decision and the sign of the max-log LLR must agree symbol by symbol."""
    torch.manual_seed(5)
    for M in (2, 16):
        pam = chain.pam["bpsk" if M == 2 else "pam16"]
        bits = torch.randint(0, 2, (3, 4 * 300), dtype=torch.uint8, device=device)
        x = pam.modulate(bits)
        y = x + 0.2 * torch.randn_like(x)
        hard = pam.hard_demod(y)
        soft = (pam.llr_maxlog(y, 0.04, None) < 0).to(torch.uint8)
        assert torch.equal(hard, soft)


def test_coded_beats_uncoded_at_moderate_noise(chain, device):
    torch.manual_seed(11)
    t = 1.2 * torch.randn(16, 25, 16, device=device)
    sigma = 1 / math.sqrt(2 * 10 ** (0.2))   # label 2 dB
    _, su = chain.run("u_bpsk", t, sigma, "awgn", "sentence")
    _, sc = chain.run("c_bpsk", t, sigma, "awgn", "sentence")
    ber_u = su["bit_errors"] / su["n_bits"]
    ber_c = sc["bit_errors"] / sc["n_bits"]
    assert 0.01 < ber_u < 0.1          # uncoded BPSK at 2 dB: Q(sqrt(2*1.58)) ~ 3.7e-2
    assert ber_c < ber_u / 10          # rate-1/2 LDPC at Eb/N0 = 5 dB: essentially error free
