"""The short code + ORBGRAND variants (g_bpsk, g_pam16) of the digital chain."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from quantized_deepsc.chain import DigitalChain
from quantized_deepsc.grand_code import GrandCode
from quantized_deepsc.modulation import PAM
from quantized_deepsc.quant import ScalarQuantizer

GRAND_DIR = Path(__file__).resolve().parents[3] / "BitSemCom" / "grand"


class _NoLink:
    Channels = None


@pytest.fixture(scope="module")
def code():
    return GrandCode(GRAND_DIR, "crc:0x9eb2:128:112", max_queries=2000)


@pytest.fixture(scope="module")
def chain(device, code):
    return DigitalChain(_NoLink(), ScalarQuantizer(4.54, 4), PAM(2, device), PAM(16, device), None, code)


def test_code_is_systematic_and_satisfies_parity(code):
    torch.manual_seed(0)
    u = torch.randint(0, 2, (50, code.k), dtype=torch.uint8)
    cw = code.encode(u)
    assert cw.shape == (50, 128) and torch.equal(cw[:, :code.k], u)
    assert not ((cw.numpy().astype(np.int64) @ code.H.T.astype(np.int64)) % 2).any()


def test_decoder_corrects_the_least_reliable_errors(code):
    """Two flipped bits that are also the two least reliable ones are found within a few queries."""
    torch.manual_seed(1)
    u = torch.randint(0, 2, (20, code.k), dtype=torch.uint8)
    cw = code.encode(u)
    llr = (1.0 - 2.0 * cw.double()) * (4.0 + torch.rand(cw.shape, dtype=torch.double))
    for i in range(cw.shape[0]):
        for j in torch.randperm(code.n)[:2]:
            llr[i, j] *= -0.05                      # wrong sign, low magnitude
    words, abandoned, queries = code.decode(llr)
    assert torch.equal(words, cw) and not abandoned.any()
    assert int(queries.max()) <= 10


def test_clean_blocks_cost_one_query_and_garbage_is_abandoned(code):
    torch.manual_seed(2)
    cw = code.encode(torch.randint(0, 2, (4, code.k), dtype=torch.uint8))
    words, abandoned, queries = code.decode((1.0 - 2.0 * cw.double()) * 5.0)
    assert torch.equal(words, cw) and not abandoned.any() and (queries == 1).all()
    noise = torch.randn(200, code.n, dtype=torch.double)   # no codeword in it: within 2000 queries a
    _, abandoned, queries = code.decode(noise)             # false hit has probability 2000/2^16 = 3 %
    assert abandoned.float().mean() > 0.9
    assert int(queries.max()) <= 2000


@pytest.mark.parametrize("variant", ["g_bpsk", "g_pam16"])
@pytest.mark.parametrize("channel", ["none", "rayleigh"])
def test_noiseless_chain_reproduces_quantised_latent(chain, device, variant, channel):
    torch.manual_seed(3)
    t = 1.2 * torch.randn(4, 23, 16, device=device)
    t_hat, st = chain.run(variant, t, 0.0, channel, "sentence")
    assert torch.equal(t_hat, chain.Q.dequantize(chain.Q.quantize(t)))
    assert st["bit_errors"] == 0 and st["block_errors"] == 0 and st["grand_abandoned"] == 0
    nblk = -(-(23 * 64) // 112)
    assert st["n_blocks"] == 4 * nblk and st["pad_bits"] == 4 * (nblk * 112 - 23 * 64)
    assert st["real_uses"] == 4 * nblk * 128 // (1 if variant == "g_bpsk" else 4)


def test_grand_beats_uncoded_at_moderate_noise(chain, device):
    torch.manual_seed(11)
    t = 1.2 * torch.randn(16, 25, 16, device=device)
    sigma = 1 / math.sqrt(2 * 10 ** 0.5)          # label 5 dB: uncoded BPSK BER = Q(2.51) = 6e-3
    _, su = chain.run("u_bpsk", t, sigma, "awgn", "sentence")
    _, sg = chain.run("g_bpsk", t, sigma, "awgn", "sentence")
    ber_u = su["bit_errors"] / su["n_bits"]
    ber_g = sg["bit_errors"] / sg["n_bits"]
    assert 0.003 < ber_u < 0.012
    assert ber_g < ber_u / 3
