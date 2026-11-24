import math

import numpy as np
import pytest
import torch

from quantized_deepsc.framing import frame, unframe
from quantized_deepsc.ldpc import LDPC, count_4cycles, gf2_rref


@pytest.fixture(scope="module")
def code(device):
    return LDPC(n=1008, dv=3, dc=6, k=504, seed=1234, cache_dir=None, device=device, bp_iters=50)


def test_gf2_rref_small():
    H = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]], dtype=np.uint8)
    R, piv = gf2_rref(H)
    assert len(piv) == 2 and R.shape[0] == 2  # third row = sum of first two


def test_construction(code):
    H = code.H
    assert H.shape == (504, 1008)
    assert (H.sum(0) == 3).all() and (H.sum(1) == 6).all()
    assert count_4cycles(H) == 0
    assert code.k == 504 and code.k_eff >= 504 and code.n_short == code.k_eff - 504
    assert code.rank == 1008 - code.k_eff


def test_encode_syndrome_and_noiseless_decode(code, device):
    u = torch.randint(0, 2, (64, code.k), dtype=torch.uint8, device=device)
    cw = code.encode(u)
    assert cw.shape == (64, 1008)
    assert torch.equal(cw[:, :code.k], u)
    assert bool(code.syndrome_ok(cw).all())
    llr = (1 - 2 * cw.float()) * 10.0
    bits, conv, it = code.decode(llr)
    assert torch.equal(bits, cw) and bool(conv.all()) and int(it.max()) == 0
    # flip 20 bits per word: must be corrected
    llr2 = llr.clone()
    for r in range(64):
        pos = torch.randperm(1008, device=device)[:20]
        llr2[r, pos] *= -1
    bits, conv, it = code.decode(llr2)
    assert torch.equal(bits, cw) and bool(conv.all())


def test_min_sum_decoder(code, device):
    code_ms = LDPC(n=1008, dv=3, dc=6, k=504, seed=1234, cache_dir=None, device=device, decoder="min_sum")
    u = torch.randint(0, 2, (32, code.k), dtype=torch.uint8, device=device)
    cw = code_ms.encode(u)
    llr = (1 - 2 * cw.float()) * 6.0 + 2.0 * torch.randn_like(cw.float())
    bits, conv, _ = code_ms.decode(llr)
    assert bool(conv.all()) and torch.equal(bits, cw)


@pytest.mark.parametrize("ebn0_db,lo,hi", [(2.0, 3e-4, 3e-3), (3.0, 0.0, 1.5e-4)])
def test_v40_awgn_ber(code, device, ebn0_db, lo, hi):
    torch.manual_seed(7)
    R = code.k / code.n
    sigma2 = 1.0 / (2 * R * 10 ** (ebn0_db / 10))   # BPSK a=1: Eb/N0 = a^2/(2 R sigma^2)
    n_blocks = 2000
    u = torch.randint(0, 2, (n_blocks, code.k), dtype=torch.uint8, device=device)
    cw = code.encode(u)
    y = (1 - 2 * cw.float()) + math.sqrt(sigma2) * torch.randn(n_blocks, code.n, device=device)
    bits, conv, it = code.decode(2 * y / sigma2)
    ber = float((bits[:, :code.k] != u).float().mean())
    print(f"Eb/N0={ebn0_db} dB: BER={ber:.2e} FER={float((~conv).float().mean()):.3f} iters={float(it.float().mean()):.1f}")
    assert lo <= ber <= hi, ber


def test_framing_roundtrip(device):
    bits = torch.randint(0, 2, (5, 1792), dtype=torch.uint8, device=device)
    blocks, nblk, npad = frame(bits, 504)
    assert nblk == 4 and npad == 4 * 504 - 1792 and blocks.shape == (20, 504)
    assert torch.equal(unframe(blocks, 5, 1792), bits)
