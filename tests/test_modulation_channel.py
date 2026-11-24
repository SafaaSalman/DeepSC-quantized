import math

import torch
from scipy.stats import norm

from quantized_deepsc.channel import awgn, rayleigh_zf
from quantized_deepsc.modulation import PAM
from quantized_deepsc.quant import ScalarQuantizer


def snr_to_sigma(L):  # repo convention
    return 1.0 / math.sqrt(2 * 10 ** (L / 10))


def Qf(x):
    return norm.sf(x)


def test_bpsk_mapping_and_llr_sign(device):
    pam = PAM(2, device)
    bits = torch.tensor([[0, 1, 0, 0, 1, 1]], dtype=torch.uint8, device=device)
    x = pam.modulate(bits)
    assert torch.equal(x, torch.tensor([[1., -1., 1., 1., -1., -1.]], device=device))
    assert torch.equal(pam.hard_demod(x), bits)
    llr = pam.llr_maxlog(x, 0.25, None)
    assert torch.all((llr > 0) == (bits == 0))
    # BPSK LLR equals 2*h2*y/sigma^2
    y = torch.randn(1, 6, device=device)
    h2 = torch.rand(1, 1, device=device) + 0.1
    ref = 2 * h2 * y / 0.25
    assert torch.allclose(pam.llr_maxlog(y, 0.25, h2), ref, atol=1e-5)


def test_pam16_unit_energy_gray_and_roundtrip(device):
    pam = PAM(16, device)
    assert abs(float((pam.levels ** 2).mean()) - 1.0) < 1e-6
    lab = pam.labels
    for k in range(15):  # neighbouring levels differ in one bit
        assert int((lab[k] ^ lab[k + 1]).sum()) == 1
    Q = ScalarQuantizer(4.54, 4)
    assert torch.equal(pam.labels.cpu(), Q.bit_labels())  # latent level k <-> PAM symbol k
    bits = torch.randint(0, 2, (7, 4 * 50), dtype=torch.uint8, device=device)
    x = pam.modulate(bits)
    assert x.shape == (7, 50)
    assert torch.equal(pam.hard_demod(x), bits)
    llr = pam.llr_maxlog(x + 0.01 * torch.randn_like(x), 0.01, None)
    assert torch.all((llr > 0) == (bits == 0))


def test_rayleigh_zf_shapes_and_noiseless(device):
    x = torch.randn(5, 40, device=device)
    y, h2 = rayleigh_zf(x, 0.0, "sentence")
    assert y.shape == x.shape and h2.shape == (5, 1)
    assert torch.allclose(y, x, atol=1e-4)
    y, h2 = rayleigh_zf(x, 0.0, "symbol")
    assert h2.shape == (5, 40)
    assert torch.equal(h2[:, ::2], h2[:, 1::2])
    y, h2 = rayleigh_zf(x, 0.0, "batch")
    assert h2.shape == (1, 1)
    # E|h|^2 = 1
    _, h2 = rayleigh_zf(torch.zeros(20000, 2, device=device), 0.0, "sentence")
    assert abs(float(h2.mean()) - 1.0) < 0.03


def test_v30_bpsk_ber_closed_forms(device):
    torch.manual_seed(0)
    pam = PAM(2, device)
    N = 2_000_000
    bits = torch.randint(0, 2, (1, N), dtype=torch.uint8, device=device)
    x = pam.modulate(bits)
    for L in (0.0, 3.0, 6.0):
        s = snr_to_sigma(L)
        ber = float((pam.hard_demod(awgn(x, s)) != bits).float().mean())
        ref = Qf(1 / s)
        assert abs(ber - ref) / ref < 0.05, (L, ber, ref)
    for L in (0.0, 6.0, 12.0):
        s = snr_to_sigma(L)
        y, _ = rayleigh_zf(x.reshape(-1, 2), s, "symbol")
        ber = float((pam.hard_demod(y.reshape(1, -1)) != bits).float().mean())
        g = 10 ** (L / 10)
        ref = 0.5 * (1 - math.sqrt(g / (1 + g)))
        assert abs(ber - ref) / ref < 0.05, (L, ber, ref)


def test_v30_pam16_ser_closed_form(device):
    torch.manual_seed(1)
    pam = PAM(16, device)
    N = 1_000_000
    bits = torch.randint(0, 2, (1, 4 * N), dtype=torch.uint8, device=device)
    x = pam.modulate(bits)
    k = pam.nearest_index(x)
    for L in (18.0, 24.0):
        s = snr_to_sigma(L)
        ser = float((pam.nearest_index(awgn(x, s)) != k).float().mean())
        ref = 2 * (1 - 1 / 16) * Qf(pam.d / s)
        assert abs(ser - ref) / ref < 0.05, (L, ser, ref)
