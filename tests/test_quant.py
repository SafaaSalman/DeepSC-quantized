import torch

from quantized_deepsc.quant import ScalarQuantizer, gray, ungray


def test_gray_roundtrip_all_b():
    for b in (2, 3, 4, 5, 6, 8):
        idx = torch.arange(2 ** b)
        g = gray(idx)
        assert torch.equal(ungray(g), idx)
        # adjacent indices differ in exactly one Gray bit
        diff = (g[1:] ^ g[:-1])
        assert all(bin(int(d)).count("1") == 1 for d in diff)


def test_bits_roundtrip_and_labels():
    Q = ScalarQuantizer(4.54, 4)
    idx = torch.arange(16)
    bits = Q.to_bits(idx)
    assert bits.shape == (16, 4)
    assert torch.equal(Q.from_bits(bits), idx)
    labels = Q.bit_labels()
    assert torch.equal(labels, bits)
    assert labels[0].tolist() == [0, 0, 0, 0] and labels[1].tolist() == [0, 0, 0, 1]


def test_quantize_bounds_and_error():
    Q = ScalarQuantizer(4.54, 4)
    t = torch.linspace(-20, 20, 100001)
    idx = Q.quantize(t)
    assert idx.min() == 0 and idx.max() == 15
    inside = t.abs() <= Q.A
    err = (Q.dequantize(idx) - t).abs()
    assert err[inside].max() <= Q.delta / 2 + 1e-5
    assert torch.equal(Q.quantize(Q.levels()), torch.arange(16))
    assert abs(Q.q_rms - Q.delta / 12 ** 0.5) < 1e-9


def test_quantizer_on_batch_shape():
    Q = ScalarQuantizer(4.0, 4)
    t = torch.randn(3, 5, 16)
    idx = Q.quantize(t)
    bits = Q.to_bits(idx)
    assert bits.shape == (3, 5, 16, 4)
    assert torch.equal(Q.from_bits(bits), idx)
