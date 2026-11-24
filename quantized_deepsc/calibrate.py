"""Stage 1: extract the transmitted latent t on the test set and calibrate the quantiser.

    python -m quantized_deepsc.calibrate [--subset N] [--b 4] [--percentile 99.9]

Writes results/s1/calibration.json (A, Delta, histogram, entropy, bit marginals for the
transmitted tensor and for the non-pad positions) and results/s1/latent_hist.png.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .deepsc_io import DeepSCLink, load_config, resolve
from .quant import ScalarQuantizer


def latent_stats(t: np.ndarray, b: int, percentile: float, A: float | None = None) -> dict:
    A = float(np.percentile(np.abs(t), percentile)) if A is None else float(A)
    Q = ScalarQuantizer(A, b)
    idx = Q.quantize(torch.from_numpy(t)).numpy()
    hist = np.bincount(idx, minlength=Q.M) / idx.size
    nz = hist[hist > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    bits = Q.to_bits(torch.from_numpy(idx)).numpy()           # [N, b] MSB first
    return {
        "n_values": int(t.size),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "rms": float(np.sqrt((t ** 2).mean())),
        "A": A,
        "delta": Q.delta,
        "q_rms": Q.q_rms,
        "clipped_fraction": float((np.abs(t) > A).mean()),
        "level_hist": hist.tolist(),
        "index_entropy_bits": entropy,
        "mass_levels_7_8": float(hist[Q.M // 2 - 1] + hist[Q.M // 2]) if Q.M >= 4 else None,
        "p_bit_is_1_msb_first": bits.mean(0).tolist(),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--subset", default=None, help="int or 'full' (default: full test set)")
    p.add_argument("--b", type=int, default=None)
    p.add_argument("--percentile", type=float, default=None)
    p.add_argument("--out", default=None, help="JSON path (default: config calibration_file)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    b = args.b or int(cfg["b"])
    pct = args.percentile or float(cfg["percentile"])
    subset = None if args.subset in (None, "full") else int(args.subset)
    out = resolve(args.out or cfg["calibration_file"])
    out.parent.mkdir(parents=True, exist_ok=True)

    link = DeepSCLink(cfg)
    t0 = time.time()
    t_all, t_np, n_sent = [], [], 0
    for sents in link.test_loader(subset):
        t, pad = link.encode(sents)
        t_all.append(t.reshape(-1).cpu().numpy())
        t_np.append(t[~pad].reshape(-1).cpu().numpy())
        n_sent += sents.shape[0]
    t_all = np.concatenate(t_all).astype(np.float32)
    t_np = np.concatenate(t_np).astype(np.float32)
    print(f"extracted {n_sent} sentences, {t_all.size} values ({t_np.size} non-pad) in {time.time() - t0:.1f}s")

    st_all = latent_stats(t_all, b, pct)
    st_np = latent_stats(t_np, b, pct)
    st_np_sameA = latent_stats(t_np, b, pct, A=st_all["A"])
    res = {
        "checkpoint": str(cfg["checkpoint"]),   # relative to repo_dir, as in config.yaml
        "ckpt_sha256": link.ckpt_sha256,
        "batch_size": int(cfg["batch_size"]),
        "n_sentences": n_sent,
        "b": b,
        "percentile": pct,
        "A": st_all["A"],
        "delta": st_all["delta"],
        "pad_fraction": 1.0 - t_np.size / t_all.size,
        "all_positions": st_all,
        "nonpad_positions_own_A": st_np,
        "nonpad_positions_with_A_all": st_np_sameA,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in st_all.items() if k != "level_hist"}, indent=2))
    print(f"wrote {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Q = ScalarQuantizer(st_all["A"], b)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(t_all, bins=400, density=True, color="#8fb3d9", label="t (transmitted, pads incl.)")
        ax.hist(t_np, bins=400, density=True, histtype="step", color="#d9534f", label="t (non-pad only)")
        for v in Q.levels().numpy():
            ax.axvline(v, color="k", lw=0.5, alpha=0.5)
        ax.set_yscale("log")
        ax.set_xlabel("latent value t")
        ax.set_ylabel("density")
        ax.set_title(f"DeepSC latent, A={Q.A:.3f}, Delta={Q.delta:.3f}, b={b}")
        ax.legend()
        fig.tight_layout()
        png = out.parent / "latent_hist.png"
        fig.savefig(png, dpi=150)
        print(f"wrote {png}")
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
