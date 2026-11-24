"""Per-stage figures from the run CSVs (Stage 5 figures are made by run_compare).

    python -m quantized_deepsc.plot_stages [--results results]

Writes results/s0/s0_baseline.png, results/s2/s2_bits_vs_bleu.png, results/s3/s3_uncoded_{bleu,ber}.png,
results/s4/s4_coded_{bleu,ber_fer}.png (whatever data exist).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .deepsc_io import resolve  # noqa: E402

COL = {"c0": "k", "u_bpsk": "#1f77b4", "u_pam16": "#2ca02c", "c_bpsk": "#d62728", "c_pam16": "#ff7f0e"}


def read(path):
    return pd.read_csv(path, comment="#").sort_values("snr_db")


def stage0(root: Path):
    d = root / "s0"
    runs = [("c0_rayleigh_repo.csv", "Rayleigh, h per batch (repo original)", "--", "gray"),
            ("c0_rayleigh_sentence.csv", "Rayleigh, h per sentence", "-", "k"),
            ("c0_awgn_-.csv", "AWGN", "-", "#1f77b4")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_ = False
    for fn, lab, ls, c in runs:
        if (d / fn).exists():
            df = read(d / fn)
            ax.plot(df.snr_db, df.bleu1, ls=ls, color=c, marker="o", ms=3, label=lab)
            any_ = True
    if (d / "c0_none_-.csv").exists():
        v = float(read(d / "c0_none_-.csv").bleu1.iloc[0])
        ax.axhline(v, color="k", ls=":", lw=0.8, label=f"noiseless = {v:.3f}")
    if not any_:
        return
    ax.set_xlabel("SNR label L (dB)")
    ax.set_ylabel("BLEU-1")
    ax.set_title("Stage 0: analog DeepSC baseline (checkpoint_80)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "s0_baseline.png", dpi=150)
    plt.close(fig)


def stage2(root: Path):
    d = root / "s2"
    rows = []
    for f in sorted(d.glob("q0_none_-*.csv")):
        df = pd.read_csv(f, comment="#")
        rows.append({"b": int(df.b.iloc[0]), "bleu1": float(df.bleu1.iloc[0]), "bleu4": float(df.bleu4.iloc[0]),
                     "mse": float(df.mse_nonpad.iloc[0])})
    if not rows:
        return
    t = pd.DataFrame(rows).sort_values("b")
    t.to_csv(d / "s2_bits_vs_bleu.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(t.b, t.bleu1, "o-", color="k", label="BLEU-1")
    ax.plot(t.b, t.bleu4, "s--", color="gray", label="BLEU-4")
    s0 = root / "s0" / "c0_none_-.csv"
    if s0.exists():
        v = float(read(s0).bleu1.iloc[0])
        ax.axhline(v, color="#1f77b4", ls=":", label=f"analog noiseless BLEU-1 = {v:.3f}")
    ax2 = ax.twinx()
    ax2.semilogy(t.b, t.mse, "^:", color="#d62728", label="latent MSE (right axis)")
    ax2.set_ylabel("quantisation MSE")
    ax.set_xlabel("bits per latent value b")
    ax.set_ylabel("BLEU")
    ax.set_title("Stage 2: quantisation-only round trip (no channel)")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(d / "s2_bits_vs_bleu.png", dpi=150)
    plt.close(fig)


def stage34(root: Path, stage: str, variants):
    d = root / stage
    files = {(v, ch): d / f"{v}_{ch}_{'sentence' if ch == 'rayleigh' else '-'}.csv" for v in variants for ch in ("rayleigh", "awgn")}
    files = {k: f for k, f in files.items() if f.exists()}
    if not files:
        return
    s0 = root / "s0"
    q0 = root / "s2" / "q0_none_-.csv"
    q0v = float(pd.read_csv(q0, comment="#").bleu1.iloc[0]) if q0.exists() else None
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for (v, ch), f in files.items():
        df = read(f)
        ax.plot(df.snr_db, df.bleu1, ls="-" if ch == "rayleigh" else "--", color=COL[v], marker="o", ms=3,
                label=f"{v} {ch} ({df.uses_per_token_all.mean():.0f} uses/token)")
    for ch, fn, ls in (("rayleigh", "c0_rayleigh_sentence.csv", "-"), ("awgn", "c0_awgn_-.csv", "--")):
        if (s0 / fn).exists():
            df = read(s0 / fn)
            ax.plot(df.snr_db, df.bleu1, ls=ls, color="k", lw=1, alpha=0.7, label=f"c0 analog {ch}")
    if q0v is not None:
        ax.axhline(q0v, color="gray", ls=":", label=f"Q0 ceiling = {q0v:.3f}")
    ax.set_xlabel("SNR label L (dB)")
    ax.set_ylabel("BLEU-1")
    ax.set_title(f"Stage {stage[1]}: {'uncoded' if stage == 's3' else 'LDPC-coded'} digital link")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(d / f"{stage}_{'uncoded' if stage == 's3' else 'coded'}_bleu.png", dpi=150)
    plt.close(fig)

    ncol = 2 if stage == "s4" else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7.5 * ncol / 1.4, 4.5), squeeze=False)
    for (v, ch), f in files.items():
        df = read(f)
        ls = "-" if ch == "rayleigh" else "--"
        axes[0, 0].semilogy(df.snr_db, df.ber.clip(lower=1e-7), ls=ls, color=COL[v], marker="o", ms=3, label=f"{v} {ch}")
        if stage == "s4":
            axes[0, 1].semilogy(df.snr_db, df.fer.clip(lower=1e-7), ls=ls, color=COL[v], marker="o", ms=3, label=f"{v} {ch}")
    axes[0, 0].set_ylabel("BER" + (" (info bits, post-BP)" if stage == "s4" else ""))
    if stage == "s4":
        axes[0, 1].set_ylabel("FER (LDPC blocks)")
    for a in axes[0]:
        a.set_xlabel("SNR label L (dB)")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7)
    fig.suptitle(f"Stage {stage[1]}: {'uncoded' if stage == 's3' else 'LDPC-coded'} link, error rates")
    fig.tight_layout()
    fig.savefig(d / f"{stage}_{'uncoded_ber' if stage == 's3' else 'coded_ber_fer'}.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    args = p.parse_args(argv)
    root = resolve(args.results)
    stage0(root)
    stage2(root)
    stage34(root, "s3", ["u_bpsk", "u_pam16"])
    stage34(root, "s4", ["c_bpsk", "c_pam16"])
    print("figures written under", root)


if __name__ == "__main__":
    main()
