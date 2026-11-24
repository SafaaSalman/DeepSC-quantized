"""Figures and numbers for the GRAND report (REPORT_GRAND.md).

    python -m quantized_deepsc.plot_grand --runs results/s0 results/s3 results/s4 results/s6 --out results/s7

Compares four links on BLEU-1: analog DeepSC (c0), uncoded digital (u_bpsk), digital with a
short CRC code decoded by ORBGRAND (g_bpsk) and digital with the rate-1/2 LDPC code (c_bpsk).

  grand_bleu1.png    BLEU-1 versus SNR label, Rayleigh (one gain per sentence) and AWGN
  grand_gain.png     BLEU-1 gain of g_bpsk over u_bpsk: at the same SNR per channel use, and at
                     the same energy per token (the uncoded link is read at L + 10 log10(uses_g / uses_u))
  grand_blocks.png   share of code blocks ORBGRAND abandons, versus SNR
  grand_numbers.json every number quoted in the report
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from .deepsc_io import resolve
from .run_compare import curve, load_runs, threshold

# categorical slots in their validated order; colour follows the link in every figure
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LINKS = [  # variant, label, colour, marker
    ("c0", "Analog DeepSC", SECONDARY, "o"),
    ("u_bpsk", "Digital, uncoded BPSK", "#2a78d6", "s"),
    ("g_bpsk", "Digital + CRC(128,112) + ORBGRAND", "#eb6834", "D"),
    ("c_bpsk", "Digital + LDPC rate 1/2", "#1baf7a", "^"),
    ("u_pam16", "Digital, uncoded 16-PAM", "#eda100", "v"),   # same channel uses per token as analog
]
GRAND_COLOUR = "#eb6834"
PANELS = [("rayleigh", "sentence", "Rayleigh fading, one gain per sentence"), ("awgn", "-", "AWGN")]
COARSE = [-3, 0, 3, 6, 9, 12, 15, 18, 24]


def style_axis(ax, title, xlabel):
    ax.set_title(title, loc="left", color=INK, fontsize=10.5, fontweight="semibold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(AXIS)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["results/s0", "results/s3", "results/s4", "results/s6"])
    p.add_argument("--out", default="results/s7")
    p.add_argument("--metric", default="bleu1")
    args = p.parse_args(argv)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs([resolve(d) for d in args.runs if resolve(d).exists()])
    m = args.metric

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    xlabel = "SNR label per channel use (dB)"
    numbers = {"metric": m, "channels": {}}
    curves = {(ch, v): curve(df, v, ch, hd) for ch, hd, _ in PANELS for v, *_ in LINKS}
    q0 = float(curves[("awgn", "u_bpsk")][m].max())  # all bits correct: the quantisation-only value
    numbers["digital_ceiling"] = q0

    # ------------------------------------------------------------------ figure 1: BLEU versus SNR
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True, facecolor=SURFACE)
    for ax, (ch, hd, title) in zip(axes, PANELS):
        for v, label, colour, marker in LINKS:
            t = curves[(ch, v)]
            marks = [i for i, s in enumerate(t.snr_db) if float(s) % 3 == 0]
            ax.plot(t.snr_db, t[m], color=colour, lw=2, marker=marker, ms=6.5, markevery=marks, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, solid_capstyle="round", zorder=3 if v == "g_bpsk" else 2,
                    label=f"{label} ({t.uses_per_token_all.mean():.0f} channel uses per token)")
        style_axis(ax, title, xlabel)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(-3, 25, 3))
    axes[0].set_ylabel("BLEU-1", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=SECONDARY,
               handlelength=3, columnspacing=3)
    fig.suptitle("BLEU-1 versus SNR on 5,000 test sentences: analog DeepSC and four digital links",
                 x=0.012, ha="left", color=INK, fontsize=11.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(out / "grand_bleu1.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------------- figure 2: gain of GRAND over the uncoded link
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True, facecolor=SURFACE)
    for ax, (ch, hd, title) in zip(axes, PANELS):
        g, u = curves[(ch, "g_bpsk")], curves[(ch, "u_bpsk")]
        x = g.snr_db.values.astype(float)
        ux, uy = u.snr_db.values.astype(float), u[m].values.astype(float)
        shift = 10 * math.log10(g.uses_per_token_all.mean() / u.uses_per_token_all.mean())
        same_snr = g[m].values - np.interp(x, ux, uy)
        ok = x + shift <= ux.max()
        same_energy = g[m].values[ok] - np.interp(x[ok] + shift, ux, uy)
        ax.axhline(0, color=AXIS, lw=1)
        ax.plot(x, same_snr, color=GRAND_COLOUR, lw=2, marker="D", ms=6, markevery=3, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label="Same SNR per channel use")
        ax.plot(x[ok], same_energy, color=GRAND_COLOUR, lw=2, ls=(0, (4, 2)), marker="o", ms=6, markevery=3,
                markerfacecolor=SURFACE, markeredgewidth=1.5,
                label=f"Same energy per token (uncoded link given {shift:.2f} dB more SNR)")
        style_axis(ax, title, xlabel)
        ax.set_xticks(range(-3, 25, 3))
        i = int(np.argmax(same_snr))
        ax.annotate(f"+{same_snr[i]:.3f} at {x[i]:g} dB", (x[i], same_snr[i]), textcoords="offset points",
                    xytext=(8, 4), color=SECONDARY, fontsize=8.5)
        numbers["channels"][ch] = {
            "title": title, "energy_shift_db": shift,
            "gain_same_snr": {f"{s:g}": float(v) for s, v in zip(x, same_snr)},
            "gain_same_energy": {f"{s:g}": float(v) for s, v in zip(x[ok], same_energy)},
            "best_gain_same_snr": {"snr_db": float(x[i]), "gain": float(same_snr[i])},
            "best_gain_same_energy": {"snr_db": float(x[ok][int(np.argmax(same_energy))]), "gain": float(same_energy.max())},
            "worst_gain_same_energy": {"snr_db": float(x[ok][int(np.argmin(same_energy))]), "gain": float(same_energy.min())},
        }
    axes[0].set_ylabel("BLEU-1 gain over the uncoded link", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=SECONDARY,
               handlelength=3.4, columnspacing=3)
    fig.suptitle("What the CRC + ORBGRAND code adds to the uncoded digital link", x=0.012, ha="left", color=INK,
                 fontsize=11.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out / "grand_gain.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)

    # --------------------------------------------------------- figure 3: blocks abandoned by ORBGRAND
    fig, ax = plt.subplots(figsize=(7.2, 4.3), facecolor=SURFACE)
    for (ch, hd, title), ls, marker, face in zip(PANELS, ["-", (0, (4, 2))], ["D", "o"], [GRAND_COLOUR, SURFACE]):
        g = curves[(ch, "g_bpsk")]
        t = g[g.abandon_rate > 0]
        marks = [i for i, s in enumerate(t.snr_db) if float(s) % 3 == 0 or i == len(t) - 1]
        ax.plot(t.snr_db, t.abandon_rate, color=GRAND_COLOUR, lw=2, ls=ls, marker=marker, ms=6, markevery=marks,
                markerfacecolor=face, markeredgecolor=SURFACE if face != SURFACE else GRAND_COLOUR, markeredgewidth=1.5,
                label=title)
        last = t.iloc[-1]
        ax.annotate(title.split(",")[0], (last.snr_db, last.abandon_rate), textcoords="offset points", xytext=(8, -3),
                    color=SECONDARY, fontsize=8.5)
    ax.set_yscale("log")
    style_axis(ax, "Share of 128-bit blocks ORBGRAND gives up on (budget 2,000 queries)", xlabel)
    ax.tick_params(which="both", colors=MUTED, length=0)
    ax.grid(color=GRID, lw=0.8, which="major")
    ax.set_ylabel("Blocks abandoned", color=MUTED)
    ax.set_xticks(range(-3, 28, 3))
    ax.set_xlim(-4, 29)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY, loc="upper right", handlelength=3.4)
    fig.tight_layout()
    fig.savefig(out / "grand_blocks.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------------------------------------------- numbers
    for ch, hd, title in PANELS:
        c = numbers["channels"][ch]
        c["bleu"], c["uses_per_token"], c["thresholds"] = {}, {}, {}
        for v, label, *_ in LINKS:
            t = curves[(ch, v)]
            x, y = t.snr_db.values.astype(float), t[m].values.astype(float)
            c["bleu"][v] = {f"{s:g}": float(b) for s, b in zip(x, y) if s in COARSE or s in (2, 4, 5)}
            c["uses_per_token"][v] = float(t.uses_per_token_all.mean())
            c["thresholds"][v] = {"max": float(y.max()), "snr_at_0.9_ceiling": threshold(x, y, 0.9 * q0),
                                  "snr_at_0.95_ceiling": threshold(x, y, 0.95 * q0)}
        g = curves[(ch, "g_bpsk")]
        u = curves[(ch, "u_bpsk")]
        c["blocks"] = {f"{r.snr_db:g}": {"abandoned": float(r.abandon_rate), "miscorrected": float(r.miscorrect_rate),
                                         "block_error": float(r.fer), "queries_mean": float(r.queries_mean),
                                         "ber_after": float(r.ber), "time_s": float(r.time_s)}
                       for r in g.itertuples() if r.snr_db in COARSE or r.snr_db in (2, 4, 5)}
        c["ber_uncoded"] = {f"{r.snr_db:g}": float(r.ber) for r in u.itertuples() if r.snr_db in COARSE or r.snr_db in (2, 4, 5)}
        c["n_sentences"] = int(g.n_sentences.iloc[0])
    with open(out / "grand_numbers.json", "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=1)
    print(json.dumps(numbers, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
