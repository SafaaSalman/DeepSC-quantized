"""Stage 5: compare the digital variants with the analog link under three fairness views.

    python -m quantized_deepsc.run_compare --runs results/s0 results/s3 results/s4 --q0 results/s2 --out results/s5

View 1  equal SNR per channel use     x = SNR label L
View 2  equal energy per token        x = L + 10*log10(uses_per_token / 16)   (digital curves move right)
View 3  equal channel uses            U-PAM16 vs C0 only (16 real uses per token in both)

Writes one CSV + PNG per view, s5_crossover.json, s5_thresholds.json and summary.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .deepsc_io import resolve

ANALOG_USES = 16.0
LABELS = {"c0": "C0 analog DeepSC", "q0": "Q0 quantisation-only ceiling", "u_bpsk": "U-BPSK (uncoded, hard)",
          "u_pam16": "U-PAM16 (uncoded, hard)", "c_bpsk": "C-BPSK (LDPC 1/2)", "c_pam16": "C-PAM16 (LDPC 1/2)",
          "g_bpsk": "G-BPSK (CRC + ORBGRAND)", "g_pam16": "G-PAM16 (CRC + ORBGRAND)"}
STYLE = {"c0": ("k", "-", "o"), "u_bpsk": ("#1f77b4", "--", "s"), "u_pam16": ("#2ca02c", "--", "^"),
         "c_bpsk": ("#d62728", "-", "s"), "c_pam16": ("#ff7f0e", "-", "^"),
         "g_bpsk": ("#9467bd", "-.", "s"), "g_pam16": ("#8c564b", "-.", "^")}
DIGITAL = ["u_bpsk", "u_pam16", "c_bpsk", "c_pam16", "g_bpsk", "g_pam16"]


def load_runs(dirs) -> pd.DataFrame:
    frames = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.csv")):
            df = pd.read_csv(f, comment="#")
            df["file"] = f.name
            frames.append(df)
    if not frames:
        raise SystemExit(f"no CSV files found in {dirs}")
    return pd.concat(frames, ignore_index=True)


def curve(df, variant, channel, h_draw=None) -> pd.DataFrame:
    sel = df[(df.variant == variant) & (df.channel == channel)]
    if h_draw is not None:
        sel = sel[sel.h_draw == h_draw]
    return sel.sort_values("snr_db").drop_duplicates("snr_db", keep="last")


def crossover(x_d, y_d, x_c, y_c):
    """First x where the digital curve rises above the analog curve (linear interpolation).
    Returns (x, status): status in {'crosses', 'always_above', 'never', 'no_overlap'}."""
    lo, hi = max(x_d.min(), x_c.min()), min(x_d.max(), x_c.max())
    if hi <= lo:
        return None, "no_overlap"
    g = np.arange(lo, hi + 1e-9, 0.05)
    diff = np.interp(g, x_d, y_d) - np.interp(g, x_c, y_c)
    if (diff >= 0).all():
        return float(lo), "always_above"
    if (diff < 0).all():
        return None, "never"
    idx = np.nonzero((diff[:-1] < 0) & (diff[1:] >= 0))[0]
    if idx.size == 0:  # digital starts above and drops below: report the last point above
        return None, "never"
    i = int(idx[-1])
    x0, x1, d0, d1 = g[i], g[i + 1], diff[i], diff[i + 1]
    return float(x0 + (x1 - x0) * (-d0) / (d1 - d0)), "crosses"


def threshold(x, y, level):
    """Smallest x where y >= level (interpolated); None if never."""
    if (y >= level).sum() == 0:
        return None
    g = np.arange(x.min(), x.max() + 1e-9, 0.05)
    yy = np.interp(g, x, y)
    i = int(np.argmax(yy >= level))
    return float(g[i])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["results/s0", "results/s3", "results/s4"])
    p.add_argument("--q0", nargs="*", default=["results/s2"])
    p.add_argument("--h-draw", default="sentence", help="analog/digital h draw used for the comparison")
    p.add_argument("--out", default="results/s5")
    p.add_argument("--metric", default="bleu1")
    args = p.parse_args(argv)

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs([resolve(d) for d in args.runs if resolve(d).exists()])
    q0_df = load_runs([resolve(d) for d in args.q0 if resolve(d).exists()]) if args.q0 else pd.DataFrame()
    m = args.metric

    q0 = None
    if len(q0_df):
        q = q0_df[q0_df.variant == "q0"]
        if "b" in q.columns:
            q = q[q.b == 4] if (q.b == 4).any() else q
        if len(q):
            q0 = float(q[m].iloc[-1])
    if q0 is None:
        q = df[df.variant == "q0"]
        q0 = float(q[m].iloc[-1]) if len(q) else None

    c0 = curve(df, "c0", "rayleigh", args.h_draw)
    if not len(c0):
        raise SystemExit("no analog (c0, rayleigh) curve found")
    c0_none = df[(df.variant == "c0") & (df.channel == "none")]
    c_inf = float(c0_none[m].iloc[-1]) if len(c0_none) else None

    views = {1: "equal SNR per channel use", 2: "equal energy per token", 3: "equal channel uses"}
    cross, thresh, tables = {}, {}, {}
    for v, title in views.items():
        rows = []
        variants = ["u_pam16"] if v == 3 else DIGITAL
        cx, cy = c0.snr_db.values.astype(float), c0[m].values.astype(float)
        for _, r in c0.iterrows():
            rows.append({"variant": "c0", "x_snr_db": float(r.snr_db), "snr_label_db": float(r.snr_db),
                         "uses_per_token": ANALOG_USES, **{k: r[k] for k in ("bleu1", "bleu2", "bleu3", "bleu4") if k in r}})
        for var in variants:
            d = curve(df, var, "rayleigh", args.h_draw)
            if not len(d):
                continue
            uses = d.uses_per_token_all.values.astype(float)
            shift = 10 * np.log10(uses / ANALOG_USES) if v == 2 else np.zeros(len(d))
            x = d.snr_db.values.astype(float) + shift
            y = d[m].values.astype(float)
            for i, (_, r) in enumerate(d.iterrows()):
                rows.append({"variant": var, "x_snr_db": float(x[i]), "snr_label_db": float(r.snr_db),
                             "uses_per_token": float(uses[i]), **{k: r[k] for k in ("bleu1", "bleu2", "bleu3", "bleu4") if k in r}})
            xc, status = crossover(x, y, cx, cy)
            cross.setdefault(f"view{v}", {})[var] = {"crossover_snr_db": xc, "status": status,
                                                    "uses_per_token": float(np.mean(uses)),
                                                    "shift_db": float(np.mean(shift))}
            if v == 1:
                thresh[var] = {"snr_at_0.9_q0": threshold(x, y, 0.9 * q0) if q0 else None,
                               "snr_at_0.95_q0": threshold(x, y, 0.95 * q0) if q0 else None,
                               "bleu1_max": float(y.max())}
        tab = pd.DataFrame(rows)
        tab.to_csv(out / f"s5_{m}_vs_snr_view{v}.csv", index=False)
        tables[v] = tab
    if q0 is not None:
        thresh["c0"] = {"snr_at_0.9_q0": threshold(cx, cy, 0.9 * q0), "snr_at_0.95_q0": threshold(cx, cy, 0.95 * q0),
                        "bleu1_max": float(cy.max())}
    meta = {"metric": m, "q0": q0, "c_inf_noiseless_analog": c_inf, "h_draw": args.h_draw,
            "snr_convention": "label L: sigma = 1/sqrt(2*10^(L/10)) per real dim, unit-energy constellations, Es/sigma^2 = L+3 dB",
            "view2_shift": "x = L + 10*log10(uses_per_token/16)"}
    with open(out / "s5_crossover.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "crossover": cross}, f, indent=2)
    with open(out / "s5_thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "thresholds": thresh}, f, indent=2)

    # ---- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable, no figures:", e)
        plt = None
    if plt is not None:
        xlab = {1: "SNR label L per channel use (dB)", 2: "SNR per token, L + 10 log10(uses/16) (dB)", 3: "SNR label L per channel use (dB)"}
        for v, tab in tables.items():
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            for var in ["c0"] + (["u_pam16"] if v == 3 else DIGITAL):
                t = tab[tab.variant == var].sort_values("x_snr_db")
                if not len(t):
                    continue
                c, ls, mk = STYLE[var]
                lab = LABELS[var] + ("" if var == "c0" else f", {t.uses_per_token.mean():.0f} uses/token")
                ax.plot(t.x_snr_db, t[m], color=c, ls=ls, marker=mk, ms=4, lw=1.4, label=lab)
            if q0 is not None:
                ax.axhline(q0, color="gray", ls=":", lw=1.2, label=f"Q0 ceiling (b=4) = {q0:.3f}")
            if c_inf is not None:
                ax.axhline(c_inf, color="k", ls=":", lw=0.8, alpha=0.6, label=f"analog, noiseless = {c_inf:.3f}")
            ax.set_xlabel(xlab[v])
            ax.set_ylabel(m.upper().replace("BLEU", "BLEU-"))
            ax.set_title(f"View {v}: {views[v]} (Rayleigh, ZF, h per {args.h_draw})")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
            fig.tight_layout()
            fig.savefig(out / f"s5_{m}_vs_snr_view{v}.png", dpi=150)
            plt.close(fig)

        # AWGN, view 1 only, if present
        aw = df[df.channel == "awgn"]
        if len(aw):
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            for var in ["c0"] + DIGITAL:
                t = curve(aw, var, "awgn")
                if not len(t):
                    continue
                c, ls, mk = STYLE[var]
                ax.plot(t.snr_db, t[m], color=c, ls=ls, marker=mk, ms=4, lw=1.4,
                        label=LABELS[var] + ("" if var == "c0" else f", {t.uses_per_token_all.mean():.0f} uses/token"))
            if q0 is not None:
                ax.axhline(q0, color="gray", ls=":", lw=1.2, label=f"Q0 ceiling = {q0:.3f}")
            ax.set_xlabel("SNR label L per channel use (dB)")
            ax.set_ylabel(m.upper().replace("BLEU", "BLEU-"))
            ax.set_title("AWGN, equal SNR per channel use")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
            fig.tight_layout()
            fig.savefig(out / f"s5_{m}_vs_snr_awgn.png", dpi=150)
            plt.close(fig)

        # latent MSE -> BLEU transfer scatter, pooled over all runs
        if "mse_nonpad" in df.columns:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for var in ["c0"] + DIGITAL:
                t = df[(df.variant == var) & (df.channel != "none")]
                if not len(t):
                    continue
                c, ls, mk = STYLE[var]
                ax.scatter(t.mse_nonpad, t[m], color=c, marker=mk, s=18, label=LABELS[var], alpha=0.8)
            if q0 is not None:
                ax.axhline(q0, color="gray", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("latent MSE at the decoder input (non-pad positions)")
            ax.set_ylabel(m.upper().replace("BLEU", "BLEU-"))
            ax.set_title("Interface noise -> semantic fidelity (all channels, all SNRs)")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "s5_mse_to_bleu.png", dpi=150)
            plt.close(fig)

    # ---- summary
    lines = [f"# Stage 5 summary ({m}, Rayleigh ZF, h per {args.h_draw})", "",
             f"Q0 ceiling (b=4): {q0:.4f}" if q0 is not None else "Q0 ceiling: n/a",
             f"Analog noiseless: {c_inf:.4f}" if c_inf is not None else "", "",
             "| variant | uses/token | max BLEU-1 | SNR @ 0.9 Q0 | SNR @ 0.95 Q0 | view-2 shift | view-2 crossover vs C0 |",
             "|---|---|---|---|---|---|---|"]
    for var in ["c0"] + [v for v in DIGITAL if v in thresh]:
        th = thresh.get(var, {})
        cr = cross.get("view2", {}).get(var, {})
        uses = cr.get("uses_per_token", ANALOG_USES if var == "c0" else float("nan"))
        fmt = lambda x: "-" if x is None else f"{x:.2f}"
        xc = cr.get("crossover_snr_db")
        st = cr.get("status", "")
        xs = "-" if var == "c0" else (f"{xc:.2f} dB" if xc is not None and st == "crosses" else st)
        lines.append(f"| {LABELS[var]} | {uses:.1f} | {fmt(th.get('bleu1_max'))} | {fmt(th.get('snr_at_0.9_q0'))} | "
                     f"{fmt(th.get('snr_at_0.95_q0'))} | {'-' if var == 'c0' else fmt(cr.get('shift_db')) + ' dB'} | {xs} |")
    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
