"""Run link variants (analog c0, ceiling q0, digital u_*/c_*) over a channel and SNR grid.

    python -m quantized_deepsc.run_digital --variant c0 --channel rayleigh --h-draw sentence --subset 5000
    python -m quantized_deepsc.run_digital --variant q0 --channel none --b 4
    python -m quantized_deepsc.run_digital --variant u_pam16 c_bpsk --channel rayleigh --fine
    python -m quantized_deepsc.run_digital --variant u_bpsk g_bpsk --channel awgn --grand-budget 2000

One CSV per variant in --out-dir, with a '#'-comment header (seed, checkpoint hash, config).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .chain import VARIANTS, DigitalChain
from .deepsc_io import DeepSCLink, load_config, resolve
from .grand_code import GrandCode
from .ldpc import LDPC
from .modulation import PAM
from .quant import ScalarQuantizer


def build_chain(link, cfg, b=None, A=None, decoder=None, bp_iters=None, need_ldpc=True, need_grand=False,
                grand_code=None, grand_budget=None):
    b = b or int(cfg["b"])
    if A is None:
        A = cfg.get("A")
    if A is None:
        cal = resolve(cfg["calibration_file"])
        if not cal.exists():
            raise SystemExit(f"no A given and {cal} not found: run  python -m quantized_deepsc.calibrate  first")
        with open(cal, "r", encoding="utf-8") as f:
            A = float(json.load(f)["A"])
    Q = ScalarQuantizer(A, b)
    dev = link.device
    ldpc = None
    if need_ldpc:
        lc = cfg["ldpc"]
        ldpc = LDPC(n=lc["n"], dv=lc["dv"], dc=lc["dc"], k=lc["k"], seed=int(cfg["seed"]),
                    cache_dir=resolve("results/ldpc"), device=dev, bp_iters=bp_iters or lc["bp_iters"],
                    decoder=decoder or lc["decoder"], llr_clip=lc.get("llr_clip", 20.0),
                    min_sum_alpha=lc.get("min_sum_alpha", 0.8))
    grand = None
    if need_grand:
        gc = cfg["grand"]
        grand = GrandCode(resolve(gc["dir"]), spec=grand_code or gc["code"],
                          max_queries=grand_budget or gc["max_queries"], variant=gc.get("variant", "1line"))
    return DigitalChain(link, Q, PAM(2, dev), PAM(16, dev), ldpc, grand), Q, ldpc


def run_variant(link, chain, variant, channel, h_draw, snrs, subset, passes, seed, max_len, verbose=True, on_row=None):
    rows = []
    for snr in snrs:
        sigma = link.snr_to_sigma(snr) if channel != "none" else 0.0
        t0 = time.time()
        acc = defaultdict(float)
        bleu = defaultdict(float)
        n_sent = 0
        for p in range(passes):
            torch.manual_seed(seed + p)
            np.random.seed(seed + p)
            targets, decoded = [], []
            for sents in link.test_loader(subset):
                t, pad = link.encode(sents)
                t_hat, st = chain.run(variant, t, sigma, channel, h_draw)
                out = link.decode(t_hat, max_len)
                decoded += link.to_text(out)
                targets += link.to_text(sents)
                err2 = (t_hat - t) ** 2
                st["mse_all_sum"] = float(err2.sum())
                st["n_all"] = int(err2.numel())
                st["mse_nonpad_sum"] = float(err2[~pad].sum())
                st["n_nonpad"] = int(err2[~pad].numel())
                st["tokens_nonpad"] = int((~pad).sum())
                for k, v in st.items():
                    acc[k] += v
            bl = link.bleu(targets, decoded)
            for k, v in bl.items():
                bleu[k] += v / passes
            n_sent = len(targets)
        row = {"variant": variant, "channel": channel, "h_draw": h_draw if channel == "rayleigh" else "-",
               "snr_db": snr if channel != "none" else float("nan"), "passes": passes, "n_sentences": n_sent,
               "b": chain.Q.b, "A": chain.Q.A}
        row.update(bleu)
        row["ber"] = acc["bit_errors"] / acc["n_bits"] if acc.get("n_bits") else float("nan")
        row["level_err_rate"] = acc["level_errors"] / acc["n_levels"] if acc.get("n_levels") else float("nan")
        row["fer"] = acc["block_errors"] / acc["n_blocks"] if acc.get("n_blocks") else float("nan")
        row["bp_iters_mean"] = acc["iters_sum"] / acc["n_blocks"] if acc.get("iters_sum") else float("nan")
        is_grand = acc.get("grand_queries") is not None
        row["abandon_rate"] = acc["grand_abandoned"] / acc["n_blocks"] if is_grand else float("nan")
        row["miscorrect_rate"] = acc["grand_miscorrected"] / acc["n_blocks"] if is_grand else float("nan")
        row["queries_mean"] = acc["grand_queries"] / acc["n_blocks"] if is_grand else float("nan")
        row["mse_all"] = acc["mse_all_sum"] / acc["n_all"]
        row["mse_nonpad"] = acc["mse_nonpad_sum"] / acc["n_nonpad"]
        row["uses_per_token_all"] = acc["real_uses"] / acc["tokens_all"]
        row["uses_per_token_nonpad"] = acc["real_uses"] / acc["tokens_nonpad"]
        row["pad_bits_frac"] = acc["pad_bits"] / acc["n_bits"] if acc.get("n_bits") and acc.get("pad_bits") else 0.0
        row["time_s"] = time.time() - t0
        rows.append(row)
        if on_row is not None:
            on_row(rows)
        if verbose:
            extra = f"aband={row['abandon_rate']:.2e}  miscorr={row['miscorrect_rate']:.2e}  " if is_grand else ""
            print(f"[{variant:8s} {channel:8s} {row['h_draw']:8s}] snr={snr:6.1f} dB  BLEU-1={row['bleu1']:.4f}  "
                  f"BER={row['ber']:.2e}  LER={row['level_err_rate']:.2e}  FER={row['fer']:.2e}  {extra}"
                  f"MSE={row['mse_nonpad']:.4f}  uses/tok={row['uses_per_token_all']:.1f}  ({row['time_s']:.0f}s)", flush=True)
        if channel == "none":
            break
    return rows


def write_csv(rows, path, header: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in header.items():
            f.write(f"# {k}={v}\n")
        df.to_csv(f, index=False, lineterminator="\n")
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", nargs="+", default=["c0"], choices=VARIANTS)
    p.add_argument("--channel", default="rayleigh", choices=["none", "awgn", "rayleigh"])
    p.add_argument("--h-draw", default="sentence", choices=["sentence", "batch", "symbol", "repo"])
    p.add_argument("--snr", nargs="*", type=float, default=None, help="SNR labels in dB (default: config grid)")
    p.add_argument("--fine", action="store_true", help="add config snr_fine_db to the grid")
    p.add_argument("--subset", default=None, help="int or 'full' (default: config dev_subset)")
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--b", type=int, default=None)
    p.add_argument("--A", type=float, default=None)
    p.add_argument("--decoder", default=None, choices=[None, "sum_product", "min_sum"])
    p.add_argument("--bp-iters", type=int, default=None)
    p.add_argument("--grand-code", default=None, help="g_* variants: code spec, e.g. crc:0x9eb2:128:112 (default: config)")
    p.add_argument("--grand-budget", type=int, default=None, help="g_* variants: ORBGRAND query budget per block (default: config)")
    p.add_argument("--out-dir", default="results/runs")
    p.add_argument("--tag", default="")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    if args.subset is None:
        subset = cfg.get("dev_subset")
    else:
        subset = None if args.subset == "full" else int(args.subset)
    snrs = list(args.snr) if args.snr else list(cfg["snr_grid_db"])
    if args.fine:
        snrs = sorted(set(snrs) | set(cfg["snr_fine_db"]))
    if args.channel == "none":
        snrs = [float("nan")]

    link = DeepSCLink(cfg)
    need_ldpc = any(v.startswith("c_") for v in args.variant)
    need_grand = any(v.startswith("g_") for v in args.variant)
    chain, Q, ldpc = build_chain(link, cfg, args.b, args.A, args.decoder, args.bp_iters, need_ldpc, need_grand,
                                 args.grand_code, args.grand_budget)
    print(f"checkpoint {link.ckpt_path.name} sha256={link.ckpt_sha256[:12]}  {Q}  {ldpc if ldpc else ''}  {chain.grand or ''}")
    print(f"subset={subset or 'full'}  passes={args.passes}  channel={args.channel}  h_draw={args.h_draw}  snrs={snrs}")

    out_dir = resolve(args.out_dir)
    for variant in args.variant:
        hd = args.h_draw if args.channel == "rayleigh" else "-"
        tag = f"_{args.tag}" if args.tag else ""
        btag = f"_b{Q.b}" if Q.b != int(cfg["b"]) else ""
        fn = out_dir / f"{variant}_{args.channel}_{hd}{btag}{tag}.csv"
        header = {"variant": variant, "channel": args.channel, "h_draw": hd, "seed": seed, "passes": args.passes,
                  "subset": subset or "full", "checkpoint": link.ckpt_path.name, "ckpt_sha256": link.ckpt_sha256,
                  "A": Q.A, "b": Q.b, "delta": Q.delta, "ldpc": repr(ldpc) if ldpc else "-",
                  "grand": repr(chain.grand) if chain.grand and variant.startswith("g_") else "-",
                  "snr_convention": "sigma=1/sqrt(2*10^(L/10)) per real dim; unit-energy constellations; Es/sigma^2 = L+3 dB",
                  "config": json.dumps(cfg)}
        # the CSV is rewritten after every SNR point so an interrupted run keeps its finished points
        run_variant(link, chain, variant, args.channel, args.h_draw, snrs, subset, args.passes, seed, link.max_len,
                    on_row=lambda rows: write_csv(rows, fn, header))
        print(f"wrote {fn}")


if __name__ == "__main__":
    main()
