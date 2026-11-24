"""Stage 0: analog DeepSC baseline (variant c0) over the repo-original channel (h per batch),
h per sentence, AWGN, and noiseless.

    python -m quantized_deepsc.run_baseline [--subset N|full] [--passes 2] [--fine]
"""
from __future__ import annotations

import argparse

from .run_digital import main as run_main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default=None)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--fine", action="store_true")
    p.add_argument("--out-dir", default="results/s0")
    p.add_argument("--only", nargs="*", default=["repo", "sentence", "awgn", "none"],
                   help="subset of {repo, sentence, awgn, none}")
    args = p.parse_args(argv)
    common = ["--variant", "c0", "--passes", str(args.passes), "--out-dir", args.out_dir]
    if args.subset is not None:
        common += ["--subset", str(args.subset)]
    if args.fine:
        common += ["--fine"]
    if "repo" in args.only:
        run_main(common + ["--channel", "rayleigh", "--h-draw", "repo"])
    if "sentence" in args.only:
        run_main(common + ["--channel", "rayleigh", "--h-draw", "sentence"])
    if "awgn" in args.only:
        run_main(common + ["--channel", "awgn"])
    if "none" in args.only:
        run_main(common + ["--channel", "none"])


if __name__ == "__main__":
    main()
