#!/bin/bash
# Dev-subset sweeps (5000 sentences), run sequentially. Logs in results/logs/.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
S=${1:-5000}
mkdir -p results/logs
rm -f results/logs/dev_sweeps.done
python -m quantized_deepsc.run_baseline --subset $S --fine --out-dir results/s0 > results/logs/s0_baseline_dev.log 2>&1
for b in 4 2 3 5 6 8; do
  python -m quantized_deepsc.run_digital --variant q0 --channel none --b $b --subset $S --out-dir results/s2 >> results/logs/s2_q0_dev.log 2>&1
done
python -m quantized_deepsc.run_digital --variant u_bpsk u_pam16 --channel rayleigh --h-draw sentence --fine --subset $S --out-dir results/s3 > results/logs/s3_uncoded_rayleigh_dev.log 2>&1
python -m quantized_deepsc.run_digital --variant c_bpsk c_pam16 --channel rayleigh --h-draw sentence --fine --subset $S --out-dir results/s4 > results/logs/s4_coded_rayleigh_dev.log 2>&1
python -m quantized_deepsc.run_digital --variant u_bpsk u_pam16 --channel awgn --fine --subset $S --out-dir results/s3 > results/logs/s3_uncoded_awgn_dev.log 2>&1
python -m quantized_deepsc.run_digital --variant c_bpsk c_pam16 --channel awgn --fine --subset $S --out-dir results/s4 > results/logs/s4_coded_awgn_dev.log 2>&1
echo DONE > results/logs/dev_sweeps.done
