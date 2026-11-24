#!/bin/bash
# Final curves on the FULL test set (120 726 sentences), coarse grid {0,3,...,18} dB, Rayleigh h per sentence.
# ~12 min per SNR point per variant on the RTX 5060 -> about 8 h total. Runs sequentially; CSVs are rewritten
# after every SNR point, so the job can be killed and resumed by variant.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p results/logs
rm -f results/logs/full.done
python -m quantized_deepsc.run_digital --variant q0 --channel none --subset full --out-dir results/full > results/logs/full_q0.log 2>&1
python -m quantized_deepsc.run_digital --variant c0 --channel none --subset full --out-dir results/full > results/logs/full_c0_none.log 2>&1
python -m quantized_deepsc.run_digital --variant c0 --channel rayleigh --h-draw sentence --subset full --out-dir results/full > results/logs/full_c0.log 2>&1
python -m quantized_deepsc.run_digital --variant u_pam16 --channel rayleigh --h-draw sentence --subset full --out-dir results/full > results/logs/full_u_pam16.log 2>&1
python -m quantized_deepsc.run_digital --variant c_pam16 --channel rayleigh --h-draw sentence --subset full --out-dir results/full > results/logs/full_c_pam16.log 2>&1
python -m quantized_deepsc.run_digital --variant c_bpsk --channel rayleigh --h-draw sentence --subset full --out-dir results/full > results/logs/full_c_bpsk.log 2>&1
python -m quantized_deepsc.run_digital --variant u_bpsk --channel rayleigh --h-draw sentence --subset full --out-dir results/full > results/logs/full_u_bpsk.log 2>&1
python -m quantized_deepsc.run_compare --runs results/full --q0 results/full --out results/full_compare > results/logs/full_compare.log 2>&1
echo DONE > results/logs/full.done
