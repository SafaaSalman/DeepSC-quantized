# Quantized DeepSC - REPORT

Status (2026-09-09): pipeline implemented and validated on the 5 000-sentence dev subset of the Europarl test set, all stages.
The full-test-set run (`run_full.sh`, about 8 h) was deliberately not performed at this point.
Code: `quantized_deepsc/` (see `README.md`). Tests: `python -m pytest tests` (27 tests).
Reproduce: `python -m quantized_deepsc.calibrate`, `bash run_dev_sweeps.sh 5000`, `python -m quantized_deepsc.plot_stages`,
`python -m quantized_deepsc.run_compare --runs results/s0 results/s3 results/s4 --q0 results/s2 --out results/s5`.

## 1. Conventions used everywhere

| Item | Definition |
|---|---|
| Checkpoint | `DeepSC-continous/checkpoints/deepsc-Rayleigh/checkpoint_80.pth`, SHA-256 `15fb163ec2c286fab8da0180467fb341da4c54f3c6f31398ff01095224cb7aa6` |
| Model | 4 layers, d_model 128, dff 512, 8 heads, D = 16 real latents/token, vocab 57 975 |
| Evaluation | batch 64, test-set order, per-batch `PowerNormalize`, pads transmitted by both systems, greedy decoding to 30 tokens, BLEU on decoded tokens up to `<END>` with reference = target, one pass, seed 1234 |
| SNR label L | `sigma = 1/sqrt(2*10^(L/10))` per real dimension (repo `SNR_to_noise`). All constellations have unit average energy per real dimension and the analog latent has RMS 1.00, so **Es/sigma^2 = L + 3 dB** per real dimension for every system. For BPSK with rate R: Eb/N0 [dB] = L + 10 log10(1/R). |
| Fading | h ~ CN(0,1), one draw per sentence (the repo draws one per batch; that run is kept as a reference only); consecutive real pairs form one complex symbol; zero forcing y = r/h; effective noise variance sigma^2/|h|^2 per real dimension given h |
| Demapper | coherent, knows |h|^2: BPSK L = 2 |h|^2 y / sigma^2; 16-PAM max-log per bit with metric -|h|^2 (y - v)^2 / (2 sigma^2). Sign: L > 0 <=> bit 0 |
| Quantiser | uniform, b = 4 bits, A = 4.5404 (99.9-th percentile of |t| on the transmitted tensor), Delta = 0.6054, Gray labels idx XOR (idx >> 1), MSB first |
| Code | (3,6)-regular LDPC, n = 1008, k = 504 (k_eff = 504, no shortening needed), 0 four-cycles, systematic, sum-product BP, 50 iterations, early stop on zero syndrome, LLR clip +-20 |
| Framing | per sentence: L x 64 bits, zero-padded to a multiple of 504, blocks decoded independently, padding stripped |
| Channel uses / token | analog 16; U-PAM16 16; U-BPSK 64; C-PAM16 32.6 (measured, incl. block padding); C-BPSK 130.5 (measured). Counted per batch with pads included for both systems. |

## 2. Acceptance checks

| Stage | Check | Expected | Obtained | Pass |
|---|---|---|---|---|
| 1 | A (99.9 %), Delta, q_rms on the transmitted tensor (batch 64, pads incl., 59.7 M values) | ~4.54 / ~0.605 / 0.175 | 4.5404 / 0.6054 / 0.1748 | yes |
| 1 | clipped fraction | 0.1 % | 0.100 % | yes |
| 1 | index entropy, mass on levels 7+8 | ~2.61 bit / ~65 % | 2.613 bit / 65.2 % | yes |
| 1 | std(t) all positions / non-pad | 1.00 / 1.24 | 0.999 / 1.236 | yes |
| 2 | Gray round trip for b in {2,3,4,5,6,8}; adjacent levels differ in one bit; quantiser never overflows; error <= Delta/2 inside [-A, A] | unit tests | pass | yes |
| 3 | BPSK: bit 0 -> +1, LLR = 2 h2 y / sigma^2; 16-PAM unit energy, Gray adjacency, level k <-> symbol k, hard/soft round trip | unit tests | pass | yes |
| 3 | V3.0 uncoded BPSK BER, AWGN, L in {0,3,6} dB vs Q(1/sigma) | within 5 % | pass (2e6 bits) | yes |
| 3 | V3.0 uncoded BPSK BER, Rayleigh ZF (h per symbol), L in {0,6,12} dB vs 0.5(1 - sqrt(g/(1+g))), g = 10^(L/10) | within 5 % | pass (2e6 bits) | yes |
| 3 | V3.0 16-PAM SER, AWGN, L in {18,24} dB vs 2(1-1/M) Q(d/sigma) | within 5 % | pass (1e6 symbols) | yes |
| 4 | H is (3,6)-regular 504x1008, no 4-cycles, syndrome(encode(u)) = 0, noiseless decode exact, 20 random flips corrected | unit tests | pass | yes |
| 4 | V4.0 LDPC BER over AWGN at Eb/N0 = 2.0 dB (2 000 blocks) | ~1e-3 (literature 8.5e-4) | 1.18e-3 (FER 0.019, 10.5 iterations mean) | yes |
| 4 | V4.0 LDPC BER at Eb/N0 = 3.0 dB | < 1e-4 | 0 errors in 1.0e6 bits | yes |
| 4 | chain end to end on synthetic latents: every variant reproduces the quantised latent exactly through a noiseless channel (also through noiseless fading for the coded ones); channel uses per token = 16/64/16/32/128 (+ block padding); hard decision = sign of max-log LLR; coded BER < uncoded BER / 10 at 2 dB AWGN | unit tests | 11/11 pass | yes |
| 4 | end-to-end consistency with the isolated code: C-BPSK over AWGN at label -1 dB (= Eb/N0 2 dB) | BER ~1e-3, BLEU = Q0 once BER = 0 | BER 9.2e-4; BLEU-1 = 0.8689 = Q0 at 0 dB and above | yes |
| 0 | analog baseline monotone in SNR (dev subset) | non-decreasing | Rayleigh h/sentence: rises 0.537 (0 dB) -> 0.829 (18 dB), then declines slowly to 0.826 (24 dB); AWGN peaks 0.867 at 5 dB and declines to the noiseless value 0.822 | **fails above the training SNR, explained**: the decoder was trained at 5-10 dB Rayleigh and prefers inputs with that noise level |
| 0 | h-per-batch (repo) vs h-per-sentence agree at >= 12 dB | +-0.02 BLEU-1 | 12 dB 0.834 vs 0.820; 15 dB 0.837 vs 0.827; 18 dB 0.833 vs 0.829 | yes |
| 2 | Q0(b) increasing in b, within 0.01 of noiseless analog at b >= 6 | - | b=2 0.376, b=3 **0.896**, b=4 0.869, b=5 0.809, b=6 0.802, b=8 0.815; noiseless analog 0.822 | **fails, explained**: same decoder noise preference (b=3 quantisation noise, q_rms 0.37, is closest to the training noise); b=8 is within 0.01 of noiseless analog |
| 3/4 | every digital curve reaches the Q0 plateau (0.869) at high SNR | +-0.01 | AWGN: all four variants reach 0.869 exactly (BER = 0). Rayleigh at 24 dB: C-BPSK 0.8675, U-BPSK 0.8639 (both within 0.01); U-PAM16 0.852 and C-PAM16 0.853 still rising (BER 1.4e-2 to 3.8e-2) | yes for AWGN and BPSK; 16-PAM over quasi-static fading has not reached the plateau by 24 dB |

## 3. Results (5 000 test sentences, one pass; BLEU-1 standard error about 0.004)

### 3.1 Analog baseline (Stage 0)

**BLEU-1, Rayleigh ZF (h per sentence unless stated)**

| variant | 0 dB | 3 dB | 6 dB | 9 dB | 12 dB | 15 dB | 18 dB | 24 dB |
|---|---|---|---|---|---|---|---|---|
| C0 analog (h/sentence) | 0.537 | 0.667 | 0.750 | 0.798 | 0.820 | 0.827 | 0.829 | 0.826 |
| C0 analog (h/batch, repo) | 0.556 | 0.707 | 0.785 | 0.818 | 0.834 | 0.837 | 0.833 | 0.825 |

AWGN: 0.738 (0 dB), 0.855 (3 dB), peak 0.867 at 5 dB, 0.838 (12 dB), 0.823 (24 dB); noiseless 0.822.
The per-batch fading of the repo is optimistic by up to 0.04 BLEU-1 at low SNR (one gain shared by 64 sentences averages out fades); the per-sentence curve is used for all comparisons.
Figure: `results/s0/s0_baseline.png`.

### 3.2 Quantisation-only ceiling (Stage 2)

| b | Delta | q_rms | latent MSE | BLEU-1 | BLEU-4 |
|---|---|---|---|---|---|
| 2 | 3.027 | 0.874 | 1.0250 | 0.3755 | 0.1467 |
| 3 | 1.297 | 0.374 | 0.1773 | 0.8955 | 0.8025 |
| 4 | 0.605 | 0.175 | 0.0371 | 0.8689 | 0.7890 |
| 5 | 0.293 | 0.085 | 0.0085 | 0.8087 | 0.7285 |
| 6 | 0.144 | 0.042 | 0.0023 | 0.8020 | 0.7187 |
| 8 | 0.036 | 0.010 | 0.0006 | 0.8149 | 0.7289 |
| analog, noiseless | - | 0 | 0 | 0.8223 | 0.7356 |

The latent MSE follows Delta^2/12 as it should (0.0371 vs 0.0305 at b = 4; the excess is the 0.1 % clipped values). The BLEU ordering does not: the frozen decoder scores highest with 3-bit quantisation noise and *lowest* with a nearly clean input. The decoder was trained on Rayleigh-faded inputs at 5-10 dB and treats a clean latent as out of distribution. All digital plateaus below are therefore at Q0 = 0.869 (b = 4), above the best analog value (0.829 on Rayleigh, 0.867 on AWGN).
Figure: `results/s2/s2_bits_vs_bleu.png`.

### 3.3 Uncoded digital links (Stage 3)

**BLEU-1, Rayleigh h per sentence**

| variant | 0 dB | 3 dB | 6 dB | 9 dB | 12 dB | 15 dB | 18 dB | 24 dB |
|---|---|---|---|---|---|---|---|---|
| U-BPSK (64 uses/token) | 0.319 | 0.483 | 0.628 | 0.732 | 0.797 | 0.832 | 0.850 | 0.864 |
| U-PAM16 (16 uses/token) | 0.155 | 0.228 | 0.357 | 0.511 | 0.646 | 0.742 | 0.800 | 0.852 |

**BLEU-1, AWGN**: U-BPSK 0.295 (0 dB), 0.842 (6 dB), 0.868 (9 dB), 0.869 from 11 dB (BER 0); U-PAM16 0.381 (6 dB), 0.692 (9 dB), 0.833 (12 dB), 0.865 (15 dB), 0.870 (24 dB).

Bit error rates on Rayleigh (h per sentence) fall as 1/SNR: U-BPSK 1.4e-1 (0 dB) -> 3.9e-3 (18 dB) -> 9.0e-4 (24 dB); U-PAM16 4.1e-1 -> 1.0e-1 -> 3.8e-2. The relation between level errors and BLEU is smooth: a level error rate of 1 % costs about 0.01 BLEU-1, 10 % about 0.07 (compare U-BPSK at 18 dB and U-PAM16 at 18 dB).
Figures: `results/s3/s3_uncoded_bleu.png`, `results/s3/s3_uncoded_ber.png`.

### 3.4 LDPC-coded digital links (Stage 4)

**BLEU-1, Rayleigh h per sentence**

| variant | 0 dB | 3 dB | 6 dB | 9 dB | 12 dB | 15 dB | 18 dB | 24 dB |
|---|---|---|---|---|---|---|---|---|
| C-BPSK (130.5 uses/token) | 0.533 | 0.675 | 0.762 | 0.815 | 0.842 | 0.853 | 0.862 | 0.868 |
| C-PAM16 (32.6 uses/token) | 0.157 | 0.229 | 0.358 | 0.511 | 0.649 | 0.743 | 0.803 | 0.853 |

**FER (LDPC blocks), Rayleigh**: C-BPSK 4.7e-1 (0 dB), 1.5e-1 (6 dB), 3.8e-2 (12 dB), 1.1e-2 (18 dB), 2.0e-3 (24 dB); C-PAM16 1.0 up to 6 dB, 6.8e-1 (12 dB), 2.5e-1 (18 dB), 7.0e-2 (24 dB).

**AWGN**: C-BPSK has its waterfall between label -3 dB (BER 1.3e-1) and 0 dB (BER 1.9e-6, BLEU-1 0.869), i.e. between Eb/N0 = 0 and 3 dB, matching the standalone code validation. C-PAM16 has its waterfall between 12 dB (BER 8.7e-2) and 15 dB (BER 6e-7, BLEU-1 0.869).

Over quasi-static Rayleigh fading (one gain per sentence) there is no waterfall: every block of a faded sentence fails together, so FER decays one decade per 10 dB (outage-limited, diversity order 1) and the coding gain over the uncoded link is modest in BLEU (C-BPSK vs U-BPSK: +0.21 at 0 dB, +0.08 at 9 dB, +0.01 at 18 dB). Mean BP iterations at 24 dB: 5.2 (C-PAM16), i.e. most blocks converge quickly and the residual errors are fades.
Figures: `results/s4/s4_coded_bleu.png`, `results/s4/s4_coded_ber_fer.png`.

### 3.5 Digital vs analog (Stage 5), Rayleigh, h per sentence

| variant | uses/token | max BLEU-1 | SNR @ 0.9 Q0 (0.782) | SNR @ 0.95 Q0 (0.825) | view-2 shift | view-2 crossover vs C0 |
|---|---|---|---|---|---|---|
| C0 analog DeepSC | 16.0 | 0.829 | 7.9 dB | 14.3 dB | - | - |
| U-BPSK (uncoded, hard) | 64.0 | 0.864 | 11.1 dB | 14.4 dB | +6.0 dB | 20.7 dB |
| U-PAM16 (uncoded, hard) | 16.0 | 0.852 | 16.9 dB | 19.9 dB | 0 dB | 20.2 dB |
| C-BPSK (LDPC 1/2) | 130.5 | 0.868 | 7.1 dB | 10.2 dB | +9.1 dB | 19.6 dB |
| C-PAM16 (LDPC 1/2) | 32.6 | 0.853 | 16.8 dB | 19.9 dB | +3.1 dB | 23.1 dB |

- **View 1, equal SNR per channel use** (`results/s5/s5_bleu1_vs_snr_view1.png`): C-BPSK is on par with analog from 0 dB and above it from 3 dB (+0.02 at 12 dB, +0.04 at 24 dB), at 8 times the bandwidth. U-BPSK overtakes analog at about 14 dB. Both 16-PAM links stay far below analog until about 20 dB.
- **View 2, equal energy per token** (`results/s5/s5_bleu1_vs_snr_view2.png`): after shifting each digital curve by its bandwidth cost, the analog link is better than every digital variant up to about 20 dB per token, by a wide margin in the middle of the range (at 10 dB per token: analog 0.81, C-BPSK 0.55, U-PAM16 0.56, C-PAM16 0.40). The crossovers at 19.6-23.1 dB are produced by the analog curve saturating at 0.83 while the digital ones climb to the Q0 plateau of 0.869; they are not a digital "cliff advantage".
- **View 3, equal channel uses** (`results/s5/s5_bleu1_vs_snr_view3.png`): uncoded 16-PAM, the only like-for-like replacement, needs about 17 dB to reach 90 % of its ceiling where analog needs 8 dB, and crosses analog at 20.2 dB.
- `results/s5/s5_mse_to_bleu.png` pools all runs: BLEU-1 is a smooth, nearly channel-independent function of the latent MSE at the decoder input down to MSE ~ 0.05, below which the decoder's noise preference makes it non-monotone.

Numbers: `results/s5/s5_crossover.json`, `results/s5/s5_thresholds.json`, `results/s5/summary.md`, `results/s5/tables.md`.

### 3.6 What the results say about the pipeline

1. Every digital variant reaches exactly the quantisation-only BLEU (0.8689) as soon as its BER is zero (AWGN, all four variants), so quantiser, Gray labelling, framing, modulation, demapping and decoding are mutually consistent end to end.
2. The coded BPSK waterfall over AWGN sits at Eb/N0 = 0-3 dB with BER 9.2e-4 at 2 dB, matching the isolated LDPC validation (1.2e-3) and the literature for this code.
3. Closed-form BER/SER checks pass for BPSK over AWGN and Rayleigh-ZF and for 16-PAM over AWGN, so the channel, the ZF and the SNR convention are right.
4. The fading curves have the outage-limited slope expected for one gain per sentence, and the repo's per-batch fading is measurably optimistic at low SNR.

### 3.7 Caveats

- Dev subset only (first 5 000 of 120 726 test sentences), one pass. The full-set script `run_full.sh` exists (about 8 h).
- BLEU-1 is the headline metric; BLEU-2..4 are in every CSV. Sentence similarity (BERT) is not computed: the repo's implementation is disabled and its dependency is not installed.
- The frozen decoder's preference for noisy inputs (sections 3.1-3.2) is a property of `checkpoint_80` and shapes the high-SNR comparison; any conclusion about "digital beats analog above 20 dB" is really "4-bit quantisation noise suits this decoder better than a clean latent".

## 4. Runtime notes

- Encoder pass over the full test set: ~10 s. Greedy decoding dominates: ~15 s per SNR point on the 5 000-sentence subset (about 12 min per point on the full set). C-PAM16 below its waterfall costs ~1 min per point because unconverged blocks run all 50 BP iterations.
- LDPC construction with 4-cycle removal: seconds, cached in `results/ldpc/`. Batched BP on 4 096 codewords x 50 iterations: well under a second on the RTX 5060.
- All runs rewrite their CSV after every SNR point, so an interrupted run keeps its finished points. The dev sweep (`run_dev_sweeps.sh`) took about 1 h 45 min in total.
