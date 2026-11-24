# Implementation Plan: Digital DeepSC (quantised latent -> bits -> [LDPC] -> M-PAM -> fading channel -> frozen DeepSC receiver)

**Scope (revised 2026-09-09).** Digitalise the *existing, already-trained* continuous DeepSC link in `../DeepSC-continous` and compare it with the analog link on BLEU vs SNR. The receiver uses textbook components only: hard decisions, the standard coherent Gaussian LLR, and sum-product LDPC decoding.

**Out of scope (for now).** Everything that is the contribution of Wakim-Ismail-Sarieddeen (`EECE499PaperCarl.pdf`): the Student-t matched LLR, GMI / calibration study, effective-noise identification (KS / Hill / moments), the semantic source prior, and the GRAND hook. The previous plan that covered those is kept as `IMPLEMENTATION_PLAN_full_wakim_reproduction_ORIGINAL.md`. Interfaces below are kept clean so those pieces can be added later without refactoring (Appendix A records the conventions that were already verified for that purpose).

**Non-goals.** No retraining. No change to any DeepSC weights. No vector quantisation.

**Working principle.** Every stage ends with a numeric acceptance check that is either a closed-form result (BER formulas, code-theory validation points) or an internal consistency check. If a check fails, stop and fix before the next stage.

---

## 0. Repository audit - what the code actually does (checked 2026-09-09)

| Item | What the repo has | Consequence for this plan |
|---|---|---|
| Model | `DeepSC(num_layers=4, d_model=128, dff=512, num_heads=8)`; channel encoder `128->256->16` (D = 16 real latents / token); channel decoder `16->128->512->128` | Digital chain sits between `PowerNormalize` and `channel_decoder` |
| Checkpoints | `checkpoints/deepsc-Rayleigh/checkpoint_01..80.pth` (every epoch saved: `record_acc` is reset inside the loop in `main.py`). No validation log exists. `performance.py` picks the highest index. | **Use `checkpoint_80.pth`** (loads with `strict=True`; verified). Record its SHA-256 in every CSV header. |
| Training regime | Rayleigh, one h per **batch** (128), noise sigma drawn once per epoch from SNR in [5, 10] dB | Decoder is tolerant to Rayleigh-ZF noise around 5-10 dB |
| Data | `europarl/{train,test}_data.pkl`, `vocab.json`. **Vocab = 57 975 tokens** (not 22 234 as in the paper). Train 1 086 536 / test 120 726 sentences; test lengths 7-31 tokens incl. `<START>/<END>`, mean 20.2 | Digital framing is vocab-independent (64 bits/token). Full test set is feasible; use a 5 000-sentence dev subset for sweeps. |
| Stored results | **None** (no CSV/log of the continuous BLEU curve) | Stage 0 *generates* the baseline instead of confirming it |
| `performance.py` | Does not run as-is: hard-coded vocab path `/import/antennas/...`, `from sklearn.preprocessing import normalize` (sklearn not installed), BERT similarity commented out (`bert4keras` not installed). `compute_blue_score(real, predicted)` is called with (decoded, target), i.e. reference/hypothesis swapped. | Do **not** import it. Re-implement its 40-line loop in the harness, reusing `greedy_decode`, `SeqtoText`, `BleuScore` from `utils.py`. BLEU-1 is the metric; similarity is optional later (`sentence-transformers` installs on this Python, dry-run verified). |
| `PowerNormalize` | Per **batch**, over all positions including pads, and only when RMS > 1 (95 % of test batches at batch 64) | Latent scale depends on batch composition (pre-norm RMS 0.92-1.19). Calibrate the quantiser on the tensor that is actually transmitted, at the same batch size as the evaluation. |
| `Channels.Rayleigh` | One complex h ~ CN(0,1) per **batch**; consecutive real pairs are treated as one complex symbol; ZF via `inverse(H)`; h is not returned | Digital chain needs h per **sentence** and needs abs(h)^2 at the demapper -> own `channel.py` mirroring the same pair convention |
| SNR convention | `SNR_to_noise(L)`: sigma = 1/sqrt(2*10^(L/10)) per real dimension; signal power ~ 1 per real dimension after normalisation | Es/sigma^2 per real dimension = **L + 3 dB**. For BPSK with a = 1 and R = 1/2, Eb/N0 [dB] = L + 3 (see section 3). |
| `greedy_decode` | Passes `None` as the source-padding mask to the decoder -> the decoder attends to **padded** memory positions | Pad positions influence decoding, so the digital chain must transmit them too (the analog chain does). Pad fraction at batch 64 in test order: **34.7 %**. |
| Environment | Python 3.14.2, torch 2.11.0+cu128, RTX 5060 Laptop GPU, numpy 2.4, scipy 1.17, matplotlib, pandas, pyyaml, pytest. Missing: sklearn, bert4keras, transformers. | Everything needed is present. Optional: `pip install pyldpc` for a reference LDPC decoder. |

### 0.1 Measured latent statistics (checkpoint_80, full test set, b = 4, A = 99.9-th percentile of abs(t))

| Extraction regime | N values | std(t) | A | Delta = 2A/15 | index entropy | mass on levels 7+8 |
|---|---|---|---|---|---|---|
| batch 64, all positions (pads included) - **what is transmitted** | 59.7 M | 0.999 | **4.540** | **0.605** | 2.61 bit | 65.2 % |
| batch 64, non-pad positions only | 39.0 M | 1.236 | 4.720 | 0.629 | 3.00 bit | 47.8 % |
| batch 1 (first 20 k sentences, no pads) | 6.4 M | 1.000 | 3.808 | 0.508 | 3.00 bit | 47.7 % |
| *paper (different checkpoint, vocab 22 234)* | - | 1.194 | 4.387 | 0.585 | 2.739 bit | 59.7 % |

Clipped fraction is 0.100 % in all regimes (by construction). Encoder pass over the full test set: ~10 s at batch 64; ~14 min at batch 1. Per-Gray-bit P(bit = 1), MSB -> LSB, batch-64/all positions: 0.545, 0.958, 0.184, 0.271.

**Decisions taken from the audit**
1. Evaluate both systems with the repo's harness settings: batch 64, test-set order, per-batch `PowerNormalize`, pads transmitted. Channel-use accounting is per batch so both systems pay the same padding cost.
2. Calibrate A on the transmitted tensor (row 1 above); do not hard-code the paper's 4.387.
3. Fading: one h per sentence in the comparison (section 3). The repo-original per-batch h is run once for reference only.

---

## 1. The chain and its variants

```
              frozen DeepSC transmitter                    digital link (new)                              frozen DeepSC receiver
tokens -> encoder -> channel_encoder -> PowerNormalize -> t -> Q -> Gray -> [LDPC enc] -> M-PAM -> h, n -> ZF -> demap -> [BP dec] -> un-Gray -> Q^-1 -> t_hat -> channel_decoder -> decoder -> greedy -> BLEU
                                                       [B,L,16] idx in [0,15] 4 bit/value    1 or 4 bit/real use              hard bits or LLR
```

| Variant | Code | Modulation (per real dimension) | Receiver decision | Real channel uses / token |
|---|---|---|---|---|
| **C0** analog | - | 16 real values sent as is | - | 16 |
| **Q0** | - | none (quantise -> dequantise, no channel) | - | - (ceiling) |
| **U-BPSK** | none | BPSK, 1 bit/use | hard | 64 |
| **U-PAM16** | none | 16-PAM Gray, 4 bit/use | hard (nearest level) | **16 (= analog)** |
| **C-BPSK** | LDPC (3,6) n = 1008, R = 1/2 | BPSK | Gaussian LLR (CSI-aware) + BP | 128 (+ block padding) |
| **C-PAM16** | LDPC (3,6) n = 1008, R = 1/2 | 16-PAM Gray | max-log per-bit LLR (CSI-aware) + BP | 32 (+ block padding) |

Because the repo maps consecutive real pairs onto one complex symbol, BPSK on both real dimensions is QPSK per complex use and 16-PAM on both is 256-QAM per complex use. Nothing in the chain depends on that view; all rates are quoted per real dimension.

---

## 2. Repository layout

```
DeepSC-quantized/
  quantized_deepsc/
    __init__.py
    config.yaml            # single source of truth (below)
    deepsc_io.py           # load checkpoint/data; encode() -> (t, pad_mask); decode(t_hat) -> token ids; bleu()
    quant.py               # quantise / dequantise / gray / ungray / levels() / bit_labels()
    framing.py             # [B,L,16] idx -> per-sentence bit stream -> 504-bit info blocks (+mask); inverse
    modulation.py          # pam_levels(M) unit energy, gray labels, modulate, hard_demod, llr_maxlog
    channel.py             # awgn(), rayleigh_zf(per_sentence=True) -> (y, h2); mirrors repo pair convention
    ldpc.py                # (3,6)-regular H, systematic G (cached), batched torch BP; encode/decode/syndrome
    chain.py               # the variants c0/q0/u_bpsk/u_pam16/c_bpsk/c_pam16 between encode() and decode()
    calibrate.py           # Stage 1: extract t, calibrate A, histogram -> results/s1/
    run_baseline.py        # Stage 0: analog curves (repo-original h per batch, and h per sentence), AWGN + noiseless
    run_digital.py         # Stages 2-4: any variant x channel x SNR grid -> CSV (rewritten after every SNR point)
    run_compare.py         # Stage 5: fairness views, crossover, thresholds, figures, summary.md
    plot_stages.py         # per-stage figures (baseline, bits-vs-BLEU, uncoded/coded BLEU and BER/FER)
  run_dev_sweeps.sh        # all dev-subset sweeps, sequential, logs in results/logs/
  run_full.sh              # same on the full test set (about 8 h); not run in the validation phase
  tests/                   # pytest acceptance checks listed per stage (python -m pytest tests)
  results/                 # CSV + PNG, one sub-folder per stage; every CSV header carries seed, ckpt hash, config
  REPORT.md                # filled progressively: conventions, acceptance table, final figures
```

All scripts are run from `DeepSC-quantized/` as `python -m quantized_deepsc.<module>`.

`deepsc_io.py` adds `../DeepSC-continous` to `sys.path` and changes the working directory for the duration of data loading (the repo's `EurDataset` uses the relative path `europarl/...`). Nothing in `DeepSC-continous/` is edited.

`config.yaml` defaults:

```yaml
repo_dir: ../DeepSC-continous
checkpoint: checkpoints/deepsc-Rayleigh/checkpoint_80.pth
batch_size: 64                 # same as performance.py
max_len: 30                    # greedy decode length, as performance.py
dev_subset: 5000               # first N test sentences for sweeps; null = full 120 726
b: 4                           # bits per latent value
D: 16                          # real latents per token
A: null                        # filled by Stage 1 (expected ~4.54 for batch 64, pads included)
percentile: 99.9
modulations: [bpsk, pam16]
ldpc: {dv: 3, dc: 6, n: 1008, k: 504, bp_iters: 50, decoder: sum_product}   # or min_sum with 0.8 scaling
fading: {model: rayleigh, draw: per_sentence}                                # per_batch only for the reference run
channels: [awgn, rayleigh]
snr_grid_db: [0, 3, 6, 9, 12, 15, 18]                # matches performance.py
snr_fine_db: [-3, -1, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 20, 22, 24]  # added around waterfalls
seed: 1234
```

---

## 3. Conventions (fixed once, written into REPORT.md, used everywhere)

- **SNR label L** is the repo's: `sigma = SNR_to_noise(L) = 1/sqrt(2*10^(L/10))` per real dimension. Every digital constellation has **unit average energy per real dimension**. Hence both systems have the same Es/sigma^2 = L + 3 dB per real dimension at the same label, and the noise generator is shared. For BPSK (a = 1): Eb/N0 = a^2/(2*R*sigma^2) -> **Eb/N0 [dB] = L + 10*log10(1/R)**, i.e. L for uncoded, L + 3 for R = 1/2.
- **Complex pairing and fading**: consecutive real pairs form one complex symbol (repo convention). `h ~ CN(0,1)`, **one draw per sentence**; `r = h*x + n`, `n ~ CN(0, 2*sigma^2)`; ZF `y = r/h`, so the effective noise per real dimension is Gaussian with variance sigma^2/abs(h)^2 *conditional on h*. `rayleigh_zf` returns `(y, h2)` with `h2 = abs(h)^2` so the demapper can use it.
- **Demapper** (standard coherent receiver, knows h because it performed ZF): BPSK `L = 2*y*h2/sigma^2`; M-PAM max-log `L(c_k) = max_{v in V_k^0} l(y,v) - max_{v in V_k^1} l(y,v)` with `l(y,v) = -h2*(y-v)^2/(2*sigma^2)`. Over AWGN set h2 = 1. (A switch `csi: false` that drops the h2 factor is allowed for later experiments but produces no result in this plan.)
- **Sign**: `L > 0 <=> bit 0`, identical in demapper and LDPC decoder.
- **Gray labelling**: `bits = idx XOR (idx >> 1)`, MSB first, shared by the quantiser (`quant.py`) and the constellation (`modulation.py`); for 16-PAM the 4 Gray bits of one latent value map directly onto one 16-PAM symbol.
- **Batching**: batch 64, test order, per-batch `PowerNormalize`, pads transmitted by both systems; BLEU is computed on decoded tokens up to `<END>` so pads never enter the metric.
- **BLEU**: `sentence_bleu([target], decoded)` (reference = target sentence), no smoothing, averaged over sentences; BLEU-1 primary, BLEU-2..4 secondary. Note the original `performance.py` passes the arguments the other way round; use the corrected order for both systems.

---

## 4. Stage 0 - Harness and analog baseline

**Implement** `deepsc_io.py` and `run_baseline.py`.
- `encode(sents)`: `encoder -> channel_encoder -> PowerNormalize`, exactly the lines in `greedy_decode`; returns `t [B,L,16]` and `pad_mask [B,L]`.
- `decode(t_hat)`: `channel_decoder` followed by the greedy loop copied from `greedy_decode` (with the channel removed). One shared function for both systems.
- `run_baseline.py` runs C0 at `snr_grid_db` over (a) `utils.Channels.Rayleigh` unchanged (h per batch, reference only), (b) `channel.rayleigh_zf` with h per sentence, (c) AWGN, plus (d) noiseless (no channel) -> **C_inf**. Two passes with different seeds, averaged (as `epochs=2` in `performance.py`).

**Outputs.** `results/s0_baseline_continuous.csv` (columns: channel, h_draw, snr_db, bleu1..4, n_sentences, seed, ckpt_sha256), `results/s0_baseline.png`.

**Acceptance.** BLEU-1 non-decreasing in SNR for (b) and (c); (a) and (b) agree within +-0.02 BLEU-1 at >= 12 dB; C_inf >= every noisy point. Record all numbers - nothing to compare against exists yet.

**Runtime budget.** Encoder pass ~10 s; greedy decode is the cost (29 decoder steps x 1 887 batches). Estimate on the dev subset first and extrapolate before launching the full set.

---

## 5. Stage 1 - Latent extraction and quantiser calibration

**What.** Run `encode()` on the full test set at batch 64; keep the tensor exactly as transmitted (pads included).

**Compute and save** (`results/s1_calibration.json`): std(t), A (99.9-th percentile of abs(t)), Delta = 2A/(2^b - 1), q = Delta/sqrt(12), clipped fraction, level-index histogram and its entropy, per-Gray-bit marginals. Also the non-pad-only variant of each, for the record.

**Plot.** `results/s1_latent_hist.png`: histogram of t with the 16 level positions.

**Acceptance.** Values within +-2 % of row 1 of section 0.1 (A ~ 4.54, Delta ~ 0.605, entropy ~ 2.61 bit); clipped fraction 0.1 %. If A comes out near 3.8, the extraction ran at batch 1; if std comes out near 1.24, pads were excluded - both are the wrong tensor for this plan.

---

## 6. Stage 2 - Quantiser, Gray labelling, quantisation-only round trip (Q0)

**Implement** `quant.py`, vectorised over any shape:
```
idx  = clip(round((t + A)/Delta), 0, 2^b - 1)
val  = -A + idx*Delta
bits = gray(idx) = idx XOR (idx >> 1)        # b bits, MSB first
idx  = ungray(bits)
levels() -> 2^b values;  bit_labels() -> [2^b, b] array (used by modulation.py)
```

**Experiment E2.1 (Q0 ceiling).** `encode -> quantise -> dequantise -> decode`, no channel, full test set. Report BLEU-1..4 and latent MSE E[(t_hat - t)^2] (expect ~ q^2 = 0.0305 at b = 4). Repeat for b in {2, 3, 4, 5, 6, 8} with A fixed and Delta = 2A/(2^b - 1) -> `results/s2_bits_vs_bleu.csv/png`.

**Tests.** `ungray(gray(idx)) == idx` for all 2^b indices; quantiser never overflows for abs(t) >> A; `abs(dequantise(quantise(t)) - t) <= Delta/2` for abs(t) <= A.

**Acceptance.** BLEU-1(b) increasing in b and within 0.01 of C_inf at b >= 6. Record the gap C_inf - Q0(b = 4): this is the pure cost of 4-bit rounding and the plateau every digital curve must reach at high SNR.

---

## 7. Stage 3 - Uncoded digital link (hard decisions)

**Implement** `modulation.py` and `channel.py`.
- `pam_levels(M)`: `(2k - (M-1))*d`, `d = sqrt(3/(M^2 - 1))` so E[x^2] = 1 (BPSK: +-1; 16-PAM: d = 0.1085). Gray labels from `quant.bit_labels()` so that latent level k <-> 16-PAM symbol k.
- `modulate(bits, M)`, `hard_demod(y, M)` (nearest level -> Gray bits), `llr_maxlog(y, M, sigma2, h2)` (section 3).
- `channel.awgn(x, sigma)`, `channel.rayleigh_zf(x, sigma, per_sentence=True)` -> `(y, h2)`; pair convention as `Channels.Rayleigh`; per-batch draw available for the Stage-0 reference only.

**Validation V3.0 (before any BLEU run).** Uncoded BPSK, >= 10^6 bits per point, `snr_grid_db`:
- AWGN: BER = Q(1/sigma) = Q(sqrt(2*10^(L/10)))
- Rayleigh-ZF, h per symbol: BER = 0.5*(1 - sqrt(g/(1 + g))), g = 10^(L/10)
- 16-PAM AWGN: SER = 2*(1 - 1/M)*Q(d/sigma)

Simulated values within 5 % of closed form -> the channel, ZF, SNR convention and Gray labels are right.

**Experiment E3.1.** U-BPSK and U-PAM16 over AWGN and Rayleigh-ZF (h per sentence), `snr_grid_db` + `snr_fine_db`, dev subset then full set. Report BER, SER (PAM), level-error rate, latent MSE, BLEU-1..4 -> `results/s3_uncoded.csv`, `results/s3_uncoded_bleu.png`, `results/s3_uncoded_ber.png`.

**Acceptance.** BLEU-1 -> Q0 plateau as SNR -> high; U-PAM16 (equal channel uses) can be read directly against C0 in the same figure.

---

## 8. Stage 4 - LDPC-coded digital link

**Implement** `ldpc.py`:
- (3,6)-regular Gallager H with n = 1008, k = 504 (random column-permutation construction, remove 4-cycles if cheap); systematic G by Gaussian elimination over GF(2); cache `H, G, seed` to `results/ldpc_1008_504.npz`.
- Batched log-domain sum-product BP in torch (tanh rule with clamping, or scaled min-sum 0.8), 50 iterations, early stop on zero syndrome. `decode(llr) -> (bits, converged, n_iters)`; expose `H` and `syndrome(bits)`.
- Optional cross-check on ~1 000 blocks against `pyldpc` (`pip install pyldpc`).

**Implement** `framing.py`: per sentence, `[L,16]` indices -> `L*64` Gray bits -> zero-pad to a multiple of 504 -> blocks; keep the mask to strip padding at the receiver. Option `shorten: true` sets the LLR of known padding bits to +inf at the decoder (standard shortening; default off). Record `info_bits`, `coded_bits`, `channel_uses` per sentence for Stage 5.

**Validation V4.0 (mandatory before any fading run).** AWGN, BPSK, `L = 2*y/sigma^2`, 2 000+ blocks per point: BER ~ 1e-3 at Eb/N0 = 2.0 dB (label L = -1 dB for R = 1/2; literature value for this code/length ~ 8.5e-4), BER < 1e-4 at Eb/N0 = 3 dB. If off by more than 3x, fix the code or decoder first. Also: `syndrome(encode(u)) == 0` for random u; noiseless decode returns the codeword in 1 iteration.

**Experiment E4.1.** C-BPSK and C-PAM16 over AWGN and Rayleigh-ZF (h per sentence), `snr_grid_db` + `snr_fine_db`. Report post-decoding BER, FER, latent MSE, BLEU-1..4 -> `results/s4_coded.csv`, `results/s4_coded_bleu.png`, `results/s4_coded_ber_fer.png`.

**Acceptance.** A waterfall exists for each variant; above it BLEU-1 equals the Q0 plateau within 0.01; the SNR at BER = 1e-2 and at BLEU-1 = 0.9*Q0 is extracted and stored in `results/s4_thresholds.json`.

---

## 9. Stage 5 - Comparison with the analog link (fairness views)

The two systems spend different numbers of channel uses per token (table in section 1). Report **three views**, each as CSV + figure with the channel-uses ratio and the SNR convention in the caption:

1. **Equal SNR per channel use** (native label L). Digital C-BPSK will look best; state its 8x bandwidth.
2. **Equal energy per token.** x-axis = L + 10*log10(uses_digital / uses_analog): U-BPSK +6.0 dB, C-PAM16 +3.0 dB, C-BPSK +9.0 dB, U-PAM16 0 dB, each increased by the measured block-padding overhead (10*log10(coded_bits_actual / (128*L))). Digital curves shift right; this is the honest comparison.
3. **Equal channel uses.** U-PAM16 vs C0 only (16 real uses per token in both). This is the only like-for-like replacement.

**Runs.** C0 (Stage 0, h per sentence), Q0 (horizontal line), U-BPSK, U-PAM16, C-BPSK, C-PAM16 at `snr_grid_db` + `snr_fine_db`, Rayleigh; AWGN as a secondary figure.

**Outputs.** `results/s5_bleu_vs_snr_view{1,2,3}.csv/png`; `results/s5_crossover.json` = for each digital variant and each view, the SNR where its BLEU-1 crosses C0 (linear interpolation on the fine grid), or `null` if it never does; optional `results/s5_mse_to_bleu.png` (latent MSE vs BLEU-1 pooled over all runs, a useful diagnostic of how the frozen decoder tolerates interface noise).

**What to look for.** C0 degrades gracefully, digital variants have a cliff then a plateau at Q0. The number the thesis needs is the **view-2 crossover** for C-PAM16 and C-BPSK, and whether U-PAM16 (view 3) ever matches C0.

---

## 10. Pitfalls checklist

- [ ] One h per **sentence** in the comparison. Per batch (repo default) gives too few realisations; curves become non-monotone.
- [ ] Calibrate A on the transmitted tensor (batch 64, pads included). A ~ 3.8 or std ~ 1.24 means the wrong tensor.
- [ ] Transmit pads in both systems; never count them in channel uses for one system only.
- [ ] Do not import `performance.py` (sklearn import error, hard-coded path). Use the corrected BLEU argument order for both systems.
- [ ] `torch.load(..., map_location=device)` of the state_dict is fine on torch 2.11 (`weights_only` default); the model must be built with vocab 57 975.
- [ ] Same Gray table in `quant.py` and `modulation.py`; unit-test that latent level k maps to 16-PAM symbol k and back.
- [ ] LLR sign `L > 0 <=> bit 0` in demapper and decoder; clamp `tanh(L/2)` away from +-1 in the BP.
- [ ] Clip `idx` to [0, 2^b - 1] before Gray mapping (0.1 % of values exceed +-A).
- [ ] Run V3.0 and V4.0 before trusting any fading or BLEU number.
- [ ] Unit-energy constellations and the repo's `SNR_to_noise` everywhere; state Es/sigma^2 = L + 3 dB once in REPORT.md.
- [ ] Fix seeds; write seed, checkpoint SHA-256 and config into every CSV header.
- [ ] Dev subset (5 000 sentences) for sweeps; full test set only for the final curves.

---

## 11. Deliverables

1. `REPORT.md`: conventions (section 3), acceptance table (Stages 0-4, expected / obtained / pass), runtime notes.
2. Figures: Stage 0 baseline, Stage 2 bits-vs-BLEU, Stage 3/4 BER and BLEU curves, the three Stage 5 views with captions stating channel uses per token and the SNR convention.
3. `results/s5_crossover.json` and `results/s4_thresholds.json`.
4. One-paragraph summary: at what SNR (view 2) each digital variant overtakes the analog DeepSC on Europarl, and how far U-PAM16 (equal channel uses) sits below it.

---

## Appendix A - Verified facts for a later reproduction of the Wakim paper (not part of this plan)

Checked numerically on 2026-09-09 with 1e7 Monte-Carlo samples; kept here so nothing has to be re-derived later.

- The paper's worked LLR example (y = 0.5 -> 4.0 / 2.41; y = 30 -> 238.9 / 0.10) and Table III (clipped 0.437, Student-t(2) 0.454, per-symbol CSI 0.565) are reproduced **exactly** with BPSK amplitude **a = 0.5** and **sigma^2 = 1/(2*10^(SNR/10))**, i.e. the repo's `SNR_to_noise`. So the paper's "L dB" is a^2/sigma^2 = L - 3 dB per real dimension, which is **6 dB below this plan's axis** at the same label (this plan uses a = 1). Shift the paper's thresholds (Table IV: 12.42 / 9.97 / 8.66 dB) by -6 dB before comparing to any curve produced here.
- Table II's KS = 0.2040 for the Gaussian is against a Gaussian with the **sample** standard deviation (fitted sigma, inflated by the heavy tail); against the nominal sigma the KS statistic is 0.071. KS vs Student-t(2, scale sigma) ~ 4e-4 here (paper 2.8e-3); Hill index 1.97 (paper 1.978); Table I moment growth reproduces qualitatively (mu4 from 373 at 1e4 to 2e6 at 1e7 samples; values are seed-dependent, as the paper says).
- The paper's checkpoint differs from this one (vocab 22 234 vs 57 975; A 4.387 vs 4.54; index entropy 2.739 vs 2.61 bit; levels 7+8 mass 59.7 % vs 65.2 %), so its source-prior numbers (Table V) will not transfer; they would have to be re-measured from `results/s1_calibration.json`.
- The demapper interface in section 3 (`llr(y, sigma2, h2) -> L`, `ldpc.decode(L)`, `ldpc.H`, `ldpc.syndrome`) is already what a Student-t demapper or an ORBGRAND decoder would plug into.
