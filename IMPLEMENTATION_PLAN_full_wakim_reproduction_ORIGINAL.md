# Implementation Plan: Digital DeepSC with Matched LLRs (Wakim–Ismail–Sarieddeen 2026)

**Goal.** Reproduce the digital DeepSC link of Wakim et al. on top of the *existing, already-trained* continuous DeepSC (Europarl), verify the reproduction against the paper's reported numbers, and then produce the comparison the paper itself leaves open: **BLEU / sentence-similarity vs SNR for digital DeepSC against the continuous DeepSC results we already have.**

**Non-goals.** Do not retrain DeepSC. Do not change the semantic or channel encoder. Do not implement VQ or ORBGRAND here (but keep the LLR interface clean so ORBGRAND can be dropped in later — see §9).

**Working principle.** Every phase ends with a numeric acceptance check against the paper. If a check fails by more than the stated tolerance, stop and report; do not proceed on a broken foundation.

---

## 0. Inputs you will find / must locate

| Item | Where | Notes |
|---|---|---|
| Trained continuous DeepSC checkpoint | existing project (Rayleigh-trained, `d_model=128, d_ff=512, L=4, H=8`) | This is the 4-layer *reference implementation* config, not the 3-layer TSP-paper table. Record which checkpoint is used. |
| Europarl preprocessed data + vocab (22,234 tokens) | existing project | Same test split used for the continuous results. |
| Existing continuous-DeepSC BLEU-1 vs SNR results | existing project (SNR ∈ {0,3,6,9,12,15,18} dB) | These are the baseline to compare against. Copy them into `results/baseline_continuous.csv` untouched. |
| Evaluation script (BLEU + sentence similarity) | existing `performance.py` or equivalent | Reuse it verbatim so metrics are comparable. |

Before anything else: **re-run the continuous baseline once** through the harness at 3 SNR points and confirm it matches the stored results (tolerance ±0.01 BLEU-1). This locks the evaluation pipeline.

---

## 1. Repository layout

```
quantized_deepsc/
  config.yaml                 # all constants below, single source of truth
  latent/
    extract.py                # dump channel-encoder outputs t for test set
    calibrate.py              # sigma_t, A, Delta, histogram, entropy
  quant/
    scalar_quantizer.py       # quantize / dequantize / index<->bits (Gray)
  channel/
    fading.py                 # Nakagami-m gain, AWGN, ZF
    noise_id.py               # KS, Hill index, moment growth (Table I/II)
  demapper/
    llr.py                    # gaussian, gaussian_clipped, student_t (binary + PAM max-log)
    gmi.py                    # GMI and calibration error
    prior.py                  # source prior: P(k), MAP, MMSE, additive LLR term
  code/
    ldpc.py                   # (3,6)-regular, n=1008, R=1/2; sum-product BP
    framing.py                # tokens -> 64-bit words -> 504-bit info blocks, padding
  e2e/
    run_digital.py            # full chain -> BLEU / similarity vs SNR
    run_continuous.py         # thin wrapper around existing eval (sanity)
  results/                    # CSVs + PNGs, one subfolder per phase
  tests/                      # acceptance checks below as pytest
  REPORT.md                   # auto-filled table of checks + final figures
```

`config.yaml` defaults (from the paper unless marked *decision*):

```yaml
b: 4                       # bits per latent value
D: 16                      # real latent values per token (DeepSC Dense-16)
A: null                    # filled by calibrate.py (paper: 4.387)
percentile: 99.9           # for A
ldpc: {dv: 3, dc: 6, n: 1008, k: 504, bp_iters: 50}
modulations: [bpsk, pam16]
fading: {model: nakagami, m: 1.0}          # m=1 -> Rayleigh; sweep m in {0.5,1,2,5}
fading_draw: per_sentence                  # paper: per sentence, NOT per batch
snr_train_reference: null                  # not applicable: DeepSC is frozen
snr_grid_db: [-2,0,2,4,6,8,10,12,14,16,18,20,22,24]
llr_models: [gaussian, gaussian_clip2, student_t]
seed: 1234
```

---

## 2. Phase 1 — Latent extraction and quantizer calibration

**What.** Run the frozen DeepSC transmitter on the full test set. Capture `t` = output of the channel encoder **after** power normalization (i.e., exactly the tensor that would be sent over the analog channel). Shape per sentence: `[L, 16]`. *Decision to record:* if the repo's `PowerNormalize` is applied per-batch rather than per-sentence, note it; calibrate on whatever is actually transmitted.

**Compute and save** (`results/p1_calibration.json`):
- `sigma_t` = std of all latent values (paper: **1.1943**)
- `A` = 99.9th percentile of |t| (paper: **4.387**)
- `Delta = 2A/(2^b-1)` (paper: **0.5849**), `q = Delta/sqrt(12)` (paper: 0.169)
- histogram of |t| and of t; fraction clipped (|t| > A) — should be ≈0.1%
- level-index histogram after quantization; entropy of index (paper: **2.739 bits**); mass on levels 7+8 (paper: **59.7%**); per-Gray-bit marginals P(c_j = 1) (paper: bit 1 → **0.9567**, bit 0 → **0.485**)

**Acceptance.** `sigma_t` within ±5%, `A` within ±5%, entropy within ±0.1 bit, levels-7-8 mass within ±5 pp. If `sigma_t` is far from 1.19, check whether you captured pre- vs post-normalization.

**Plot.** `results/p1_latent_hist.png`: histogram of t with the 16 level positions overlaid.

---

## 3. Phase 2 — Quantizer, Gray labelling, noiseless round-trip

**Implement** in `scalar_quantizer.py`:
```
idx  = clip(round((t + A)/Delta), 0, 2^b-1)          # int in [0,15]
val  = -A + idx*Delta                                 # dequantize
bits = gray(idx)  = idx XOR (idx >> 1)                # 4 bits, MSB first
idx  = ungray(bits)
```
Vectorized over `[L,16]`. Also provide `levels()` → the 16 values, and `bit_labels()` → 16×4 array (needed by the PAM demapper and the prior).

**Experiment E2.1 (quantization-only loss).** Chain: DeepSC encoder → quantize → dequantize → DeepSC decoder, **no channel**. Report BLEU-1..4 and sentence similarity. This is the ceiling of the digital system; the gap to continuous-noiseless DeepSC is the pure cost of 4-bit rounding. Also run b ∈ {3,4,5,6} for a rate–quality curve (`results/p2_bits_vs_bleu.csv`).

**Experiment E2.2 (σ_L1 vs σ_L2 crossover, optional).** Over AWGN, compare (a) analog transmission of `val` with no decision at the receiver and (b) hard level decision at the receiver. Locate the SNR where (b) overtakes (a) in BLEU. Paper: 12–13 dB.

**Acceptance.** E2.1 BLEU-1 loss vs continuous-noiseless ≤ a few points at b=4 (no paper number; record it). Round-trip `ungray(gray(idx)) == idx` for all 16 (unit test).

---

## 4. Phase 3 — Channel, ZF, and effective-noise identification

**Implement** `fading.py`:
- `|h|^2 ~ Gamma(shape=m, scale=1/m)` so `E|h|^2 = 1`; `h = |h| * exp(j*phase)`; **one h per sentence** (or per LDPC block — record which; must be per-sentence for the Student-t statistics to match).
- Real-baseband handling: with BPSK on the real axis, use `r = |h| x + n_real` after phase compensation; `n_real ~ N(0, σ²)`. Define SNR **per real dimension** as `a²/σ²` with `a` the BPSK amplitude — **and write this definition in REPORT.md**, because the paper's SNR convention is not unambiguous (its 6 dB LLR example is reproduced by `a=0.5, σ²=0.125`, i.e., `a²/σ² = 2`, which is 3 dB, so either the SNR is defined per complex symbol or `a` differs). Reproduce the paper's *numbers* (Tables III–IV) under whichever convention matches, then state it.
- ZF: `z = r/|h| - x`.

**Implement** `noise_id.py` on ≥10⁷ samples of `z` at 6 dB, Rayleigh:
- KS statistic vs `N(0,σ²)` and vs Student-t(ν=2m, scale σ). Paper: **0.2040** (Gaussian) vs **0.0028** (t(2)).
- Hill tail-index estimator on |z| (top 1% order statistics). Paper: **1.978** (theory 2).
- Sample moments µ̂2, µ̂4, µ̂6 at N ∈ {10⁴,10⁵,10⁶,10⁷}: must **grow without bound** (paper Table I: µ̂4 from 186 to 4.3×10⁵).
- Sweep m ∈ {0.5, 1, 2, 5, 10}: Hill index should track 2m; KS vs t(2m) stays ≈3×10⁻³.
- Rician K ∈ {0,5,10,15,20} dB: Hill index rises from ≈2 to ≈8.7 at K=20 dB.

**Acceptance.** KS(t(2)) < 0.01 and KS(Gauss) > 0.15; Hill ∈ [1.9, 2.1] at m=1; moment growth monotone in N.

**Plots.** `p3_density_logscale.png` (empirical vs Gaussian vs t(2)), `p3_qq.png`, `p3_hill_vs_m.png`.

---

## 5. Phase 4 — Demappers, GMI, calibration

**Implement** `llr.py` (all take `y`, `a`, `σ²`, and for t also `ν`):
- `gaussian`: `Λ = 2 a y / σ²`
- `gaussian_clip2`: same, clipped to ±2 — *the practical baseline the paper beats by 1.31 dB*
- `student_t` (binary): `Λ = (ν+1)/2 · [ln(1 + (y+a)²/(νσ²)) − ln(1 + (y−a)²/(νσ²))]`
- `student_t_pam` / `gaussian_pam` (16-PAM, per-bit max-log): `Λ(c_k) = max_{v∈V_k^0} ℓ(y,v) − max_{v∈V_k^1} ℓ(y,v)` with `ℓ_t = −(ν+1)/2 · ln(1+(y−v)²/(νσ²))`, `ℓ_g = −(y−v)²/(2σ²)`; `V_k^β` from `bit_labels()`.
- Optional reference: `gaussian_persymbol_csi`: `Λ = 2 a |h|² y_pre_zf / σ²` (conditions on h instead of marginalizing) — paper Table III last row, GMI 0.565.

Convention: `Λ > 0 ⇔ bit 0`. Sign convention must be identical for LDPC and for ORBGRAND later.

**Implement** `gmi.py`: `GMI = 1 − E[log2(1 + exp(−Λ̃))]`, `Λ̃ = (1−2c)Λ`; calibration error = mean |P̂(c=1 | Λ bin) − (1+e^Λ)^{-1}| over 50 equal-count bins.

**Experiment E4.1.** Rayleigh, ZF, 6 dB, BPSK, ≥10⁷ bits: reproduce Table III.

| Model | Calib. err. | GMI (paper) |
|---|---|---|
| gaussian | 0.0662 | **−0.216** |
| gaussian_clip2 | 0.0456 | 0.437 |
| student_t(2) | 0.0057 | **0.454** |
| per-symbol CSI | 0.0064 | 0.565 |

**Experiment E4.2.** GMI gain (t − gaussian_clip2) vs m ∈ {0.5,1,2,5}: paper 3.2 → 0.76 → … → 0.016 bits. Also vs Rician K: gain goes **negative** above ≈5 dB.

**Experiment E4.3.** GMI vs SNR ∈ [−2, 12] dB for the Gaussian LLR: must be negative across the range (paper's claim).

**Acceptance.** Gaussian GMI < 0 at 6 dB; t(2) GMI within ±0.02 of 0.454; ordering of models as in the table.

**Plot.** `p4_llr_transfer.png` (Λ vs y, both models, near region and tail region), `p4_gmi_vs_m.png`.

---

## 6. Phase 5 — LDPC and coded performance

**Implement** `ldpc.py`:
- Random (3,6)-regular Gallager parity-check H, n=1008, k=504, 4-cycle-free if feasible; systematic encoder via Gaussian elimination (store G once, cache to disk with the seed).
- Sum-product BP in the log domain (tanh rule or Jacobian log), 50 iterations, early stop on syndrome = 0. Vectorize across a batch of codewords (numpy or torch).
- Also expose `decode(llr) -> (bits, success_flag, n_iters)`.

**Validation V5.0 (mandatory before any fading run).** AWGN, BPSK, Gaussian LLR: BER at Eb/N0 = 2.0 dB must be ≈ **8.5×10⁻⁴** (paper's validation point). If off by >3×, fix the code/decoder before continuing.

**Implement** `framing.py`: sentence → L tokens × 64 bits → concatenate → pad with zeros to a multiple of 504 → LDPC blocks. Keep a mask to strip padding at the receiver. For fairness bookkeeping, record `bits_per_token = 64`, `coded_bits_per_token = 128`, `channel_uses_per_token` = 128 (BPSK) or 32 (16-PAM).

**Experiment E5.1 (Table IV reproduction).** Rayleigh, ZF, BPSK, per-sentence h, LDPC n=1008 R=1/2, SNR grid, ≥2000 blocks per point. Report post-LDPC BER and FER for each LLR model. Extract SNR at BER=10⁻².

| Model | SNR @ BER 1e-2 (paper) |
|---|---|
| gaussian | 12.42 dB |
| gaussian_clip2 | 9.97 dB |
| student_t(2) | **8.66 dB** |

Also confirm: gaussian never reaches 10⁻⁴ even at 24 dB; all corrected models have a waterfall between 8 and 10 dB.

**Experiment E5.2.** Same over AWGN (m→∞): t-LLR vs Gaussian difference must be ≈0 (paper −0.07 dB) — proves the gain is not a harness artefact.

**Experiment E5.3.** 16-PAM with max-log per-bit LLRs: gain at BER 10⁻³ ≈ 2.36 dB.

**Experiment E5.4 (robustness).** Rates {1/2, 2/3, 3/4} × lengths {504, 1008, 2016}: gain vs clip2 should be present everywhere and grow with rate (paper: 1.24 → 7.19 dB).

**Acceptance.** E5.1 t(2) threshold within ±0.4 dB of 8.66; gain vs gaussian ≥ 3 dB; E5.2 |Δ| < 0.2 dB.

**Plot.** `p5_ber_fer_rayleigh.png` (both panels, all models), `p5_pam16.png`.

---

## 7. Phase 6 — Source prior (uncoded only)

**Implement** `prior.py`: `P(k)` = empirical level-index distribution from Phase 1 (per latent dimension, or pooled — try both; paper pools). Provide:
- MAP level decision: `argmax_k [log p(y|k) + log P(k)]` with `p(y|k)` Student-t around level value `v_k`.
- MMSE / posterior-mean estimate: `x̂ = Σ_k v_k P(k|y)`.
- Additive per-bit prior term for LLRs: `log P(c=0)/P(c=1)` from the per-bit marginals.

**Experiment E6.1 (Table V).** Uncoded 16-PAM over Rayleigh ZF at SNR ∈ {0,6,12,18} dB. Report symbol-error rate (ML vs MAP) and latent MSE (ML-hard vs MMSE). Paper reductions: 64.1% (0 dB) → 23.7% (18 dB).

**Experiment E6.2 (composition).** At 6 dB, GMI with channel-correction only (+0.378), prior only (+0.053), both (+0.432). Confirm bit position 0 (P=0.485) gains nothing.

**Do not** apply the prior on top of LDPC-coded bits: the paper shows coding scrambles the level distribution (entropy 2.739 → 4.000). Add a unit test that measures index entropy *after* LDPC encoding to reproduce this and document why the prior is uncoded-only.

---

## 8. Phase 7 — The comparison we actually want: BLEU vs continuous DeepSC

This is the paper's stated open direction ("transfer function from interface-noise magnitude to semantic similarity"). Design it carefully; the fairness protocol is the main risk.

### 8.1 Fairness protocol (write this into REPORT.md before running)

The two systems use **different numbers of channel uses per token**:

| System | Real values / token | Channel uses / token | SNR definition |
|---|---|---|---|
| Continuous DeepSC | 16 | 8 complex = 16 real | per real dimension (as in existing results) |
| Digital, BPSK, R=1/2 | 64 bits → 128 coded bits | 128 real (**8× more**) | per real dimension |
| Digital, 16-PAM, R=1/2 | 128 coded bits / 4 | 32 real (**2× more**) | per real dimension |
| Digital, uncoded 16-PAM + prior | 64 bits / 4 | 16 real (**equal**) | per real dimension |

Report **three** views, all as CSV + one figure each:
1. **Equal SNR per channel use** (what the paper's tables use). Digital-BPSK-LDPC will look best; state the 8× bandwidth cost in the caption.
2. **Equal energy per token**: shift the digital curves by 10·log10(channel_uses_ratio) dB (i.e., BPSK-LDPC by −9.03 dB, 16-PAM-LDPC by −3.01 dB). This is the honest comparison.
3. **Equal channel uses**: uncoded 16-PAM with the source prior (16 real per token, same as continuous) — this is the only configuration that is a like-for-like replacement, and it is where Wakim's prior result matters.

### 8.2 Runs

For each of the 7 continuous-baseline SNR points {0,3,6,9,12,15,18} dB plus the fine grid, Rayleigh (m=1), per-sentence h, full test set, seed fixed:

- **D1**: BPSK + LDPC(1008, 1/2) + `gaussian` LLR → BLEU-1..4, similarity
- **D2**: same with `gaussian_clip2`
- **D3**: same with `student_t`
- **D4**: 16-PAM + LDPC + `student_t_pam`
- **D5**: uncoded 16-PAM, ML hard decision, `student_t` likelihood
- **D6**: uncoded 16-PAM, MAP with prior
- **D7**: uncoded 16-PAM, MMSE posterior-mean with prior (feeds *continuous* estimates to the DeepSC decoder — closest to analog behaviour)
- **C0**: continuous DeepSC (existing results; re-run once to confirm)
- **Q0**: quantization-only ceiling from E2.1 (horizontal line)

Also for D3: repeat at m ∈ {0.5, 2, 5} to show the semantic-level gain shrinks with milder fading (mirrors E4.2).

### 8.3 What to look for (hypotheses to confirm or refute)

- D1 vs D3 should show a large BLEU gap around 8–12 dB (the 3.77 dB shifted into semantic space). Report the SNR at which each reaches BLEU-1 = 0.8.
- D3 above its waterfall should sit at the Q0 ceiling (digital plateau = quantizer quality, independent of SNR).
- C0 degrades gracefully; D3 has a cliff. Identify the crossover SNR under view (2) — below it the analog system wins, above it digital wins. **This crossover is the number the thesis needs.**
- D7 vs D5 vs D6: does the prior's 24–64% latent-MSE reduction translate to BLEU? Report the "transfer function": scatter of latent MSE (x) vs BLEU-1 (y) pooled across all D-runs and SNRs, `results/p7_transfer_mse_to_bleu.png`. Wakim reports this looks well defined; verify.

### 8.4 Outputs

- `results/p7_bleu_vs_snr_view{1,2,3}.csv/png` (one curve per D-run + C0 + Q0)
- `results/p7_similarity_vs_snr_view{1,2,3}.csv/png`
- `results/p7_crossover.json`: crossover SNRs under each view
- `results/p7_transfer_mse_to_bleu.png`

---

## 9. GRAND-ready interface (do not implement GRAND; just leave the hook)

`demapper/llr.py` must return, per codeword, a float array `Λ` with the documented sign convention, and `code/ldpc.py` must expose `H` and a `syndrome(bits)` function. A future `code/orbgrand.py` will consume `(Λ, H)` only. Add `tests/test_interface.py` asserting shapes, dtype, and that `decode` works from `Λ` alone (no access to `h`, `σ`, or the modulation). The reliability ordering used by ORBGRAND is `argsort(|Λ|)`; add a diagnostic that, at 6 dB Rayleigh, plots the rank of the bits that were actually in error under the Gaussian vs the matched LLR (expect errors ranked near the top for Gaussian, near the bottom for matched).

---

## 10. Pitfalls checklist

- [ ] Per-sentence fading draws. Per-batch (as in the DeepSC repo) gives too few realizations; curves will be non-monotone.
- [ ] Capture `t` *after* power normalization. If `sigma_t` ≠ ≈1.19, you captured the wrong tensor.
- [ ] Do not let padding bits enter BLEU: strip by mask before the DeepSC decoder.
- [ ] LLR sign convention consistent across demapper, LDPC, prior, and the future GRAND hook.
- [ ] Clip `idx` to [0,15] — values beyond ±A exist (0.1%) and must not overflow the Gray table.
- [ ] Validate LDPC on AWGN (V5.0) before trusting any fading number.
- [ ] Never apply the source prior to coded bits.
- [ ] State the SNR convention once in REPORT.md and use it everywhere; convert the continuous baseline to the same convention if it differs.
- [ ] Fix seeds; log the seed, checkpoint hash, and config in every CSV header.
- [ ] Use ≥10⁷ samples for noise-identification statistics; heavy tails need it.

---

## 11. Deliverables

1. `REPORT.md` with the acceptance-check table (paper value / reproduced value / pass-fail) for Phases 1–6.
2. The three Phase-7 comparison figures with captions that state channel-uses-per-token and SNR convention.
3. `results/p7_crossover.json`.
4. One-paragraph summary: at what SNR (view 2) does digital DeepSC with the matched LLR overtake continuous DeepSC on Europarl, and does the source prior close any of the gap in the equal-channel-use configuration.
