# Stage 5 summary (bleu1, Rayleigh ZF, h per sentence)

Q0 ceiling (b=4): 0.8689
Analog noiseless: 0.8223

| variant | uses/token | max BLEU-1 | SNR @ 0.9 Q0 | SNR @ 0.95 Q0 | view-2 shift | view-2 crossover vs C0 |
|---|---|---|---|---|---|---|
| C0 analog DeepSC | 16.0 | 0.83 | 7.85 | 14.30 | - | - |
| U-BPSK (uncoded, hard) | 64.0 | 0.86 | 11.10 | 14.35 | 6.02 dB | 20.68 dB |
| U-PAM16 (uncoded, hard) | 16.0 | 0.85 | 16.90 | 19.85 | 0.00 dB | 20.16 dB |
| C-BPSK (LDPC 1/2) | 130.5 | 0.87 | 7.10 | 10.15 | 9.11 dB | 19.64 dB |
| C-PAM16 (LDPC 1/2) | 32.6 | 0.85 | 16.80 | 19.90 | 3.09 dB | 23.09 dB |
| G-BPSK (CRC + ORBGRAND) | 74.6 | 0.87 | 10.50 | 13.60 | 6.68 dB | 20.53 dB |
