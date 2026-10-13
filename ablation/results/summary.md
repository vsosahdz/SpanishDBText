# Evaluator ablation — summary

LLM scores before the programmatic check. Distribution on the complete
translations (without the truncations 7, 24 and 41).

| Config | Description | n | Mean | Median | SD | % ≥95 | % ≥90 | Mode (%) | Distinct values | Cost US$ |
|---|---|---|---|---|---|---|---|---|---|---|
| G1L1 | production: guide + length evidence | 177 | 95.75 | 96 | 2.01 | 80.2 | 100.0 | 95 (28.2) | 8 | 0.7809 |
| G0L1 | no guide, with length evidence | 177 | 96.79 | 97 | 0.99 | 96.6 | 100.0 | 97 (81.9) | 7 | 0.7371 |
| G1L0 | with guide, no length evidence | 177 | 94.92 | 95 | 1.7 | 76.3 | 100.0 | 95 (37.3) | 8 | 0.8297 |
| G0L0 | no guide and no length evidence | 177 | 96.44 | 97 | 1.5 | 91.5 | 100.0 | 97 (78.5) | 8 | 0.735 |

## Known truncations

*Detected* = LLM completeness ≤ 70. *Mentioned* = the truncation or omission appears in the list of errors.

| Config | ID | Ratio | LLM completeness | LLM overall | Adjusted overall | Detected | Mentioned |
|---|---|---|---|---|---|---|---|
| G1L1 | 7 | 0.362 | [36, 36, 36] | [50, 50, 50] | [50, 50, 50] | 3/3 | 3/3 |
| G1L1 | 24 | 0.262 | [26, 26, 26] | [40, 40, 30] | [40, 40, 30] | 3/3 | 3/3 |
| G1L1 | 41 | 0.788 | [79, 70, 79] | [85, 80, 85] | [85, 80, 85] | 1/3 | 3/3 |
| G0L1 | 7 | 0.362 | [36, 36, 36] | [50, 60, 50] | [50, 60, 50] | 3/3 | 3/3 |
| G0L1 | 24 | 0.262 | [30, 30, 30] | [50, 50, 50] | [50, 50, 50] | 3/3 | 3/3 |
| G0L1 | 41 | 0.788 | [70, 70, 70] | [80, 80, 80] | [80, 80, 80] | 3/3 | 3/3 |
| G1L0 | 7 | 0.362 | [90, 95, 95] | [92, 85, 95] | [72, 72, 73] | 0/3 | 2/3 |
| G1L0 | 24 | 0.262 | [95, 95, 95] | [95, 94, 96] | [69, 69, 69] | 0/3 | 2/3 |
| G1L0 | 41 | 0.788 | [99, 99, 99] | [98, 98, 97] | [90, 90, 90] | 0/3 | 3/3 |
| G0L0 | 7 | 0.362 | [90, 90, 90] | [90, 85, 92] | [72, 72, 72] | 0/3 | 3/3 |
| G0L0 | 24 | 0.262 | [90, 90, 90] | [85, 90, 92] | [68, 68, 68] | 0/3 | 2/3 |
| G0L0 | 41 | 0.788 | [100, 99, 99] | [97, 97, 97] | [88, 89, 89] | 0/3 | 0/3 |

## Paired difference against production (overall, complete translations)

| Config | Articles | Mean difference (config − G1L1) | 95% bootstrap CI |
|---|---|---|---|
| G0L1 | 59 | 1.04 | [0.68, 1.42] |
| G1L0 | 59 | -0.82 | [-1.13, -0.52] |
| G0L0 | 59 | 0.69 | [0.23, 1.14] |
