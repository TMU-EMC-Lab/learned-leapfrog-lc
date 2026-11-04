## Planned tests

| Family / test | In the letter | A/B | Ratio [95% seed interval] | Runs | Seeds | p runs (Holm) | p seed means (Holm) | Dagger |
|---|---|---|---|---|---|---|---|---|
| non-lc-interaction / H1-Duffing | repository only | (LF-U/LF)/(Res2/SIE) | 3.75 [3.2, 4.35] | 40 | 10 | 1.0e-05 (7.0e-05) | 9.8e-04 (0.0068) | † |
| non-lc-interaction / H1-DW | repository only | (LF-U/LF)/(Res2/SIE) | 8.26 [7.25, 9.41] | 40 | 10 | 1.0e-05 (7.0e-05) | 9.8e-04 (0.0068) | † |
| non-lc-interaction / H1-Morse | repository only | (LF-U/LF)/(Res2/SIE) | 4.09 [3.34, 5.17] | 40 | 10 | 1.0e-05 (7.0e-05) | 9.8e-04 (0.0068) | † |
| non-lc-interaction / H1-FPU | repository only | (LF-U/LF)/(Res2/SIE) | 4.34 [4.17, 4.52] | 40 | 10 | 1.0e-05 (7.0e-05) | 9.8e-04 (0.0068) | † |
| non-lc-interaction / H1-B1 | repository only | (LF-U/LF)/(Res2/SIE) | 1.77 [1.61, 1.98] | 40 | 10 | 1.0e-05 (7.0e-05) | 9.8e-04 (0.0068) | † |
| non-lc-interaction / H1-B10 | repository only | (LF-U/LF)/(Res2/SIE) | 1.9 [1.56, 2.32] | 20 | 10 | 1.0e-05 (7.0e-05) | 9.8e-04 (0.0068) | † |
| non-lc-interaction / H1-B20 | repository only | (LF-U/LF)/(Res2/SIE) | 2.24 [1.92, 2.6] | 20 | 10 | 1.0e-05 (7.0e-05) | 9.8e-04 (0.0068) | † |
| non-lc-midpoint / H2-Duffing | repository only | SIE/SIE-M | 1.95 [1.82, 2.09] | 80 | 20 | 1.0e-05 (7.0e-05) | 9.5e-07 (6.7e-06) | † |
| non-lc-midpoint / H2-DW | repository only | SIE/SIE-M | 7.75 [6.73, 8.85] | 80 | 20 | 1.0e-05 (7.0e-05) | 9.5e-07 (6.7e-06) | † |
| non-lc-midpoint / H2-Morse | repository only | SIE/SIE-M | 4.94 [4.28, 5.72] | 80 | 20 | 1.0e-05 (7.0e-05) | 9.5e-07 (6.7e-06) | † |
| non-lc-midpoint / H2-FPU | repository only | SIE/SIE-M | 2.78 [2.62, 2.94] | 80 | 20 | 1.0e-05 (7.0e-05) | 9.5e-07 (6.7e-06) | † |
| non-lc-midpoint / H2-B1 | repository only | SIE/SIE-M | 1.9 [1.81, 2] | 80 | 20 | 1.0e-05 (7.0e-05) | 9.5e-07 (6.7e-06) | † |
| non-lc-midpoint / H2-B10 | repository only | SIE/SIE-M | 2 [1.88, 2.12] | 40 | 20 | 1.0e-05 (7.0e-05) | 9.5e-07 (6.7e-06) | † |
| non-lc-midpoint / H2-B20 | repository only | SIE/SIE-M | 2.03 [1.88, 2.18] | 40 | 20 | 1.0e-05 (7.0e-05) | 9.5e-07 (6.7e-06) | † |
| non-lc-duffing-beta / H4 | repository only | (LF-U/LF)/(Res2/SIE) | 1.05 [0.991, 1.11] < 1.14 [1.07, 1.23] < 4.03 [3.4, 4.66] < 5.96 [4.4, 8.2] | 20 + 20 + 20 + 20 | 10 | ordering held (no p) | -- | |
| vi-third-circuit / H5a | Table VI | LF-U/LF | 1.03 [0.94, 1.13] | 20 | 10 | 0.23 (0.46) | 0.25 (0.51) |  |
| vi-third-circuit / H5b | repository only | SIE/SIE-M | 0.969 [0.934, 1.01] | 20 | 10 | 0.92 (0.92) | 0.92 (0.92) |  |
| tan-circuit / H6a | repository only | (LF-U/LF)/(Res2/SIE) | 2.19 [1.65, 2.97] | 40 | 10 | 2.0e-05 (4.0e-05) | 9.8e-04 (0.002) | † |
| tan-circuit / H6b | repository only | SIE/SIE-M | 0.264 [0.212, 0.332] | 40 | 10 | 1 (1) | 1 (1) |  |
| circuit / C1 | Fig. 3 | SIE-M/LF | 3.76 [3.16, 4.46] | 40 | 10 | 1.0e-05 (5.0e-05) | 9.8e-04 (0.0049) | † |
| circuit / C2 | Sect. 4.1 | Res2/SIE | 3.23 [2.76, 3.85] | 40 | 10 | 1.0e-05 (5.0e-05) | 9.8e-04 (0.0049) | † |
| circuit / C3 | repository only | SIE-M/SIE | 1.63 [1.33, 2.05] | 40 | 10 | 3.0e-05 (5.0e-05) | 9.8e-04 (0.0049) | † |
| circuit / C4 | Fig. 3; Sect. 4.1 | Res/LF | 4.43 [3.85, 5.04] | 40 | 10 | 1.0e-05 (5.0e-05) | 9.8e-04 (0.0049) | † |
| circuit / C5 | Fig. 3; Sect. 4.1 | RK4-net/LF | 3.75 [3.02, 4.69] | 40 | 10 | 1.0e-05 (5.0e-05) | 9.8e-04 (0.0049) | † |
| kcl-circuit / H7a | repository only | LF/LF-KCL | 5.1 [4.43, 5.83] | 40 | 10 | 1.0e-05 (2.0e-05) | 9.8e-04 (0.002) | † |
| kcl-circuit / H7b | Sect. 4.1 | LF/LF-KCL | 1.92 [1.8, 2.04] | 40 | 10 | 1.0e-05 (2.0e-05) | 9.8e-04 (0.002) | † |
| line / L1 | Fig. 3 | SIE-M/LF | 3.44 [3.06, 3.9] | 40 | 10 | 1.0e-05 (8.0e-05) | 9.8e-04 (0.0078) | † |
| line / L2 | Sect. 4.1 | Res2/SIE | 1.79 [1.7, 1.9] | 30 | 10 | 1.0e-05 (8.0e-05) | 9.8e-04 (0.0078) | † |
| line / L3 | repository only | SIE-M/SIE | 1.19 [1.08, 1.29] | 30 | 10 | 0.0013 (0.0025) | 0.0068 (0.014) | † |
| line / L4 | Fig. 3; Sect. 4.1 | Res/LF | 3.81 [3.46, 4.19] | 40 | 10 | 1.0e-05 (8.0e-05) | 9.8e-04 (0.0078) | † |
| line / L5 | Fig. 3; Sect. 4.1 | RK4-net/LF | 2.14 [1.97, 2.33] | 40 | 10 | 1.0e-05 (8.0e-05) | 9.8e-04 (0.0078) | † |
| line / L6 | Sect. 4.1 | LF/LF-Sep | 1.44 [1.31, 1.58] | 40 | 10 | 1.0e-05 (8.0e-05) | 9.8e-04 (0.0078) | † |
| line / L7 | Fig. 3; Sect. 4.1 | Res/LF | 1.96 [1.7, 2.24] | 40 | 10 | 1.0e-05 (8.0e-05) | 9.8e-04 (0.0078) | † |
| line / L8 | repository only | LF/LF-Sep | 1.16 [1.03, 1.3] | 40 | 10 | 0.0016 (0.0025) | 0.021 (0.021) | † |
| centered-update / K1 | Table VI | LF/LF-C | 1.04 [0.954, 1.15] | 40 | 10 | 0.2 (1) | 0.21 (1) |  |
| centered-update / K2 | Table VI | LF/LF-C | 0.982 [0.896, 1.08] | 40 | 10 | 0.62 (1) | 0.64 (1) |  |
| centered-update / K3 | Table VI | LF/LF-C | 1.03 [0.971, 1.08] | 40 | 10 | 0.24 (1) | 0.21 (1) |  |
| centered-update / K4 | Table VI | LF/LF-C | 1.12 [0.937, 1.33] | 40 | 10 | 0.077 (0.62) | 0.13 (1) |  |
| centered-update / K5 | Table VI | Res2/LF-C | 1 [0.937, 1.09] | 40 | 10 | 0.46 (1) | 0.46 (1) |  |
| centered-update / K6 | Table VI | Res2/LF-C | 1.03 [0.978, 1.1] | 40 | 10 | 0.23 (1) | 0.19 (1) |  |
| centered-update / K7 | Table VI | Res2/LF-C | 1.06 [0.966, 1.17] | 40 | 10 | 0.083 (0.62) | 0.13 (1) |  |
| centered-update / K8 | Table VI | Res2/LF-C | 0.984 [0.769, 1.22] | 40 | 10 | 0.57 (1) | 0.55 (1) |  |
| embedded / E1 | Fig. 3; Sect. 4.2 | Res/LF | 13.4 [10.9, 16.4] | 40 | 10 | 1.0e-05 (3.0e-05) | 9.8e-04 (0.0029) | † |
| embedded / E2 | Fig. 3; Sect. 4.2 | RK4-net/LF | 3.87 [3.28, 4.51] | 40 | 10 | 1.0e-05 (3.0e-05) | 9.8e-04 (0.0029) | † |
| embedded / E3 | Fig. 3; Sect. 4.2 | SIE-M/LF | 12.4 [10.5, 15.1] | 40 | 10 | 1.0e-05 (3.0e-05) | 9.8e-04 (0.0029) | † |
| leaky / G1 | Sect. 4.1 | LF-Sep/LF | 19.1 [15.8, 23.2] | 40 | 10 | 1.0e-05 (6.0e-05) | 9.8e-04 (0.0059) | † |
| leaky / G2 | Fig. 3; Sect. 4.1 | Res/LF | 2.66 [2.08, 3.38] | 40 | 10 | 1.0e-05 (6.0e-05) | 9.8e-04 (0.0059) | † |
| leaky / G3 | Fig. 3; Sect. 4.1 | RK4-net/LF | 1.37 [1.12, 1.69] | 40 | 10 | 0.0025 (0.0025) | 0.013 (0.013) | † |
| leaky / G4 | Fig. 3 | SIE-M/LF | 2.13 [1.71, 2.61] | 40 | 10 | 1.0e-05 (6.0e-05) | 9.8e-04 (0.0059) | † |
| leaky / G5 | repository only | Res/LF | 2.23 [1.83, 2.69] | 40 | 10 | 1.0e-05 (6.0e-05) | 9.8e-04 (0.0059) | † |
| leaky / G6 | repository only | Res/LF | 1.93 [1.67, 2.24] | 40 | 10 | 1.0e-05 (6.0e-05) | 9.8e-04 (0.0059) | † |
| leaky-structured / S1 | repository only | LF/LF-SepD | 1.14 [0.876, 1.48] | 40 | 10 | 0.22 (0.65) | 0.19 (0.56) |  |
| leaky-structured / S2 | repository only | Res/LF-SepD | 3.03 [2.31, 4.1] | 40 | 10 | 1.0e-05 (5.0e-05) | 9.8e-04 (0.0049) | † |
| leaky-structured / S3 | repository only | RK4-net/LF-SepD | 1.56 [1.2, 2.08] | 40 | 10 | 0.0025 (0.0098) | 0.0059 (0.023) | † |
| leaky-structured / S4 | repository only | LF/LF-SepD | 1.02 [0.84, 1.24] | 40 | 10 | 0.43 (0.67) | 0.42 (0.64) |  |
| leaky-structured / S5 | repository only | LF/LF-SepD | 1.04 [0.887, 1.23] | 40 | 10 | 0.34 (0.67) | 0.32 (0.64) |  |
| time-step / D1 | Sect. 4.4 | Res/LF | 0.325 [0.266, 0.393] | 40 | 10 | 1 (1) | 1 (1) |  |
| time-step / D2 | Sect. 4.4 | SIE-M/LF | 0.368 [0.29, 0.454] | 40 | 10 | 1 (1) | 1 (1) |  |
| time-step / D3 | Sect. 4.4 | Res/LF | 0.663 [0.561, 0.802] | 40 | 10 | 1 (1) | 1 (1) |  |
| time-step / D4 | Sect. 4.4 | SIE-M/LF | 0.522 [0.472, 0.579] | 40 | 10 | 1 (1) | 1 (1) |  |
| time-step / D5 | Sect. 4.4 | RK4-net/LF | 0.867 [0.749, 0.98] | 40 | 10 | 0.92 (1) | 0.97 (1) |  |
| time-step / D6 | Sect. 4.4 | RK4-net/LF | 0.924 [0.839, 1.02] | 40 | 10 | 0.97 (1) | 0.91 (1) |  |
| training / T1-M-circuit | Table VI | Res/LF | 3.44 [2.92, 4.08] | 40 | 10 | 1.0e-05 (1.2e-04) | 9.8e-04 (0.012) | † |
| training / T2-M-circuit | Table VI | RK4-net/LF | 3.67 [2.92, 4.62] | 40 | 10 | 1.0e-05 (1.2e-04) | 9.8e-04 (0.012) | † |
| training / T3-M-circuit | Fig. 3 | SIE-M/LF | 4.31 [3.61, 5.06] | 40 | 10 | 1.0e-05 (1.2e-04) | 9.8e-04 (0.012) | † |
| training / T1-M-line | Table VI | Res/LF | 3.29 [2.96, 3.62] | 40 | 10 | 1.0e-05 (1.2e-04) | 9.8e-04 (0.012) | † |
| training / T2-M-line | Table VI | RK4-net/LF | 1.95 [1.75, 2.23] | 40 | 10 | 1.0e-05 (1.2e-04) | 9.8e-04 (0.012) | † |
| training / T3-M-line | Fig. 3 | SIE-M/LF | 2.57 [2.43, 2.75] | 40 | 10 | 1.0e-05 (1.2e-04) | 9.8e-04 (0.012) | † |
| training / T1-N-circuit | Table VI | Res/LF | no stratum left | 0 | 0 | 1 (1) | 1 (1) | |
| training / T2-N-circuit | Table VI | RK4-net/LF | no stratum left | 0 | 0 | 1 (1) | 1 (1) | |
| training / T3-N-circuit | Table VI | SIE-M/LF | no stratum left | 0 | 0 | 1 (1) | 1 (1) | |
| training / T1-N-line | Table VI | Res/LF | no stratum left | 0 | 0 | 1 (1) | 1 (1) | |
| training / T2-N-line | Table VI | RK4-net/LF | 0.972 [0.88, 1.08] | 10 | 10 | 0.7 (1) | 0.7 (1) |  |
| training / T3-N-line | Table VI | SIE-M/LF | no stratum left | 0 | 0 | 1 (1) | 1 (1) | |
| vi-variables / X1 | Table VI | LF (V-I)/LF (q-phi) | 1.13 [0.944, 1.33] | 40 | 10 | 0.078 (0.078) | 0.11 (0.11) |  |
| vi-variables / X2 | Table VI | LF (V-I)/LF (q-phi) | 1.57 [1.38, 1.81] | 40 | 10 | 1.0e-05 (3.0e-05) | 9.8e-04 (0.0029) | † |
| vi-variables / X3 | Table VI | LF (V-I)/LF (q-phi) | 3.16 [2.79, 3.57] | 40 | 10 | 1.0e-05 (3.0e-05) | 9.8e-04 (0.0029) | † |
| kcl-line / K1 | Sect. 4.1 | LF/LF-KCL | 6.01 [5.65, 6.4] | 40 | 10 | 1.0e-05 (3.0e-05) | 9.8e-04 (0.0029) | † |
| kcl-line / K2 | repository only | LF/LF-KCL | 17.2 [15, 19.8] | 40 | 10 | 1.0e-05 (3.0e-05) | 9.8e-04 (0.0029) | † |
| kcl-line / K3 | Sect. 4.1 | LF/LF-KCLD | 6.86 [5.42, 8.71] | 40 | 10 | 1.0e-05 (3.0e-05) | 9.8e-04 (0.0029) | † |
| input-noise / N2-Res-circuit | Table VI | Res/LF | 1.67 [1.31, 2.12] | 40 | 10 | 5.0e-05 (1.5e-04) | 0.0029 (0.0088) | † |
| input-noise / N2-RK4-circuit | Table VI | RK4-net/LF | 1.6 [1.23, 2.09] | 40 | 10 | 2.2e-04 (4.4e-04) | 0.0078 (0.012) | † |
| input-noise / N2-SIEM-circuit | Fig. 3 | SIE-M/LF | 1.5 [1.2, 1.89] | 40 | 10 | 4.4e-04 (4.4e-04) | 0.0059 (0.012) | † |
| input-noise / N2-Res-line | Table VI | Res/LF | 3.4 [3.05, 3.81] | 40 | 10 | 1.0e-05 (6.0e-05) | 9.8e-04 (0.0059) | † |
| input-noise / N2-RK4-line | Table VI | RK4-net/LF | 1.98 [1.81, 2.18] | 40 | 10 | 1.0e-05 (6.0e-05) | 9.8e-04 (0.0059) | † |
| input-noise / N2-SIEM-line | Fig. 3 | SIE-M/LF | 3.27 [2.72, 4.18] | 40 | 10 | 1.0e-05 (6.0e-05) | 9.8e-04 (0.0059) | † |
| kcl-time-step / E1 | Sect. 4.4 | LF-KCL/LF | 4.31 [3.93, 4.74] | 40 | 10 | 1.0e-05 (2.0e-05) | 9.8e-04 (0.002) | † |
| kcl-time-step / E2 | Sect. 4.4 | LF-KCL/LF | 0.717 [0.682, 0.755] | 40 | 10 | 1 (1) | 1 (1) |  |
| rk4-kcl / F1 | repository only | LF/RK4-KCL | 31.4 [29.2, 33.5] | 40 | 10 | 1.0e-05 (1.0e-05) | 9.8e-04 (9.8e-04) | † |
| varactor / V1 | Table V | SIE/LF | 1.49 [1.23, 1.79] | 20 | 10 | 1.7e-04 (6.8e-04) | 0.0029 (0.015) | † |
| varactor / V2 | Table V | Res/LF | 2.26 [1.87, 2.64] | 20 | 10 | 2.0e-05 (1.2e-04) | 9.8e-04 (0.0059) | † |
| varactor / V3 | Table V | RK4-net/LF | 2.22 [1.54, 3.09] | 20 | 10 | 1.3e-04 (6.5e-04) | 0.0039 (0.016) | † |
| varactor / V4 | Table V | SIE/LF | 1.13 [0.882, 1.45] | 20 | 10 | 0.23 (0.23) | 0.19 (0.19) |  |
| varactor / V5 | Table V | Res/LF | 1.59 [1.2, 2.07] | 20 | 10 | 0.0029 (0.0086) | 0.0098 (0.029) | † |
| varactor / V6 | Table V | RK4-net/LF | 1.42 [1.06, 1.9] | 20 | 10 | 0.028 (0.057) | 0.027 (0.055) |  |
| varactor-reflected / R1 | Table V | SIE/LF | 1.05 [0.87, 1.31] | 20 | 10 | 0.35 (1) | 0.33 (1) |  |
| varactor-reflected / R2 | Table V | LF-U/LF | 1 [0.797, 1.27] | 20 | 10 | 0.49 (1) | 0.5 (1) |  |
| varactor-reflected / R3 | Table V | Res/LF | 1.57 [1.18, 2.01] | 20 | 10 | 0.0059 (0.029) | 0.0098 (0.049) | † |
| varactor-reflected / R4 | Table V | RK4-net/LF | 1.72 [1.36, 2.14] | 20 | 10 | 0.0011 (0.0068) | 0.002 (0.012) | † |
| varactor-reflected / R5 | repository only | SIE/LF | 0.908 [0.794, 1.03] | 20 | 10 | 0.84 (1) | 0.9 (1) |  |
| varactor-reflected / R6 | repository only | LF-U/LF | 0.908 [0.808, 1.02] | 20 | 10 | 0.85 (1) | 0.92 (1) |  |
| varactor-drives / P1 | Table V | SIE/LF | 1.32 [1.15, 1.54] | 20 | 10 | 0.0047 (0.028) | 0.0039 (0.023) | † |
| varactor-drives / P2 | Table V | LF-U/LF | 1.33 [0.968, 1.8] | 20 | 10 | 0.053 (0.16) | 0.059 (0.13) |  |
| varactor-drives / P3 | repository only | SIE/LF | 1.21 [0.994, 1.43] | 20 | 10 | 0.078 (0.16) | 0.044 (0.13) |  |
| varactor-drives / P4 | repository only | LF-U/LF | 1.29 [1.09, 1.49] | 20 | 10 | 0.021 (0.1) | 0.011 (0.054) |  |
| varactor-drives / P5 | repository only | SIE/LF | 1.17 [0.97, 1.41] | 20 | 10 | 0.12 (0.16) | 0.07 (0.13) |  |
| varactor-drives / P6 | repository only | LF-U/LF | 1.27 [1.09, 1.48] | 20 | 10 | 0.026 (0.1) | 0.012 (0.054) |  |
| varactor-rest-corrected / Q1 | Sect. 4.3 | SIE/LF | 0.965 [0.815, 1.13] | 20 | 10 | 0.64 (1) | 0.65 (1) |  |
| varactor-rest-corrected / Q2 | Sect. 4.3 | LF-U/LF | 1 [0.859, 1.15] | 20 | 10 | 0.49 (1) | 0.48 (1) |  |
| varactor-rest-corrected / Q3 | Sect. 4.3 | SIE/LF | 1.02 [0.859, 1.2] | 20 | 10 | 0.43 (1) | 0.42 (1) |  |
| varactor-rest-corrected / Q4 | Sect. 4.3 | LF-U/LF | 1.01 [0.846, 1.21] | 20 | 10 | 0.45 (1) | 0.45 (1) |  |
| varactor-rest-corrected / Q5 | Sect. 4.3 | SIE/LF | 1.02 [0.867, 1.18] | 20 | 10 | 0.44 (1) | 0.42 (1) |  |
| varactor-rest-corrected / Q6 | Sect. 4.3 | LF-U/LF | 1.03 [0.875, 1.2] | 20 | 10 | 0.4 (1) | 0.37 (1) |  |
| varactor-sinusoids-2 / S1 | Table V | SIE/LF | 1.15 [0.974, 1.39] | 20 | 10 | 0.074 (0.074) | 0.092 (0.092) |  |
| varactor-sinusoids-2 / S2 | Table V | LF-U/LF | 1.48 [1.35, 1.67] | 20 | 10 | 1.0e-05 (2.0e-05) | 9.8e-04 (0.002) | † |
| varactor-energy-2 / E1 | Table V | LF-U/LF | 1.6 [1.32, 2] | 20 | 10 | 2.7e-04 (5.4e-04) | 9.8e-04 (0.002) | † |
| varactor-energy-2 / E2 | Table V | SIE/LF | 1.37 [1.09, 1.76] | 20 | 10 | 0.0044 (0.0044) | 0.016 (0.016) | † |
| varactor-components / W1 | repository only | LF-U/LF | 1.34 [1.19, 1.55] | 20 | 10 | 1.1e-04 (6.6e-04) | 0.002 (0.012) | † |
| varactor-components / W2 | repository only | SIE/LF | 1.1 [0.916, 1.37] | 20 | 10 | 0.16 (0.48) | 0.22 (0.67) |  |
| varactor-components / W3 | repository only | LF-U/LF | 1.96 [1.44, 2.75] | 20 | 10 | 7.4e-04 (0.0037) | 0.002 (0.012) | † |
| varactor-components / W4 | repository only | SIE/LF | 1.48 [1.02, 2.12] | 20 | 10 | 0.016 (0.065) | 0.04 (0.16) |  |
| varactor-components / W5 | repository only | LF-U/LF | 1.05 [0.744, 1.53] | 20 | 10 | 0.41 (0.82) | 0.41 (0.82) |  |
| varactor-components / W6 | repository only | SIE/LF | 0.983 [0.56, 1.71] | 20 | 10 | 0.53 (0.82) | 0.52 (0.82) |  |
| second-set / R1 | Fig. 3; Sect. 4.1 | SIE/LF | 2.41 [1.93, 3.17] | 40 | 10 | 1.0e-05 (4.0e-05) | 9.8e-04 (0.0039) | † |
| second-set / R2 | Fig. 3; Sect. 4.1 | LF-U/LF | 7.84 [6.53, 9.32] | 40 | 10 | 1.0e-05 (4.0e-05) | 9.8e-04 (0.0039) | † |
| second-set / R3 | Fig. 3; Sect. 4.1 | SIE/LF | 2.83 [2.59, 3.08] | 40 | 10 | 1.0e-05 (4.0e-05) | 9.8e-04 (0.0039) | † |
| second-set / R4 | Fig. 3; Sect. 4.1 | LF-U/LF | 3.9 [3.53, 4.3] | 40 | 10 | 1.0e-05 (4.0e-05) | 9.8e-04 (0.0039) | † |
| varactor-energy-3 / F1 | Table V | SIE/LF | 1.44 [1.24, 1.66] | 60 | 30 | 2.0e-05 (2.0e-05) | 4.0e-05 (4.0e-05) | † |

## Table IV

Medians over the runs of each model (run counts in parentheses).
| Model | Circuit | N=8 | N=32 | Div. | Energy |
|---|---|---|---|---|---|
| LF | 0.0129 (40) | 0.0622 (40) | 0.141 (40) | 1 (40) | 0.0385 (40) |
| Res | 0.052 (40) | 0.29 (40) | 0.288 (40) | 122 (40) | 0.523 (40) |
| RK4-net | 0.0466 (40) | 0.147 (40) | 0.194 (40) | 48.5 (40) | 0.148 (40) |
| LF-U | 0.0792 (40) | 0.245 (40) | 0.253 (40) | 228 (40) | 0.358 (40) |
| SIE | 0.0275 (40) | 0.218 (40) | 0.215 (40) | 19.5 (40) | 0.273 (40) |
| SIE-M | 0.0467 (40) | 0.197 (40) | 0.24 (40) | 173 (40) | 0.434 (40) |
| LF-KCL | 0.0063 (40) | 0.0115 (40) | 0.0571 (40) | 0 (40) | 0.00244 (40) |
| Classical | 0.00751 | 0.0212 | 0.0548 | 0 | 0.00123 |

## Table V

Ratios to LF. †: significant (both Holm-adjusted p < 0.05); *: planned test, not significant; plain: not tested.
| Row | Load Res | Load RK4-net | Load SIE | Load LF-U | Energy SIE | Energy LF-U |
|---|---|---|---|---|---|---|
| Training ring | 2.26† | 2.22† | 1.49† | 1.59 | -- | -- |
| Pulses, matched | 1.59† | 1.42* | 1.13* | 1.04 | 1.23 | 1.38 |
| Pulses, reflected | 1.57† | 1.72† | 1.05* | 1* | 1.25 | 1.51 |
| Pulse trains | 1.62 | 1.45 | 1.13 | 1.05 | 1.25 | 1.35 |
| Bit streams | 1.48 | 1.33 | 1.09 | 1.04 | 1.12 | 1.27 |
| Sinusoids | 1.99 | 2.09 | 1.32† | 1.33* | 1.38 | 1.37 |
| Pulses, 2nd set | -- | -- | 1.1 | 1.22 | 1.37† | 1.6† |
| Pulses, 3rd set | -- | -- | 1.14 | -- | 1.44† | -- |
| Sinusoids, 2nd set | -- | -- | 1.15* | 1.48† | 1.23 | 1.43 |

## Fig. 3

| Group | Row | A/B | Ratio | 95% seed interval | Dagger | Test |
|---|---|---|---|---|---|---|
| one-network baselines | Cubic circuit | Res/LF | 4.43 | [3.85, 5.04] | † | circuit / C4 |
| one-network baselines | Cubic circuit | RK4-net/LF | 3.75 | [3.02, 4.69] | † | circuit / C5 |
| one-network baselines | Line (N=8) | Res/LF | 3.81 | [3.46, 4.19] | † | line / L4 |
| one-network baselines | Line (N=8) | RK4-net/LF | 2.14 | [1.97, 2.33] | † | line / L5 |
| one-network baselines | Line (N=32) | Res/LF | 1.96 | [1.7, 2.24] | † | line / L7 |
| one-network baselines | Leaky ring | Res/LF | 2.66 | [2.08, 3.38] | † | leaky / G2 |
| one-network baselines | Leaky ring | RK4-net/LF | 1.37 | [1.12, 1.69] | † | leaky / G3 |
| one-network baselines | Embedded ring, energy | Res/LF | 13.4 | [10.9, 16.4] | † | embedded / E1 |
| one-network baselines | Embedded ring, energy | RK4-net/LF | 3.87 | [3.28, 4.51] | † | embedded / E2 |
| off-center ablations | Cubic circuit | SIE-M/LF | 3.76 | [3.16, 4.46] | † | circuit / C1 |
| off-center ablations | Cubic circuit, 2nd set | SIE/LF | 2.41 | [1.93, 3.17] | † | second-set / R1 |
| off-center ablations | Cubic circuit, 2nd set | LF-U/LF | 7.84 | [6.53, 9.32] | † | second-set / R2 |
| off-center ablations | Circuit, rollout loss | SIE-M/LF | 4.31 | [3.61, 5.06] | † | training / T3-M-circuit |
| off-center ablations | Circuit, 0.1% noise | SIE-M/LF | 1.5 | [1.2, 1.89] | † | input-noise / N2-SIEM-circuit |
| off-center ablations | Line (N=8) | SIE-M/LF | 3.44 | [3.06, 3.9] | † | line / L1 |
| off-center ablations | Line (N=8), 2nd set | SIE/LF | 2.83 | [2.59, 3.08] | † | second-set / R3 |
| off-center ablations | Line (N=8), 2nd set | LF-U/LF | 3.9 | [3.53, 4.3] | † | second-set / R4 |
| off-center ablations | Line, rollout loss | SIE-M/LF | 2.57 | [2.43, 2.75] | † | training / T3-M-line |
| off-center ablations | Line, 0.1% noise | SIE-M/LF | 3.27 | [2.72, 4.18] | † | input-noise / N2-SIEM-line |
| off-center ablations | Leaky ring | SIE-M/LF | 2.13 | [1.71, 2.61] | † | leaky / G4 |
| off-center ablations | Embedded ring, energy | SIE-M/LF | 12.4 | [10.5, 15.1] | † | embedded / E3 |
