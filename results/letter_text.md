# Numbers in the letter's text and README.md

Computed by `python scripts/letter_text.py` from results/letter_runs.csv, results/text_runs.csv, results/text_values.json, results/fig5_runs.csv, and docs/planned_tests.csv. A/B [95% seed interval] (pairs) as in scripts/letter_stats.py.

## Sect. 3, Tables II and III

- Time step 0.05 'about 8% of the circuit's stability limit linearized at the highest energy': limit 0.6258, 0.05 / limit = 0.0799; 'Courant number 0.05 for small signals to 0.11': 0.05 and 0.1121 (line limit 0.4459)
- Classical scheme, true laws, initial conditions of 512 diverged (t = 50, t = 500): nltl32_v1_dt0p2 123, 205; nltl32_v1_dt0p1 0, 0; nltl8_v1_dt0p2 0, 38
- Learning rate chosen per model and size by the validation median (runs >= 8): LF small 3e2; LF medium 3e2; Res small 1e2; Res medium 3e2; RK4-net small 1e2; RK4-net medium 1e2; SIE-M small 3e3; SIE-M medium 3e2
  - Res/LF with the chosen rates 5.5 [4.16, 7.16] (20 pairs); with 1e-2 and 3e-2 4.43 [3.86, 5.04] (40 pairs)
  - RK4-net/LF with the chosen rates 4.48 [3.67, 5.53] (20 pairs); with 1e-2 and 3e-2 3.75 [3.02, 4.7] (40 pairs)
  - SIE-M/LF with the chosen rates 4.49 [3.67, 5.51] (20 pairs); with 1e-2 and 3e-2 3.76 [3.18, 4.49] (40 pairs)
- Free runs at dt = 0.05 with any diverged test initial condition (17 sources, 5060 runs): none
- Planned tests: 128 in 28 families of 1-12; repository only 52

- Varactor line: C_b = 0.5092 pF, t0 = 25.43 ps, Z0 = 49.94 Ohm, dt = 1.271 ps; u(q) = q + alpha q^2 with alpha = 1/(4 (phi + V_bias)) = 0.0925926 (0.0925 in the code); dt at 3 V: 0.0603 of the stability limit
- Matched pulses: 1.004-2.981 V, edges 108.3-302.3 ps, 2.543 ns; median 10-90% rise 122.2 ps incident, 72.47 ps at the all-classical load
- Reflecting terminations: 4.994 and 499.4 Ohm, 0.5183-0.9948 V, 10.01 ns; first arrival 38.85 t0, arrivals 77.12 t0 apart: 10.2 block crossings
- Pulse trains: 1.032-2.984 V; period_ns 1.529-3.031; 5.086 ns
- Bit streams: 0.5093-1.498 V; bit_rate_Gbps 0.9856-2.318; 5.086 ns
- Sinusoids: 0.5199-1.494 V; frequency_GHz 1.195-4.693; 5.086 ns
- Pulse trains, 2nd set: 1-2.972 V; period_ns 1.526-3.047; 5.086 ns
- Sinusoids, 2nd set: 0.5088-1.483 V; frequency_GHz 1.188-4.681; 5.086 ns
- Signals left out of the block-energy deviation, with no block energy in the all-classical line: open 0, reflect 0, train 0, prbs 2, cw 0

## Sect. 4.1

- 32 sections: RK4-net/LF 1.38 [1.21, 1.55] (40 pairs); LF/LF-KCL 2.68 [2.49, 2.93] (40 pairs)
- Res2/Res: circuit 1.68 [1.59, 1.79] (40 pairs), line 1.42 [1.29, 1.55] (20 pairs)
- Planned off-center ablation / LF comparisons in charge and flux, free runs and the embedded ring: 15, 13 with a stratum left, 13 with a dagger, 8 against SIE-M: circuit/C1 SIE-M/LF 3.76†, line/L1 SIE-M/LF 3.44†, embedded/E3 SIE-M/LF 12.4†, leaky/G4 SIE-M/LF 2.13†, training/T3-M-circuit SIE-M/LF 4.31†, training/T3-M-line SIE-M/LF 2.57†, input-noise/N2-SIEM-circuit SIE-M/LF 1.5†, input-noise/N2-SIEM-line SIE-M/LF 3.27†, varactor/V1 SIE/LF 1.49†, second-set/R1 SIE/LF 2.41†, second-set/R2 LF-U/LF 7.84†, second-set/R3 SIE/LF 2.83†, second-set/R4 LF-U/LF 3.9†
- First set: circuit SIE/LF 2.31 [2.12, 2.5] (40 pairs); circuit LF-U/LF 6.32 [5.52, 7.45] (40 pairs); line SIE/LF 3.01 [2.73, 3.36] (40 pairs); line LF-U/LF 3.52 [3.24, 3.86] (40 pairs)
- Second set: circuit SIE/LF 2.41 [1.93, 3.18] (40 pairs) (small 2.83, medium 2.05); circuit LF-U/LF 7.84 [6.5, 9.32] (40 pairs) (small 8.3, medium 7.41); line SIE/LF 2.83 [2.59, 3.08] (40 pairs) (small 3.25, medium 2.46); line LF-U/LF 3.9 [3.52, 4.3] (40 pairs) (small 3.57, medium 4.27)
- X/LF at about 700 and 2400 parameters: narrowed in 13 of 15; smallest at 2400 1.301; circuit Res 4.93 -> 3.97; circuit RK4-net 4.22 -> 3.32; circuit SIE 2.39 -> 2.23; circuit LF-U 8.16 -> 4.89; circuit SIE-M 4.75 -> 2.98; line N=8 Res 4.5 -> 3.23; line N=8 RK4-net 2.29 -> 2; line N=8 SIE 3.95 -> 2.29; line N=8 LF-U 3.32 -> 3.74; line N=8 SIE-M 3.93 -> 3.02; line N=32 Res 2.41 -> 1.6; line N=32 RK4-net 1.46 -> 1.3; line N=32 SIE 2.76 -> 1.42; line N=32 LF-U 1.84 -> 1.9; line N=32 SIE-M 2.56 -> 1.82
- Line N=8 medians at 2400 parameters: LF 0.0563, Res 0.177, RK4-net 0.112, LF-U 0.217, SIE 0.121, SIE-M 0.164, LF-KCL 0.0114; at 700: LF 0.0711, Res 0.366, RK4-net 0.194, LF-U 0.254, SIE 0.295, SIE-M 0.334, LF-KCL 0.0116
- LF-Sep/LF in the line (1 / line L6): 0.695

## Sect. 4.2 (and README.md)

- Pairs in which LF lost fewer initial conditions to divergence: Res 37 of 40, RK4-net 35 of 40, SIE 32 of 40, LF-U 40 of 40, SIE-M 40 of 40
- Total charge and flux changed by t = 500 (medians, % of the initial norms): LF 23.52 and 20.77; Res 76.4 and 63.3; RK4-net 44.16 and 44.04; LF-KCL 1.272 and 0.9261; SIE-M 25.59 and 69.68; LF-Sep 17.34 and 14.11
- Energy change with divergence counted as the worst (median over initial conditions), X/LF: Res 18.5 [14.2, 24.1] (40 pairs); RK4-net 4.29 [3.43, 5.46] (40 pairs); SIE 14.4 [9.66, 22.9] (40 pairs); SIE-M 33.3 [21.7, 53.9] (40 pairs); LF-U 58.7 [27.9, 158] (40 pairs)

## Sect. 4.3 (and README.md)

- LF's median load-voltage error (% of the amplitude): pulses 0.4166, reflected 2.364, trains 0.3863, bits 0.8266, sinusoids 1.413, pulses-2 0.4533, reflected-2 2.868, trains-2 0.4495, pulses-3 0.5913, reflected-3 2.931, trains-3 0.579, sinusoids-2 1.703; range 0.386-2.93, without the 3rd set 0.386-2.87, in the rows of Table V 0.386-2.36
- Runs with diverged signals in the 109 files of the driven line: busy_low_Res.json Res 3e2 seed 0: 1 of 64; cw_new_SIE.json SIE 3e2 seed 18: 8 of 64; cw_rc_RK4.json RK4 3e2 seed 8: 3 of 64; cw_rc_SIE.json SIE 3e2 seed 5: 3 of 64
- Block energy, first set (README.md): pulses Res/LF 1.773; pulses RK4-net/LF 1.684; reflected Res/LF 1.987; reflected RK4-net/LF 1.952; trains Res/LF 1.854; trains RK4-net/LF 1.79; bits Res/LF 1.461; bits RK4-net/LF 1.464; sinusoids Res/LF 2.384; sinusoids RK4-net/LF 2.647; range 1.46-2.65
- LF-KCL/LF: ring 0.268; pulses 0.128; reflected 0.0618; trains 0.149; bits 0.0616; sinusoids 0.211; pulses-energy 0.106; reflected-energy 0.0222; trains-energy 0.125; bits-energy 0.028; sinusoids-energy 0.0512; load voltage and the ring 0.0616-0.268, block energy 0.0222-0.125
- Uniform-state increments subtracted (README.md; each drive as in README.md; load voltage, then block energy):
  - Pulses, matched: load SIE/LF 1.02 [0.866, 1.18] (20 pairs); load LF-U/LF 1.03 [0.877, 1.2] (20 pairs); load Res/LF 1.49 [1.2, 1.87] (20 pairs); load RK4-net/LF 1.83 [1.36, 2.4] (20 pairs); energy SIE/LF 0.95 [0.789, 1.13] (20 pairs); energy LF-U/LF 1 [0.872, 1.16] (20 pairs); energy Res/LF 1.4 [1.11, 1.79] (20 pairs); energy RK4-net/LF 1.59 [1.22, 2.04] (20 pairs)
  - Pulses, reflected: load SIE/LF 0.952 [0.816, 1.12] (20 pairs); load LF-U/LF 0.969 [0.796, 1.19] (20 pairs); energy SIE/LF 0.904 [0.759, 1.09] (20 pairs); energy LF-U/LF 1.05 [0.9, 1.22] (20 pairs)
  - Pulse trains: load SIE/LF 1.02 [0.86, 1.2] (20 pairs); load LF-U/LF 1.01 [0.839, 1.21] (20 pairs); load Res/LF 1.48 [1.22, 1.8] (20 pairs); load RK4-net/LF 1.88 [1.37, 2.5] (20 pairs); energy SIE/LF 0.977 [0.813, 1.15] (20 pairs); energy LF-U/LF 1.02 [0.909, 1.14] (20 pairs); energy Res/LF 1.43 [1.18, 1.75] (20 pairs); energy RK4-net/LF 1.67 [1.24, 2.21] (20 pairs)
  - Bit streams: load SIE/LF 0.965 [0.81, 1.13] (20 pairs); load LF-U/LF 1 [0.859, 1.15] (20 pairs); load Res/LF 1.46 [1.08, 1.97] (20 pairs); load RK4-net/LF 1.84 [1.38, 2.46] (20 pairs); energy SIE/LF 0.873 [0.667, 1.11] (20 pairs); energy LF-U/LF 0.991 [0.833, 1.16] (20 pairs); energy Res/LF 1.25 [0.842, 1.92] (20 pairs); energy RK4-net/LF 1.56 [1.15, 2.16] (20 pairs)
  - Sinusoids: load SIE/LF 1.11 [0.924, 1.31] (20 pairs); load LF-U/LF 1.23 [0.822, 1.79] (20 pairs); load Res/LF 1.75 [1.29, 2.4] (20 pairs); load RK4-net/LF 1.92 [1.48, 2.53] (20 pairs); energy SIE/LF 1.31 [0.981, 1.8] (20 pairs); energy LF-U/LF 1.16 [0.807, 1.65] (20 pairs); energy Res/LF 1.7 [1.33, 2.09] (20 pairs); energy RK4-net/LF 2.21 [1.84, 2.72] (20 pairs)
  - Pulses, 2nd set: load SIE/LF 0.976 [0.87, 1.08] (20 pairs); load LF-U/LF 1.1 [0.974, 1.23] (20 pairs); energy SIE/LF 0.957 [0.813, 1.11] (20 pairs); energy LF-U/LF 1.06 [0.891, 1.26] (20 pairs)
  - Pulses, 3rd set: load SIE/LF 1.09 [0.964, 1.24] (60 pairs); energy SIE/LF 1.06 [0.936, 1.19] (60 pairs)
  - Sinusoids, 2nd set: load SIE/LF 1.27 [1.05, 1.66] (20 pairs); load LF-U/LF 1.45 [1.27, 1.66] (20 pairs); energy SIE/LF 1.2 [0.93, 1.72] (20 pairs); energy LF-U/LF 1.32 [1.1, 1.59] (20 pairs)
- Arrival-time drift per block crossing (40 networks per model): SIE/LF 0.761 [0.595, 0.962] (40 pairs); LF-U/LF 0.712 [0.555, 0.905] (40 pairs)
- Measured / predicted dc load voltage, matched line: median 0.998 over 140 networks (10-90%: 0.823-1.2)
- Busy signals, load voltage: SIE/LF 1.33 [1.09, 1.65] (20 pairs); LF-U/LF 1.39 [1.03, 1.87] (20 pairs)
- Busy signals, low frequency, load voltage: SIE/LF 1.17 [1.02, 1.36] (20 pairs); LF-U/LF 1.22 [0.923, 1.61] (20 pairs)
- Bursts, load voltage: SIE/LF 1.13 [0.949, 1.38] (20 pairs); LF-U/LF 1.22 [0.892, 1.67] (20 pairs)
- Training ring, low third: SIE/LF 1.45 [1.19, 1.78] (20 pairs); LF-U/LF 1.63 [1.21, 2.24] (20 pairs)
- Training ring, mid third: SIE/LF 1.5 [1.25, 1.82] (20 pairs); LF-U/LF 1.54 [1.22, 1.99] (20 pairs)
- Training ring, high third: SIE/LF 1.5 [1.22, 1.84] (20 pairs); LF-U/LF 1.58 [1.23, 2.06] (20 pairs)
- Pulses, other measures; charge in the block: SIE/LF 1.18 [0.852, 1.63] (20 pairs), LF-U/LF 1.12 [0.664, 1.88] (20 pairs); flux in the block: SIE/LF 1.17 [0.926, 1.52] (20 pairs), LF-U/LF 1.08 [0.742, 1.55] (20 pairs); charge on all sections: SIE/LF 1.18 [0.864, 1.61] (20 pairs), LF-U/LF 1.11 [0.672, 1.85] (20 pairs); block energy: SIE/LF 1.23 [0.975, 1.54] (20 pairs), LF-U/LF 1.38 [1.06, 1.8] (20 pairs)
- Reflected, other measures; charge in the block: SIE/LF 1.03 [0.861, 1.28] (20 pairs), LF-U/LF 1.03 [0.808, 1.34] (20 pairs); flux in the block: SIE/LF 1.07 [0.874, 1.38] (20 pairs), LF-U/LF 1.15 [0.798, 1.67] (20 pairs); charge on all sections: SIE/LF 1.04 [0.865, 1.28] (20 pairs), LF-U/LF 1.03 [0.806, 1.32] (20 pairs); block energy: SIE/LF 1.25 [0.974, 1.69] (20 pairs), LF-U/LF 1.51 [1.08, 2.12] (20 pairs)

## Sect. 4.4, Table VI, Fig. 5, and Sect. 5

- results/fig5_runs.csv and the runs files agree for LF (largest difference 0.0e+00)
- Circuit, dt = 0.1: LF-KCL/LF 1.5 [1.41, 1.59] (40 pairs), ratio of medians 1.538; classical / LF's median 2.186
- Circuit, dt = 0.2: LF-KCL/LF 2.06 [1.68, 2.51] (40 pairs), ratio of medians 2.204; classical / LF's median 3.487
- Circuit, dt = 0.4: LF-KCL/LF 3.83 [3.29, 4.37] (40 pairs), ratio of medians 4.421; classical / LF's median 7.377
- Line, dt = 0.2: LF median 0.1818, classical 0.244; LF-KCL/LF 0.717 [0.681, 0.756] (40 pairs)
- dt = 0.4 in the circuit: 0.639 of the linearized limit; dt = 0.2 in the line: 0.449
- Circuit, dt = 0.05 -> 0.4: Res/LF 4.43 -> 1.44 (fine / coarse 3.07); LF-U/LF 6.32 -> 1.65 (fine / coarse 3.82); SIE-M/LF 3.76 -> 1.38 (fine / coarse 2.72); SIE/LF 2.31 -> 3.74 (fine / coarse 0.616); medians at 0.4: LF 0.066, Res 0.109, RK4-net 0.067, LF-U 0.121, SIE 0.287, SIE-M 0.102
- Line, dt = 0.05 -> 0.2: Res/LF 3.81 -> 2.53 (fine / coarse 1.51); LF-U/LF 3.52 -> 2.05 (fine / coarse 1.72); SIE-M/LF 3.44 -> 1.8 (fine / coarse 1.92); SIE/LF 3.01 -> 4.76 (fine / coarse 0.632); medians at 0.2: LF 0.182, Res 0.358, RK4-net 0.172, LF-U 0.35, SIE 0.498, SIE-M 0.28
- 32 sections, dt = 0.2: RK4-net/LF 0.72 [0.586, 0.893] (40 pairs)
- RK4-net with about 700 parameters / LF with about 2400 (matched by seed and learning rate): circuit-dt0.4 1.13 [0.933, 1.35] (20 pairs); line-dt0.2 1.37 [1.23, 1.5] (20 pairs); line-32-dt0.2 1.18 [0.992, 1.42] (20 pairs); circuit 3.53 [2.82, 4.36] (20 pairs); line 3.28 [2.87, 3.69] (20 pairs); line-32 1.94 [1.74, 2.18] (20 pairs)
- circuit-noise1 medians: LF 0.4314, LF-KCL 0.02027, RK4-net 0.382, Res 0.3954, SIE 0.4131, SIE-M 0.4501
- line-noise1 medians: LF 0.3192, LF-Sep 0.1158, RK4-net 0.3158, Res 0.3353, SIE 0.3205, SIE-M 0.336
- circuit-extrap medians: LF 0.5511, LF-C 0.5555, LF-KCL 0.4146, LF-U 0.61, RK4-net 0.7118, Res 0.6228, Res2 0.5715, SIE 0.4988, SIE-M 0.4283
- line-extrap medians: LF 0.3014, LF-KCL 0.08616, LF-Sep 0.2629, LF-U 0.637, RK4-net 0.4162, Res 0.555, SIE 0.4416, SIE-M 0.5519
- Classical, extrapolation: circuit 0.0205, line 0.1019
- V-I / charge-flux training, voltage error: cubic LF 3.16 [2.78, 3.58] (40 pairs), SIE 1.45 [1.22, 1.77] (40 pairs); ind LF 1.57 [1.37, 1.81] (40 pairs), SIE 0.251 [0.223, 0.284] (40 pairs); cap LF 1.13 [0.947, 1.33] (40 pairs), SIE 0.664 [0.585, 0.758] (40 pairs); both LF 1.86 [1.69, 2.03] (40 pairs), SIE 0.618 [0.461, 0.828] (40 pairs)
- SIE (V-I) / LF (V-I): cubic ratio of medians 1.022, matched 1.053; ind ratio of medians 0.9593, matched 1.024; cap ratio of medians 1.001, matched 1.018; both ratio of medians 1.029, matched 0.9326
- Classical V-I update, cubic circuit: 0.6325 (vi_voltage.json), 0.6325, and 0.01074 with iteration (classical_centered_20261022.py)
- Off-center ablations by time step (Sect. 5): circuit 0.05: LF-U/LF 6.32, SIE/LF 2.31; circuit 0.1: LF-U/LF 7.04, SIE/LF 3.56; circuit 0.2: LF-U/LF 2.92, SIE/LF 1.95; circuit 0.4: LF-U/LF 1.65, SIE/LF 3.74; line 0.05: LF-U/LF 3.52, SIE/LF 3.01; line 0.1: LF-U/LF 2.9, SIE/LF 2.82; line 0.2: LF-U/LF 2.05, SIE/LF 4.76; line 0.4: LF failed

## README.md

- RK4-KCL/LF-KCL in the line: 0.192 [0.184, 0.199] (40 pairs)
- Holm over all 122 planned tests with p-values (6 without): of the 77 with a dagger, 68 keep p < 0.05 over runs and 8 over seed means; losing over runs: line/L3, line/L8, leaky/G3, leaky-structured/S3, varactor/V5, varactor-reflected/R3, varactor-reflected/R4, varactor-drives/P1, varactor-energy-2/E2
- --leave-out any: H6a 1.23 [0.732, 2.19] (10 pairs), Holm p 0.51 and 0.51; no stratum left in leaky/G1, training/T2-N-line; daggers lost: tan-circuit/H6a, leaky/G1; E1 8.66 (both: 13.4)

## Sources of results/text_runs.csv

Further models of circuit-noise1, line-noise1, ring, pulses, reflected, trains, bits, sinusoids, pulses-energy, reflected-energy, trains-energy, bits-energy, sinusoids-energy, pulses-rc, trains-rc, bits-rc, voltage-cap, voltage-ind, voltage-cubic (scripts/letter_data.py), and:

- line-32-dt0.2: free run, nltl8_v1_dt0p2 networks on nltl32_v1_dt0p2: test error up to t = 50
- circuit-extrap: free run, nlqf3_v1_dtd2: extrapolation error up to t = 50
- line-extrap: free run, nltl8_v1_dtd2: extrapolation error up to t = 50
- circuit-lr: free run, nlqf3_v1_dtd2: test error up to t = 50, learning rates 3e-3 to 1e-1
- circuit-lr-val: free run, nlqf3_v1_dtd2: validation error up to t = 50, learning rates 3e-3 to 1e-1
- reflected-rc: driven varactor line, drives/reflect_rc_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- sinusoids-rc: driven varactor line, drives/cw_rc_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- pulses-rc-2: driven varactor line, drives/open_new_rc_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- reflected-rc-2: driven varactor line, drives/reflect_new_rc_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- trains-rc-2: driven varactor line, drives/train_new_rc_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- sinusoids-rc-2: driven varactor line, drives/cw_new_rc_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- pulses-rc-3: driven varactor line, drives/open_third_rc_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- reflected-rc-3: driven varactor line, drives/reflect_third_rc_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- trains-rc-3: driven varactor line, drives/train_third_rc_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- pulses-rc-energy: driven varactor line, drives/open_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- reflected-rc-energy: driven varactor line, drives/reflect_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- trains-rc-energy: driven varactor line, drives/train_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- bits-rc-energy: driven varactor line, drives/prbs_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- sinusoids-rc-energy: driven varactor line, drives/cw_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- pulses-rc-energy-2: driven varactor line, drives/open_new_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- reflected-rc-energy-2: driven varactor line, drives/reflect_new_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- trains-rc-energy-2: driven varactor line, drives/train_new_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- sinusoids-rc-energy-2: driven varactor line, drives/cw_new_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- pulses-rc-energy-3: driven varactor line, drives/open_third_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- reflected-rc-energy-3: driven varactor line, drives/reflect_third_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- trains-rc-energy-3: driven varactor line, drives/train_third_rc_<model>.json: block-energy deviation / classical maximum, mean over the signals
- busy: driven varactor line, drives/busy_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- busy-low: driven varactor line, drives/busy_low_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- burst: driven varactor line, drives/burst_<model>.json: RMS load-voltage error / amplitude, mean over the signals
- pulses-charge-block: driven varactor line, drives/open_<model>.json: RMS charge error in the block / amplitude, mean over the signals
- pulses-flux-block: driven varactor line, drives/open_<model>.json: RMS flux error in the block / amplitude, mean over the signals
- pulses-charge-all: driven varactor line, drives/open_<model>.json: RMS charge error on all 40 sections / amplitude, mean over the signals
- reflected-charge-block: driven varactor line, drives/reflect_<model>.json: RMS charge error in the block / amplitude, mean over the signals
- reflected-flux-block: driven varactor line, drives/reflect_<model>.json: RMS flux error in the block / amplitude, mean over the signals
- reflected-charge-all: driven varactor line, drives/reflect_<model>.json: RMS charge error on all 40 sections / amplitude, mean over the signals
- voltage-both: voltage error, circuit with C(V) and L(I) nonlinear, V-I (nllc_e1) and q-phi (nllcqf) networks (vi_voltage.json)
- embedded-worst: embedded ring, median over the initial conditions of the energy change over the last 10^3 of 10^4 steps with divergence counted as infinite (embedded/*.json, T10000 drift_med)
- embedded-charge: embedded ring, |total charge - initial| / initial ||q||_2 at t = 500, mean over the non-diverged initial conditions (embedded_charge/*.json and embedded/*.json, T10000 dq)
- embedded-flux: the same for the total flux (T10000 dphi)
- ring-third-low: training ring of the varactor networks, charge error over the third of the test initial conditions with the smallest sum of squared differences between adjacent sections (ring_thirds.json)
- ring-third-mid: the same, middle third
- ring-third-high: the same, largest third
- dc-offset: varactor line, matched, sources off: measured dc load voltage / 320 (dq + dphi) from the block's increments at the bias point (pulse_physics/*.json)
- arrival-drift: varactor line with reflections: |mean over the pulses of the least-squares slope of the arrival shift over block crossings 1, 3, 5, 7, 9| in ns (pulse_physics/*.json)
