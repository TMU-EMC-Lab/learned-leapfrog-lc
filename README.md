# Code for "Leapfrog scheduling as an inductive bias for learning updates in nonlinear LC circuits"

This repository contains the code used in the experiments of the letter

> T. Sakai, R. Oshio, and Y. Suzuki, "Leapfrog scheduling as an inductive bias for learning updates in nonlinear LC circuits," submitted to *IEICE Electronics Express*, 2026.

The authors are with the Graduate School of Systems Design, Tokyo Metropolitan University.

**Version 1.0.0** is the release that the letter cites. It is archived on Zenodo with the DOI [10.5281/zenodo.24131453](https://doi.org/10.5281/zenodo.24131453); [10.5281/zenodo.24131452](https://doi.org/10.5281/zenodo.24131452) always resolves to the latest release.

## Contents

```
systems.py                    circuits and lines; RK4 reference solutions
system_dataset.py             initial conditions of a dataset -> datasets/<id>/manifest.json
harmonic_models.py            models (LF, Res, RK4-net, LF-U, SIE, SIE-M, LF-KCL, RK4-KCL, ...)
harmonic_losses.py            losses: one step, 10-step rollout, input noise
train_ic_generalization.py    training
ic_generalization_eval.py     evaluation of a trained network (uses system_eval.py)
sweep_ic_generalization.py    training followed by evaluation
harmonic_*.py                 data loading, reference solutions, optimizer, timing, progress
datasets/                     manifests of the datasets
scripts/                      further evaluations, the letter's tables, and helpers (see below)
docs/planned_tests.csv        every planned comparison and its family, whether the letter reports it or not (see "Tables and daggers")
results/                      the per-run values behind the planned tests, Tables IV and V, and Figs. 4 and 5, the tables and plotted medians recomputed from them, and the outputs behind further numbers in the text
tests/                        tests (pytest)
```

Some names come from the history of the code. The prefix `harmonic_` and the name `ic_generalization` come from an earlier study of harmonic oscillators, and some modules still contain models and systems that the letter does not use. The dates in script names are the dates on which the scripts were written. Comments and docstrings are mostly in Japanese.

## Environment

Python 3.11, torch 2.1.2 (CUDA 11.8 wheels), numpy 1.26.4. `requirements.txt` lists the packages of the environment that trained and evaluated most networks, and `docker/Dockerfile` is the container used on one of the machines. On a CPU-only machine, install torch 2.1.2 from the CPU index instead. The tests also need pytest.

```
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install pytest
python -m pytest -q tests      # about 20 minutes on four CPU threads; some tests train small networks
```

Run every command from the repository root. Training writes each network to `picture/ic_gen/<dataset>/<run_id>/<model>/hs<width>/`. The trained networks of the letter are not included in this repository; the per-run values behind the planned tests, Tables IV and V, and Figs. 4 and 5 are in `results/`. The scripts in `scripts/` read trained networks from `$LLLC_RUNS_ROOT` (default `picture/ic_gen`) and write working files to `$LLLC_WORK_DIR` (default `work/`). Reference solutions are recomputed from the manifests by RK4 with step 10^-3 and cached in `datasets/<id>/ref_cache/`.

## Datasets

`datasets/<id>/manifest.json` holds the system settings and the initial conditions: 1024 for training, 256 for validation, 512 for testing, and 256 for extrapolation. The suffix `_v1_dtd2` means Δt = 0.05; `_v1_dt0p1`, `_v1_dt0p2`, and `_v1_dt0p4` mean Δt = 0.1, 0.2, and 0.4.

| Id | System (class in `systems.py`) |
|---|---|
| `nlqf3` | cubic LC circuit in charge and flux, u(q) = q + q³, i(φ) = φ + φ³ (`CubicChargeFluxLC`) |
| `nltl8`, `nltl16`, `nltl32` | ring of 8, 16, or 32 such sections (`ChargeFluxLine`) |
| `nltlg8`, `nltlg32` | the same ring with shunt conductance 0.03 (`ChargeFluxLossyLine`) |
| `nltlv8` | varactor ring, u(q) = q + 0.0925 q², i(φ) = φ (`VaractorLine`) |
| `nlqf3vi` | the cubic circuit in voltage and current (`CubicLCVoltageCurrent`) |
| `nlcap`, `nlcapqf` | circuit with a nonlinear C(V), in voltage–current and in charge–flux variables (`NonlinearCapLC`, `NonlinearCapLCChargeFlux`) |
| `nlind`, `nlindqf` | circuit with a nonlinear L(I) and capacitance 2, in the same two pairs of variables (`NonlinearIndLC`, `NonlinearIndLCChargeFlux`) |
| `nllc`, `nllc_e1`, `nllcqf` | circuit with both, in voltage–current variables (energies up to 0.5 and up to 1) and in charge–flux variables, where u(q) = tan q and i(φ) = tan φ: the tan circuit of `docs/planned_tests.csv` (`NonlinearLC`, `NonlinearLCChargeFlux`) |
| `duffing`, `duffing_b0p1`, `duffing_b0p3`, `duffing_b3` | Duffing oscillator ẍ = −x − βx³ with β = 1, 0.1, 0.3, and 3 (`Duffing`) |
| `dwell` | double-well oscillator ẍ = x − x³ (`DoubleWell`) |
| `morse` | oscillator in the Morse potential (1 − e^−x)² (`Morse`) |
| `fpu4` | four masses between fixed ends, joined by springs of potential d²/2 + d⁴/4 (`FPUChain`) |
| `charged`, `charged_B10`, `charged_B20` | charged particle in the plane, in the well (x² + 2y²)/2 + (x² + y²)²/4 and a magnetic field of 1, 10, and 20 (`ChargedParticle`) |

The last five rows are not LC circuits. Only the first three families of `docs/planned_tests.csv`, which are given only in this repository, use them; their error is that of the position.

## Models

The circuit models act on one section. The line models act on every section of a ring with one network shared by all sections, which reads a window of five sections; their names end in `_Loc`. The two widths of each model give about 700 and about 2400 parameters.

| Letter | Circuit: name (width → parameters) | Line: name (width → parameters) |
|---|---|---|
| LF | `LF_Free_MLP` (16 → 674, 32 → 2370) | `LF_Free_Loc` (13 → 678, 29 → 2438) |
| Euler-net (Res in the code) | `Res_MLP_state` (24 → 722, 46 → 2394) | `Res_Loc_state` (20 → 682, 43 → 2453) |
| RK4-net | `RK4_MLP_state` (as Res) | `RK4_Loc_state` (as Res) |
| LF-U | `LF_Par_MLP` (as LF) | `LF_Par_Loc` (as LF) |
| SIE | `SIE_MLP_state` (as LF) | `SIE_Loc_state` (as LF) |
| SIE-M | `SIEMid_MLP_state` (as LF) | `SIEMid_Loc_state` (as LF) |
| LF-KCL | `LF_Sep_MLP` (16 → 642, 32 → 2306) | `LF_KCL_Loc` (17 → 716, 33 → 2444) |

The letter's Euler-net is called Res in the code, in `docs/planned_tests.csv`, and in `results/`. Its conventional scheme is likewise called classical in the code, in script and file names, and in `results/`, and its Euler-split is called Res2 in the code, in `docs/planned_tests.csv`, and in `results/`. In a single circuit, `LF_Sep_MLP` (two networks, each taking only the other variable) has the same structure as learned element laws joined by Kirchhoff's laws, so the letter calls it LF-KCL. The driven varactor line uses the line models with about 2400 parameters only. For the charged particle and the chain of four masses, whose states have 4 and 8 variables, each circuit model takes the width whose parameter count is closest to 706 or 2434 (`memoryless_width` in `harmonic_models.py`).

Four further models appear in the letter, two of them without a name of their own, and two more, LF-SepD and RK4-KCL, appear only in tests given only in this repository. `docs/planned_tests.csv` calls them as follows.

| Name in the test list | In the letter | Circuit | Line |
|---|---|---|---|
| Res2 | Euler-split: Euler-net (Res) split into two networks, each fed the state at the step's start (Table I) | `Res2_MLP_state` (as LF) | `Res2_Loc_state` (as LF) |
| LF-Sep | each of LF's networks fed only the other variable (Table I) | `LF_Sep_MLP` (LF-KCL above) | `LF_Sep_Loc` (15 → 692, 31 → 2420) |
| LF-KCLD | LF-KCL with a learned shunt term, in the lossy ring (Sect. 4.1) | — | `LF_KCLD_Loc` (13 → 666, 27 → 2514) |
| LF-C | LF centering each update's own variable, in the V–I circuits (Sect. 4.4) | `LF_Cen_MLP` (as LF) | — |
| LF-SepD | not in the letter: LF-Sep with a learned loss term, in the lossy ring | — | `LF_SepD_Loc` (12 → 723, 25 → 2478) |
| RK4-KCL | not in the letter: LF-KCL's structure with separately trained scalar laws, advanced by RK4 | — | `RK4_KCL_Loc` (as LF-KCL) |

`LF_KCLD_Loc` adds to `LF_KCL_Loc` a leakage current from each node to ground, given by one scalar network shared by all sections at the mid-step charge. `LF_SepD_Loc` adds to the charge update of `LF_Sep_Loc` a loss term, given by a third network at the mid-step charge (q_n + q_{n+1})/2 and solved by three fixed-point iterations. `LF_Cen_MLP` also feeds each update its own variable at mid-step; this makes the update implicit, and three fixed-point iterations solve it in training and in test runs.

## Training and evaluation

One command trains one network for 1000 epochs and evaluates it on the test splits up to t = 10, 20, and 50. For example, LF with about 700 parameters in the cubic circuit, learning rate 10^-2, seed 0:

```
python sweep_ic_generalization.py --dataset-id nlqf3_v1_dtd2 --run-id abs_mse_lr1e2_bs64_gc1_num_ltctau_cos_s0_gpu \
    --models LF_Free_MLP --hidden-sizes 16 --seed 0 \
    --loss-type mse --lr 0.01 --batch-size 64 --grad-clip-max-norm 1.0 --optimizer adam --teacher numerical \
    --fast-onestep --lr-schedule cosine --lr-min-ratio 0.01 --epochs 1000 --t-evals 10 20 50 \
    --ltc-variant paper --ltc-activation relu --skip-dataset --skip-if-trained --no-show --no-progress
```

The networks of the letter used the learning rates 10^-2 and 3 × 10^-2 (`--lr 0.03`), the two widths in the table above, and seeds 0–9 (the first set); the second set used seeds 10–19 for LF, SIE, and LF-U. The tests of SIE/LF and LF-U/LF in the cubic circuit and the eight-section line were planned after the first-set values had been seen, so they use the second set only. A third set of varactor networks, seeds 20–49 for LF and SIE, appears only in tests given only in this repository. `--rollout-steps 10` selects the 10-step rollout loss, and `--input-noise 0.01` or `0.001` adds input noise. `--ltc-variant` and `--ltc-activation` set cells of the earlier study and do not affect these models.

Further settings of the letter:

- Every comparison uses both learning rates, 10^-2 and 3 × 10^-2, for every model. The rates 3 × 10^-3 and 10^-1 (`--lr 0.003`, `--lr 0.1`) enter only one check, given only in this repository: choosing one of the four rates per model and size by validation rollout error widened LF's circuit margins over Euler-net and RK4-net.
- The evaluated weights are those of the last epoch; `--best-val`, which keeps the weights of the best validation epoch, was not used.
- The staggered models (family `lf` in `harmonic_models.py`: LF, LF-U, LF-KCL, LF-Sep, LF-KCLD, and LF-C) are trained on the exact half-step values of the reference solution's second variable, the flux or, in V–I variables, the current (`--teacher numerical`), not on values interpolated from full steps; their test runs start from the reference value at t = −Δt/2.
- The free-run errors up to t = 50 use 1000 points for the staggered models (t = 0 to 49.95 at Δt = 0.05) and 1001 for the integer-time models (t = 0 to 50), because a staggered rollout has its flux at integer times only up to the step before the last (`system_eval.evaluate`). Leaving out the final time lowers the integer-time models' errors by 0.06–0.42% at Δt = 0.05. At Δt = 0.4 it lowers them by a median of 1.4% in the cubic circuit and 0.7% in the eight-section line, and single runs change by −2.9% to +2.7%. These values, over 1600 runs of Euler-net (Res), RK4-net, Euler-split (Res2), SIE, and SIE-M, are in `results/eval_points_runs.csv`, written by `scripts/eval_points_check.py`. The errors reported in the letter keep the two counts.
- `BHIC_LOCAL_INPUT_SCALE`, an input scale of the line networks read in `harmonic_models.py`, keeps its default of 1 for all networks of the letter.
- Input noise (`--input-noise`) is Gaussian, with a standard deviation of the given fraction of each variable's RMS over the training data. It is added only to the state at the start of each training step, (q_n, φ_{n−1/2}) or (q_n, φ_n); the later flux that the charge network of the staggered models, SIE, and SIE-M receives (φ_{n+1/2} or φ_{n+1}) and the targets stay exact.
- The varactor ring `nltlv8_v1_dtd2` starts from dc-shifted waves around the bias: all charges get one dc shift drawn uniformly in [−0.5, 2.6] and all fluxes one in [−1, 3.2], in the normalized units of `systems.VaractorLine`, and a smooth random wave of the same form as in the cubic ring is added, with a per-section RMS drawn uniformly in (0.05, 1] ((1, 1.44] for extrapolation), separately for charge and flux.
- `scripts/letter_data.py` finds the runs by their run ids: `abs_mse_lr<1e2 or 3e2>_bs64_gc1_num_ltctau_cos_s<seed>` followed by `_gpu` for the default training, `_ro10_gpu` for the 10-step rollout loss, `_nz1_gpu` and `_nz01_gpu` for 1% and 0.1% input noise, and `_cpu` for networks trained on CPUs: those of two V–I tests (Sect. 4.4) in the circuit with only L(I) nonlinear (LF-U/LF and SIE/SIE-M, about 700 parameters) and those of the systems that are not LC circuits.

The letter states that the training and test trajectories were computed in double precision and that the networks, alone or inside rings and lines, ran in single precision. The precision of each computation is as follows.

| Computation | Precision | Code |
|---|---|---|
| Training and test trajectories (fourth-order Runge–Kutta, substep 10^-3) | double; converted to single precision for training and stored in single precision for evaluation (relative rounding about 10^-7) | `systems.py` (`rk4_integrate`), `train_ic_generalization.py`, `system_eval.py` |
| Networks, including LF-KCL, in free runs | single | `system_eval.py` |
| Rings and lines with learned blocks (Sects. 4.2 and 4.3) | single | `scripts/hybrid_line_20261021.py`, `scripts/nltlv_*.py` |
| Fully conventional reference rings and lines computed alongside them (Sect. 4.3 compares with these) | double | the same scripts |
| Conventional row of Table IV, rollout-error columns; conventional curves of Fig. 5; conventional extrapolation in Table VI | single (the networks' evaluation path) | `scripts/line_classical_ref.py`, `scripts/dt_classical_ref_20261021.py` |
| Conventional row of Table IV, embedded columns (fully conventional ring) | single (the embedding path without a network) | `scripts/hybrid_line_20261021.py` |
| Conventional V–I update (values under "Details that the letter leaves to this repository") | double | `scripts/vi_voltage_error.py` |

A trained network can be evaluated again, on another split or on another dataset with the same system, for example an eight-section line network on 32 sections:

```
python ic_generalization_eval.py --model LF_Free_Loc --hidden-size 13 --dataset-id nltl8_v1_dtd2 \
    --run-id abs_mse_lr1e2_bs64_gc1_num_ltctau_cos_s0_gpu --t-evals 10 20 50 --splits test_interp \
    --data-dataset-id nltl32_v1_dtd2 --out-subdir eval_nltl32_v1_dtd2 --no-progress
```

Retraining gives slightly different networks on different machines: PyTorch does not guarantee identical results across releases, platforms, or devices, and the training does not force deterministic GPU kernels.

## Further evaluations

The scripts in `scripts/` run trained networks in further settings of the letter. Most of them give their usage at the top of the file.

| Script | What it does |
|---|---|
| `hybrid_line_20261021.py` | replaces eight contiguous sections of a 32-section conventional leapfrog ring with a trained eight-section line network, and measures divergence and energy change |
| `nltlv_open_20261022.py`, `nltlv_reflect_20261022.py` | driven varactor line with matched and with reflecting terminations: a trained block of 32 sections between conventional end sections |
| `nltlv_diag_20261022.py`, `nltlv_diag_20261023f_restfix.py` | further drives of the varactor line (`--mode`) and measures inside the block; the second file corrects the LF-U case of `--rest-correct` and `--mode static` |
| `nltlv_waveform_diag_20261023.py` | splits the driven line's load-voltage error into the times with and without a pulse at the load, and measures the error of the energy delivered to the load (tests given only in this repository) |
| `nltlv_pulse_physics_20261023.py` | per-network dc load voltage with the sources off and pulse arrival times in the driven varactor line (values in Sect. 4.3 of the letter) |
| `eval_points_check.py` | scores the integer-time models with and without the final time (see "Further settings of the letter") |
| `line_classical_ref.py`, `dt_classical_ref_20261021.py` | the conventional leapfrog scheme with the true element laws, evaluated as the networks are, at Δt = 0.05 and at the other time steps |
| `classical_div_20261022.py` | counts the initial conditions for which the conventional scheme diverges in the ring |
| `classical_centered_20261022.py` | conventional updates in voltage–current and in charge–flux variables, with and without centering each update's own variable |
| `vi_voltage_error.py` | voltage error of circuit networks trained in V–I and in charge–flux variables (Sect. 4.4) |
| `freerun_energy_20261023.py` | energy change and divergence of the free-running networks over 1000 and 10^4 steps (Sect. 4.1) |
| `energy_projection_20261020.py` | one-step errors of the two networks of LF and of the ablations, and the energy change that they imply (Sect. 4.1) |
| `learned_start_20261028.py`, `kcl_leaky_20261028.py` | LF started half a learned step back (Table VI); LF-KCL and LF-KCLD against LF in the lossy ring (Sect. 4.1) |
| `nltlv_fine_rk_ref_20261029.py` | the fully conventional varactor line against RK4 with substep 0.005 (Δt/10) under every drive of Table V (Sect. 4.3) |
| `letter_tables.py`, `letter_stats.py`, `letter_data.py` | the planned tests, Table IV, Table V, and Fig. 3 from the saved per-run errors or from the outputs of the commands above (see "Tables and daggers") |
| `letter_figures.py` | the medians plotted in Figs. 4 and 5, and the legend of Fig. 4, from the saved per-run values or from the same outputs (see "Tables and daggers") |
| `build_ref_cache.py`, `derive_dataset.py`, `lllc_paths.py` | helpers: reference solutions computed in advance, datasets derived from others, and the locations of the runs and working files |

For example:

```
python scripts/hybrid_line_20261021.py --block 8 --steps 10000 --out embedded.json
python scripts/nltlv_open_20261022.py --out open.json
python scripts/nltlv_reflect_20261022.py --out reflect.json
```

## Tables and daggers

`results/letter_runs.csv` holds, for every run that the planned tests, Table IV, and Table V use, its value of each measure: one row per run and measure, with the columns source, size, lr, model, seed, and value (`scripts/letter_data.py` describes each source). `results/letter_classical.csv` holds the values of the conventional scheme in Table IV. From these two files, with numpy alone, `scripts/letter_tables.py` recomputes the planned tests, Table IV, Table V, and Fig. 3 of the letter; `results/letter_tables.md` is its output:

```
python scripts/letter_tables.py all > results/letter_tables.md   # every planned test, Table IV, Table V, Fig. 3
python scripts/letter_tables.py tests --family varactor --json varactor.json
```

With `--raw`, the script reads the trained runs (`--runs-root`, default `$LLLC_RUNS_ROOT`) and the outputs of the evaluation scripts (`--outputs`, default `$LLLC_OUTPUTS` or `work/outputs`) instead; `python scripts/letter_tables.py export` rewrites the two CSV files from them.

`results/fig4_runs.csv` and `results/fig5_runs.csv` hold the per-run values of Figs. 4 and 5. For the embedded ring, they give the energy change and the number of diverged trajectories at each plotted step; for Fig. 5, the free-run error up to t = 50 and the number of diverged trajectories at each time step (a trajectory has diverged when its error is non-finite or above 100, and the error averages over the others). Both include the reference curves. The line at Δt = 0.4, where all generic networks failed, is included but not plotted. The conventional errors are those plotted, and `dt_classical_ref_20261021.py` also gives the counts: run on another device, it gives the same counts and errors that agree with the saved ones to about 10^-8 (relative). `scripts/letter_text.py` writes the values quoted in the letter's text, and some beside them, to `results/letter_text.md`, `results/text_values.json`, and `results/text_runs.csv`; for example, RK4-net with about 700 parameters against LF with about 2400 at Δt = 0.05 is in `results/letter_text.md`. `scripts/letter_figures.py` prints the plotted medians from these files, and `results/letter_figures.md` is its output; `--raw` and `export` work as for `letter_tables.py`. Table IV and the legend of Fig. 4 round the medians 19.5 (SIE) and 48.5 (RK4-net) of the diverged trajectories to 20 and 49. Sect. 4.4's RK4-net/LF of 1.42 (circuit, Δt = 0.2), 3.26 (circuit, Δt = 0.1), and 1.55 (line, Δt = 0.1) are geometric means over the matched runs of `results/fig5_runs.csv`.

```
python scripts/letter_figures.py > results/letter_figures.md   # the medians of Figs. 4 and 5
```

`docs/planned_tests.csv` lists all 128 planned comparisons in 28 families of 1 to 12 tests, in the order in which the families were planned. The letter shows 76 of them; the other 52, marked "repository only", are given here with their results. Ten whole families are given only here: the first three, on systems that are not LC circuits; the tan circuit; the lossy ring against LF-SepD; RK4-KCL in the line; and, on the driven line, the load-voltage tests with uniform-state increments subtracted, the block-energy tests of the second and third sets, and the components of the load-voltage error and delivered energy. Each row gives the family and the test, where the letter shows it (or "repository only"), the system and the measure, the models A and B, the seeds and the sizes, the direction tested, and the source of the values; `scripts/letter_data.py` describes each source. In the source column, `X / Y` is a ratio of two ratios (for example A/B at Δt = 0.4 over A/B at Δt = 0.05), and `X + Y + Z` averages the log ratios over several sources, for example three drives. Models written as `A1/B1` and `A2/B2` give the ratio of two ratios in one condition, (A1/B1)/(A2/B2), matched by seed. `X < Y < Z` (H4) is a criterion without a p-value: it holds when the ratios rise in that order and the interval of the last lies above that of the first. Names of files, families, and result keys keep the word `leaky` for the lossy ring.

Of the 52 tests given only here, 22 compare LF directly with another model (values in `results/letter_tables.md`). Against generic networks, the nine ratios of G5, G6, W1–W4, and the block-energy tests exceed 1, significantly in G5 and G6 (Euler-net/LF 2.23 on 32 sections of the lossy ring and 1.93 in the embedded lossy ring), in W1 and W3 (LF-U/LF 1.34 in the load-voltage error while a pulse is at the load, and 1.96 in the error of the energy delivered to the load), and in the block-energy deviation (varactor-energy-2 / E1 and E2, LF-U/LF 1.60 and SIE/LF 1.37 under the second set's pulses; varactor-energy-3 / F1, SIE/LF 1.44 under the third set's). With uniform-state increments subtracted (varactor-rest-corrected / Q1–Q6), SIE/LF and LF-U/LF in load voltage were 0.965–1.03, none with a dagger. Against models given more structure, LF was less accurate, as in the letter, in H7a and kcl-line / K2 (LF/LF-KCL 5.1 in the tan circuit and 17.2 in the energy change of the embedded ring), in rk4-kcl / F1 (LF/RK4-KCL 31.4 in the eight-section line, where RK4-KCL had 0.19 times LF-KCL's error), and in L8 (LF/LF-Sep 1.16 with 32 sections), and could not be told apart from LF-SepD in leaky-structured / S1, S4, and S5. Of the 17 tests on a ratio of two ratios, the nine of H1, H4, and H6a held as planned (H6a only with the rule of Sect. 3.3; see below), and the eight of R5, R6, P3–P6, W5, and W6 did not. The other 13 do not compare LF with another model: leaky-structured / S2 and S3 found LF-SepD more accurate than Euler-net and RK4-net (3.03 and 1.56), and the eleven others compare SIE with SIE-M.

The letter compares each ablation with LF, not the ablations with each other. In the cubic circuit and the line, SIE-M was less accurate than SIE, as planned (C3 and L3, SIE-M/SIE = 1.63 and 1.19); it was also less accurate in the tan circuit, against the plan (H6b, SIE/SIE-M = 0.264), and more accurate in the seven tests on systems that are not LC circuits, as planned (H2, SIE/SIE-M = 1.9–7.8). Sect. 2's account does not fix the order of SIE and SIE-M: with the offsets of Table I, their drift rates are proportional to ε_G − ε_F and −ε_F, so SIE drifts less when 0 < ε_G/ε_F < 2, and the letter does not estimate ε_G or ε_F.

`scripts/letter_stats.py` follows Sect. 3.3 of the letter:

- Runs of A and B are matched by seed, size, and learning rate, with D = log(e_A/e_B) for each pair; the ratio is exp(mean D).
- A size × learning-rate stratum is left out when both models' medians are 0.3 or more (both failed), or when either model has fewer than five runs in it. In the planned tests, only the first rule leaves out strata; line / L2, for example, keeps 30 of 40 pairs.
- The interval is a 95% percentile bootstrap over seeds (10^4 draws; a drawn seed brings all its runs).
- D is tested by two upper-tailed sign-flip tests on the ratio expected to exceed 1 (the `direction` column of `docs/planned_tests.csv`): over runs, with 10^5 random sign patterns and p = (1 + c)/(1 + N), and over seed means, exactly over all 2^n patterns for up to 20 seeds and with 10^5 random patterns for more (the third varactor set has 30 seeds).
- The letter writes two values with a dagger as the inverse of the tested ratio: LF-Sep's 0.69 times LF's error is line / L6 (LF/LF-Sep), and LF-KCL's 0.15 times LF's error with the shunt term is kcl-line / K3 (LF/LF-KCLD).
- Holm's correction is applied within each family, to the two tests separately. A dagger marks a test whose two corrected p are both below 0.05. A test with no stratum left counts in its family with p = 1.

The plans of the first five families (2026-10-19 and 10-21) differed from Sect. 3.3 in three details: a stratum was left out when either model failed, H1 and H6a compared the two log ratios by a two-sample permutation test within each stratum, and H5a and H5b were tested over seed means only. The script computes every family as in Sect. 3, and `--leave-out any` applies the first of these rules. With the plans' own procedures, every outcome is the same as in `results/letter_tables.md` except that of H6a, which did not hold; with `--leave-out any`, H6a keeps one stratum and gives 1.23 [0.73, 2.19] with Holm-corrected p = 0.51. Applied to every test, the option also leaves no stratum in G1 (LF-Sep failed in every stratum of the lossy ring) and in T2-N-line, keeps every other dagger, and changes some ratios (embedded / E1, for example, from 13.4 to 8.66).

Holm's correction can also be applied across all 122 planned tests that have p-values, instead of within each family; the other six are H4, an ordering, and five tests with 1% input noise, in which no stratum is left. Over runs, this correction keeps p < 0.05 for 68 of the 77 tests with a dagger. Over seed means, it keeps only the eight tests with 20 or 30 seeds, since with ten seeds the p-value cannot fall below 1/1024. The nine tests that lose p < 0.05 over runs are G3, V5, R3, R4, and P1, which the letter marks with daggers (RK4-net/LF in the lossy ring; Euler-net/LF under matched pulses; Euler-net/LF and RK4-net/LF under reflected pulses; SIE/LF under sinusoids), and L3, L8, S3, and varactor-energy-2 / E2 (SIE/LF in the second set's block energy), which are given only here. The p-values before correction are in `results/letter_tables.md`.

Each comparison draws its random numbers from its own fixed stream, so the output does not change from run to run. The intervals and Monte Carlo p-values in the letter were computed with other random draws and can differ from this output in the last digit; the ratios, the medians, and the daggers agree.

The commands below fill the outputs folder (`$O`), whose subfolders must exist first (`mkdir -p $O/embedded $O/leaky_embedded $O/drives $O/waveforms`). `M` stands for one model per call, with the names of the scripts: LF, SIE, LF-U, Res, RK4 (RK4-net), and LF-KCL. All scripts are in `scripts/`.

| File in `$O` | Command | Used in |
|---|---|---|
| `embedded/all.json` | `hybrid_line_20261021.py --block 8 --steps 10000 --horizons 100 200 500 1000 2000 5000 10000 --out $O/embedded/all.json` | Table IV, Figs. 3 and 4 |
| `embedded_references.json` | `hybrid_line_20261021.py --block 8 --steps 10000 --horizons 100 200 500 1000 2000 5000 10000 --references --out $O/embedded_references.json` | Table IV, Fig. 4 |
| `leaky_embedded/all.json` | `hybrid_line_20261021.py --train-ds nltlg8_v1_dtd2 --test-ds nltlg32_v1_dtd2 --loss-g 0.03 --block 8 --steps 1000 --out $O/leaky_embedded/all.json` | two tests given only in this repository |
| `classical_steps.json` | `dt_classical_ref_20261021.py nlqf3_v1_dtd2 nlqf3_v1_dt0p1 nlqf3_v1_dt0p2 nlqf3_v1_dt0p4 nltl8_v1_dtd2 nltl8_v1_dt0p1 nltl8_v1_dt0p2 nltl8_v1_dt0p4 --out $O/classical_steps.json` | Table IV, Fig. 5 |
| `classical_line.json` | `line_classical_ref.py --out $O/classical_line.json` | Table IV |
| `vi_voltage.json` | `vi_voltage_error.py --out $O/vi_voltage.json` | Sect. 4.4 |
| `pulses.json` | `nltlv_open_20261022.py --out $O/pulses.json` | Table V |
| `reflected.json` | `nltlv_reflect_20261022.py --out $O/reflected.json` | Table V |
| `drives/<drive>_M.json` | `nltlv_diag_20261022.py --mode <mode> --models M --out $O/drives/<drive>_M.json`, as in the next table | Table V |
| `drives/<mode>_rc_M.json` | `nltlv_diag_20261023f_restfix.py --mode <mode> --rest-correct --models M --out $O/drives/<mode>_rc_M.json`, with `<mode>` open, train, or prbs | six tests given only in this repository |
| `waveforms/<mode>_new_M.json` | `nltlv_waveform_diag_20261023.py --mode <mode> --models M --seeds 10 19 --out $O/waveforms/<mode>_new_M.json`, with `<mode>` open, reflect, or train (adding `--sig-seed 20261031`), and M LF, SIE, or LF-U | six tests given only in this repository |

The other values are read from the training runs themselves, among them those of Fig. 5 at Δt = 0.1, 0.2, and 0.4. The values with 32 sections (in Table IV, Fig. 3, and four tests) also need the evaluation on 32 sections shown above (`--out-subdir eval_nltl32_v1_dtd2`, or `eval_nltlg32_v1_dtd2` for the lossy ring).

In the embedded ring (`embedded/all.json`), each initial condition has two measures: its RMS charge deviation from the fully conventional ring (relative to the largest conventional |q|, at least 0.05) and its energy change. It counts as diverged in a measure that is non-finite or above 100, and each mean leaves out the initial conditions diverged in its own measure. The diverged trajectories in Table IV and Fig. 4 (`n_div`) are the larger of the two counts; the number diverged in either measure, which can be larger, was not recorded. `--horizons` only adds steps at which these values are recorded; the values at 1000, 2000, and 10^4 steps are the same with and without it. At the interfaces of the embedded ring and of the driven line, the conventional sections next to the learned block use the true element laws of their learned neighbours; this would be unavailable for an element known only from data.

The driven line uses the line networks of about 2400 parameters trained on `nltlv8_v1_dtd2`, with seeds 0–9 unless stated. Table V of the letter gives load voltage, or charge on the training ring, without the rows "Pulses, 2nd set" and "Pulses, 3rd set"; those rows and the block energy are given only in this repository (`results/letter_tables.md`). In the next table, the options are those of `nltlv_diag_20261022.py --models M`:

| Drive | Load voltage, or charge on the training ring | Block energy |
|---|---|---|
| Training ring | the training runs | — |
| Pulses, matched | `pulses.json` | `--mode open` → `drives/open_M.json` |
| Pulses, reflected | `reflected.json` | `--mode reflect` → `drives/reflect_M.json` |
| Pulse trains, bit streams, sinusoids | `--mode train`, `--mode prbs`, `--mode cw` → `drives/train_M.json`, `drives/prbs_M.json`, `drives/cw_M.json` | the same files |
| Pulses, 2nd set | `--seeds 10 19` with `--mode open`, `--mode reflect`, and `--mode train --sig-seed 20261031` → `drives/open_new_M.json`, `drives/reflect_new_M.json`, `drives/train_new_M.json` | the same files |
| Pulses, 3rd set | the same with `--seeds 20 49` → `drives/open_third_M.json`, `drives/reflect_third_M.json`, `drives/train_third_M.json` | the same files |
| Sinusoids, 2nd set | `--mode cw --seeds 10 19 --sig-seed 20261029` → `drives/cw_new_M.json` | the same file |

The matched and reflected pulses of the 2nd and 3rd sets are the signals of the first set, run with networks of other seeds; only their pulse trains are new (`--sig-seed 20261031`).

### The driven varactor line (Sect. 4.3)

The line is an idealized 50-Ω nonlinear transmission line of varactors, C(V) = C_j0/√(1 + V/V_bi) at reverse voltage V (C_j0 = 1 pF, built-in potential V_bi = 0.7 V, reverse bias 2 V, so that the bias-point capacitance is 0.509 pF), and linear 1.27-nH inductors; the normalized units are 1 V, 0.509 pC, 20 mA, and t0 = 25.4 ps (`systems.VaractorLine`). Under the matched pulses of Table V, the fully conventional line shortens the median 10–90% rise time from 122 to 72 ps. The block energy Σ_k [U(q_k) + K(φ_k)] of the 32 learned sections, given only in this repository, is measured from the bias with q_{n+1} and φ_{n+1/2}; for the full-step models, φ_{n+1/2} is the mean of φ_n and φ_{n+1}. The block-energy deviation is not a drift, since energy crosses the interfaces. Under bit streams, it excludes the two signals whose bits are all zero, where the conventional block energy is 0.

Table V measures every error against the fully conventional line. Against fine Runge–Kutta solutions (substep Δt/10), the fully conventional line's RMS load-voltage deviation is at most 0.33% of the amplitude and a seventh of LF's median error under each drive (`nltlv_fine_rk_ref_20261029.py --runs-csv results/letter_runs.csv --out results/fine_rk_reference.json`, which uses no trained network; outputs `results/fine_rk_reference.txt` and `.json`).

### Details that the letter leaves to this repository

The letter is limited to six pages, so the following details of its setup and tests are given here.

- **Training.** Besides the settings in Table II of the letter, every network was trained with a cosine learning-rate schedule (to 1% of the initial rate), gradient clipping (maximum norm 1), and 64 trajectories per batch, as in the command above.
- **Courant number.** In the line, Δt = 0.05 is a Courant number of 0.05 for small signals and 0.11 at the highest energy.
- **Energy of staggered states.** In the standalone rollouts and the embedded ring, the energy of a staggered state uses the flux averaged to integer times, that is, the mean of the half-step fluxes before and after.
- **Energy change of the embedded ring.** For each number of steps T in Fig. 4, the energy change is |⟨H⟩ − H(0)|/H(0) with ⟨H⟩ the mean over the last tenth of the T steps; the letter's "last 10^3 steps" is the case T = 10^4 (Table IV).
- **Error scale.** The rollout error divides by the reference amplitude of q, at least 0.05: the amplitude given by the energy in the single circuit, and the largest |q| over all sections of the reference solution up to t = 10 in rings and lines.
- **Reference data.** The reference solutions are computed in double precision and stored in single precision (see the precision table above).
- **p-values of Fig. 3.** Circuit 0.0049; rings with N = 8 and N = 32, 0.0078; lossy ring 0.0059, but 0.013 for RK4-net; embedded ring 0.0029; second set 0.0039; rollout loss 0.012; 0.1% noise 0.012 (circuit) and 0.0059 (line). With ten seeds, an uncorrected seed-mean p cannot fall below 2^-10.
- **Drives of the varactor line (Sect. 4.3).** Each drive has 64 signals.
  - Pulses, matched: 50-Ω terminations, matched at small signals and low frequencies; pulses of 1–3 V with raised-cosine edges of 100–300 ps, propagated for 2.5 ns.
  - Pulses, reflected: terminations of 5 and 500 Ω; pulses of 0.5–1.0 V, which cross the block about ten times in 10 ns.
  - Pulse trains: the matched line driven for 5.1 ns by 1–3-V pulses repeated every 1.5–3 ns.
  - Bit streams: random non-return-to-zero bit streams of 0 or 0.5–1.5 V at 1–2.5 Gb/s, for 5.1 ns.
  - Sinusoids: 0.5–1.5 V at 1.2–4.7 GHz, for 5.1 ns.
- **V–I variables (Sect. 4.4).** The four circuits are the cubic circuit and circuits with incremental C(V) = 1/(1 + V^2), with incremental L(I) = 1/(1 + I^2) (and C = 2), or with both; the last is a section of Hirota's self-dual network. Energies are 0.00125–0.5 with C(V), and also 0.0025–1 with both. LF-C centers each update's own variable by fixed-point iteration, and a conventional V–I update was also run.
  - LF's voltage error with V–I training over that with charge–flux training: 3.2† (cubic), 1.6† (only L(I) nonlinear; each p = 0.0029), 1.13 (p = 0.11; only C(V) nonlinear), 1.86 (both).
  - In V–I variables with only L(I) nonlinear, LF-U/LF = 1.03 (p = 0.51).
  - Centering each update's own variable gave no significant gain over LF or Euler-split in eight tests (each p = 1).
  - The conventional V–I update in the cubic circuit has a relative voltage error of 0.63, or 0.011 with fixed-point iteration.

### Numbers in the text

The scripts below give further numbers in Sects. 4.1 and 4.3, and their outputs are in `results/`. The scripts read the trained runs from `--root` (default `$LLLC_RUNS_ROOT`).

| Number in the letter | Command | Output in `results/` |
|---|---|---|
| Table VI: LF started half a learned step back, error 1.33- and 1.02-fold; Euler-net/LF 3.3 and 3.7, RK4-net/LF 2.8 and 2.1 | the evaluation below for every LF run of the circuit and the line, then `learned_start_20261028.py --out results/learned_start.json` | `learned_start.json` (`LF_learned/LF_ref`, `Res/LF_learned`, `RK4/LF_learned`) |
| Sect. 4.1: LF-KCL/LF = 19 in the lossy ring without a learned shunt term | `kcl_leaky_20261028.py --out results/kcl_leaky.json` | `kcl_leaky.json` |
| Sect. 4.1: energy change over 1000 steps, 3.3–9.0 times LF's for Euler-net and RK4-net and 2.2–8.4 times for the ablations; over 10^4 steps in the line, a median of 1 diverged trajectory with LF and 38.5–85.5 with the others | `freerun_energy_20261023.py --out results/freerun_energy` | `freerun_energy.txt`, `freerun_energy.json` |
| Sect. 4.1: one-step circuit errors of the ablations at most 1.6 times LF's | `energy_projection_20261020.py --dataset-id nlqf3_v1_dtd2 --size small --out results/one_step_circuit_small.json`, and the same with `medium` | `one_step_circuit_small.json`, `one_step_circuit_medium.json` (`rms_ratio_to_LF`) |

The evaluation from half a learned step back, φ_{−1/2} = φ_0 − (Δt/2) F(q_0, φ_0), writes to `eval_learnedstart/` in each run, for example

```
BHIC_LF_LEARNED_START=1 python ic_generalization_eval.py --model LF_Free_MLP --hidden-size 16 --dataset-id nlqf3_v1_dtd2 \
    --run-id abs_mse_lr1e2_bs64_gc1_num_ltctau_cos_s0_gpu --t-evals 50 --splits test_interp --out-subdir eval_learnedstart --no-progress
```

and with `LF_Free_Loc` on `nltl8_v1_dtd2` for the line.

## Further explorations

Apart from the six load-voltage tests that the last item extends, the comparisons below are not planned tests, and their settings differ from the letter's as noted. Ratios are geometric means over runs matched by seed, size, and learning rate, with 95% bootstrap intervals over seeds and no correction for multiple comparisons. The datasets, networks, and scripts of the first two are not in this repository.

- Hirota's self-dual line, u = tan q and i = tan φ, with about 700 and 2400 parameters (40 runs per model). Trained on sums of sinusoids on eight sections, as in the letter, LF had the lowest error of the generic networks in the letter's rollout test over 1000 steps (SIE/LF 1.40 [1.09, 1.76], RK4-net/LF 1.47 [1.01, 2.15], and 1.60–1.86 for SIE-M, LF-U, and Euler-net). Trained on 32 sections with broader wavenumbers, only SIE-M/LF and LF-U/LF had intervals above 1 (1.25 and 1.43); SIE/LF, Euler-net/LF, and RK4-net/LF were 1.18 [0.96, 1.52], 1.01 [0.65, 1.56], and 1.11 [0.72, 1.75].
- Lines coarse-grained to cells of three or five sections, keeping the charge and flux of every third or fifth section, at Δt = 0.05 (6 runs per model). With the letter's energies, tried with cells of three sections, every learned model and the conventional coarse-grid scheme failed (median errors 0.42–0.85). With energies up to 0.03 per section, the ablations' ratios to LF were 1.00–1.01, while Euler-net/LF and RK4-net/LF were 0.79–0.84 and 0.79–0.86.
- Diagnostics of the first-set varactor networks with `nltlv_diag_20261022.py`, before the pulse trains, bit streams, and sinusoids of Table V were planned. `--mode busy` drives the matched line for 5.1 ns with a dc level of 0–1.5 V and three sinusoids of wavelengths 4–10 sections, `--mode busy_low` with wavelengths 20–50 sections, and `--mode burst` with the busy signals windowed for 24 t0 every 80 t0; `--mode ring` splits the training ring's test initial conditions into thirds by the squared differences between adjacent sections that set the drift rate of Sect. 2's account in a line. In load voltage, SIE/LF and LF-U/LF were 1.33 [1.07, 1.67] and 1.39 [1.02, 1.90] under the busy signals and 1.13–1.22, every interval including 1, under the other two; in charge error on the ring, they were 1.45–1.50 and 1.54–1.63 in the three thirds. Under the matched and reflected pulses, the other measures of these runs, charge and flux inside the block, charge on all 40 sections, and block energy, gave SIE/LF 1.03–1.25 and LF-U/LF 1.03–1.51, with intervals above 1 only for LF-U/LF in block energy (1.38 and 1.51); the block-energy tests of the second and third sets were chosen from these diagnostics.
- The driven varactor line with each learned section's uniform-state increments subtracted, as in the six planned load-voltage tests of varactor-rest-corrected, under every drive above (`nltlv_diag_20261023f_restfix.py --rest-correct` with the options of the drives above; the per-run values are in `results/text_runs.csv`). A section's uniform-state increments are its outputs with every section at its charge and flux, which are zero for the true update. Among these runs, one RK4-net and one SIE network diverged, each on 3 of 64 sinusoids. Under pulses and bit streams, SIE/LF and LF-U/LF were 0.87–1.06 in block energy and 0.95–1.10 in load voltage, every interval including 1; the block-energy ratios of the later pulse sets, 1.37, 1.60, and 1.44 without the subtraction (`results/letter_tables.md`), became 0.96, 1.06, and 1.06. Under sinusoids, they were 1.16–1.32 in block energy and 1.11–1.45 in load voltage, with three of the four intervals of the second set above 1. Under the first set's four matched drives, Euler-net/LF and RK4-net/LF stayed at 1.25–2.21 in block energy and 1.46–1.92 in load voltage.

## Citation

Please cite the letter. `CITATION.cff` has the details. To refer to the code itself, cite the Zenodo record of the release you used.

```
@article{sakai2026leapfrog,
  author  = {Sakai, Tomohiro and Oshio, Reon and Suzuki, Yukihisa},
  title   = {Leapfrog scheduling as an inductive bias for learning updates in nonlinear {LC} circuits},
  journal = {IEICE Electronics Express},
  year    = {2026},
  note    = {submitted}
}
```

## License

MIT License. See `LICENSE`.
