#!/usr/bin/env python3
"""The numbers that the letter's text, Tables II, III, and VI, and README.md state beyond the planned tests, Tables IV
and V, and Figs. 3-5, computed from the per-run values.

  python scripts/letter_text.py           every such number and what it is computed from (results/letter_text.md)
  python scripts/letter_text.py export    rewrite results/text_runs.csv and results/text_values.json

results/text_runs.csv has the columns of results/letter_runs.csv and holds the runs that the planned tests and tables
do not use: further models of the sources of scripts/letter_data.py, and the sources of TEXT_SOURCES below. Printing
merges the two files, so that a comparison can take one model from each. results/text_values.json holds what has one
value per setting: the classical scheme's errors and diverged initial conditions, the driven line's signals, rise
times, and block crossings, the signals that the block energy leaves out, the runs that diverged under a drive, and the
free runs with diverged initial conditions.
Printing also reads results/fig5_runs.csv and docs/planned_tests.csv, and runs the planned tests of
scripts/letter_tables.py for the counts and the corrections that README.md reports. Ratios follow
scripts/letter_stats.py (geometric means of matched ratios, strata where both models failed left out, seed intervals),
except where a line says otherwise.

export reads the trained runs (--runs-root) and the outputs folder (--outputs) of scripts/letter_data.py, which here
also holds
  embedded_charge/*.json   hybrid_line_20261021.py --block 8 --steps 10000 --models M --seeds 0 4 (and 5 9), M one of
                           LF, Res, RK4, SIE-M, LF-Sep, SIE, LF-U, and Res2: the total charge and flux of the embedded
                           ring (those of LF-KCL are in embedded/)
  drives/<drive>_rc_M.json nltlv_diag_20261023f_restfix.py --rest-correct with the options of each drive in README.md:
                           reflect_rc, cw_rc, open_new_rc, reflect_new_rc, train_new_rc, cw_new_rc, open_third_rc,
                           reflect_third_rc, and train_third_rc, besides open_rc, train_rc, and prbs_rc
  drives/<mode>_M.json     nltlv_diag_20261022.py --mode <mode>, with <mode> busy, busy_low, or burst
  ring_thirds.json         nltlv_diag_20261022.py --mode ring
  pulses_reference.json    nltlv_open_20261022.py --reference
  classical_div.txt        classical_div_20261022.py nltl32_v1_dt0p2 nltl32_v1_dt0p1 nltl8_v1_dt0p2
  classical_centered.json  classical_centered_20261022.py (it writes $LLLC_WORK_DIR/crossvar/classical_centered_20261022.json)
  pulse_physics/<set>_M.json  nltlv_pulse_physics_20261023.py --set <set> --models M: per-network dc offsets and pulse
                           arrival times of the varactor line (sources dc-offset and arrival-drift)
and the evaluations on the validation split (ic_generalization_eval.py --splits val --out-subdir eval_val) of the
cubic-circuit runs with learning rates 3e-3, 1e-2, 3e-2, and 1e-1. export also runs the all-classical reflecting line
of nltlv_reflect_20261022.py once, without networks, to time the block crossings.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import letter_data as LD  # noqa: E402
import letter_stats as LS  # noqa: E402

TEXT_RUNS_CSV = LD.REPO / "results" / "text_runs.csv"
TEXT_VALUES = LD.REPO / "results" / "text_values.json"
FIG5_CSV = LD.REPO / "results" / "fig5_runs.csv"
PLAN = LD.REPO / "docs" / "planned_tests.csv"
INTERP, EXTRAP = "test_interp_x_rmse_rel_mean", "test_extrap_x_rmse_rel_mean"
DIVERGED = "test_interp_x_rmse_rel_n_diverged"
FIRST, SECOND, THIRD = range(0, 10), range(10, 20), range(20, 50)
LR_GRID = ("3e3", "1e2", "3e2", "1e1")  # 3e-3, 1e-2, 3e-2, 1e-1: the check of Table II that chooses one rate by validation
T4_MODELS = ("LF", "Res", "RK4-net", "LF-U", "SIE", "SIE-M", "LF-KCL")

# Free runs that letter_data.FREE does not have: source -> (training dataset, models, subfolder, end of the run id, key)
TEXT_FREE = {
    "line-32-dt0.2": ("nltl8_v1_dt0p2", LD.LINE, "eval_nltl32_v1_dt0p2", "_gpu", INTERP),
    "circuit-extrap": ("nlqf3_v1_dtd2", LD.CIRCUIT, "", "_gpu", EXTRAP),
    "line-extrap": ("nltl8_v1_dtd2", LD.LINE, "", "_gpu", EXTRAP),
    "circuit-lr": ("nlqf3_v1_dtd2", LD.CIRCUIT, "", "_gpu", INTERP),
    "circuit-lr-val": ("nlqf3_v1_dtd2", LD.CIRCUIT, "eval_val", "_gpu", "val_x_rmse_rel_mean"),
}
LR_MODELS = ("LF", "Res", "RK4-net", "SIE-M")
# Driven varactor line, sources that letter_data.DRIVES does not have: source -> (file prefix in drives/, key of the run)
TEXT_DRIVES = {
    "reflected-rc": ("reflect_rc", "load"), "sinusoids-rc": ("cw_rc", "load"), "pulses-rc-2": ("open_new_rc", "load"),
    "reflected-rc-2": ("reflect_new_rc", "load"), "trains-rc-2": ("train_new_rc", "load"),
    "sinusoids-rc-2": ("cw_new_rc", "load"), "pulses-rc-3": ("open_third_rc", "load"),
    "reflected-rc-3": ("reflect_third_rc", "load"), "trains-rc-3": ("train_third_rc", "load"),
    "pulses-rc-energy": ("open_rc", "energy"), "reflected-rc-energy": ("reflect_rc", "energy"),
    "trains-rc-energy": ("train_rc", "energy"), "bits-rc-energy": ("prbs_rc", "energy"),
    "sinusoids-rc-energy": ("cw_rc", "energy"), "pulses-rc-energy-2": ("open_new_rc", "energy"),
    "reflected-rc-energy-2": ("reflect_new_rc", "energy"), "trains-rc-energy-2": ("train_new_rc", "energy"),
    "sinusoids-rc-energy-2": ("cw_new_rc", "energy"), "pulses-rc-energy-3": ("open_third_rc", "energy"),
    "reflected-rc-energy-3": ("reflect_third_rc", "energy"), "trains-rc-energy-3": ("train_third_rc", "energy"),
    "busy": ("busy", "load"), "busy-low": ("busy_low", "load"), "burst": ("burst", "load"),
    "pulses-charge-block": ("open", "q_blk"), "pulses-flux-block": ("open", "f_blk"), "pulses-charge-all": ("open", "q_all"),
    "reflected-charge-block": ("reflect", "q_blk"), "reflected-flux-block": ("reflect", "f_blk"),
    "reflected-charge-all": ("reflect", "q_all"),
}
# Sources of letter_data.py whose further models are written: source -> seeds
MORE_MODELS = {s: FIRST for s in ("line-dt0.4", "circuit-noise1", "line-noise1", "ring", "pulses", "reflected", "trains", "bits",
                                  "sinusoids", "pulses-energy", "reflected-energy", "trains-energy", "bits-energy",
                                  "sinusoids-energy", "pulses-rc", "trains-rc", "bits-rc", "voltage-cap",
                                  "voltage-ind", "voltage-cubic")}
TEXT_SOURCES = {
    **{s: f"free run, {v[0]}" + (f" networks on {v[2][5:]}" if v[2].startswith("eval_n") else "")
       + f": {'validation' if 'val' in v[4] else 'extrapolation' if 'extrap' in v[4] else 'test'} error up to t = 50"
       + (", learning rates 3e-3 to 1e-1" if s.startswith("circuit-lr") else "") for s, v in TEXT_FREE.items()},
    **{s: f"driven varactor line, drives/{p}_<model>.json: "
          + {"load": "RMS load-voltage error / amplitude", "energy": "block-energy deviation / classical maximum",
             "q_blk": "RMS charge error in the block / amplitude", "f_blk": "RMS flux error in the block / amplitude",
             "q_all": "RMS charge error on all 40 sections / amplitude"}[k] + ", mean over the signals"
       for s, (p, k) in TEXT_DRIVES.items()},
    "voltage-both": "voltage error, circuit with C(V) and L(I) nonlinear, V-I (nllc_e1) and q-phi (nllcqf) networks (vi_voltage.json)",
    "embedded-worst": "embedded ring, median over the initial conditions of the energy change over the last 10^3 of 10^4 "
                      "steps with divergence counted as infinite (embedded/*.json, T10000 drift_med)",
    "embedded-charge": "embedded ring, |total charge - initial| / initial ||q||_2 at t = 500, mean over the non-diverged "
                       "initial conditions (embedded_charge/*.json and embedded/*.json, T10000 dq)",
    "embedded-flux": "the same for the total flux (T10000 dphi)",
    "ring-third-low": "training ring of the varactor networks, charge error over the third of the test initial conditions "
                      "with the smallest sum of squared differences between adjacent sections (ring_thirds.json)",
    "ring-third-mid": "the same, middle third", "ring-third-high": "the same, largest third",
    "dc-offset": "varactor line, matched, sources off: measured dc load voltage / 320 (dq + dphi) from the block's "
                 "increments at the bias point (pulse_physics/*.json)",
    "arrival-drift": "varactor line with reflections: |mean over the pulses of the least-squares slope of the arrival "
                     "shift over block crossings 1, 3, 5, 7, 9| in ns (pulse_physics/*.json)",
}

# Driven varactor line (Sect. 3.2): varactors, C(V) = C_j0/sqrt(1 + V/V_bi), and linear inductors, in units of 1 V and C_b * 1 V
CJ0, PHI_BI, V_BIAS, L_IND = 1e-12, 0.7, 2.0, 1.27e-9
ALPHA = 0.0925  # the coefficient of u(q) = q + ALPHA q^2 in scripts/nltlv_open_20261022.py
DT = 0.05


def constants() -> dict:
    """Derived values of Table II and Sect. 3.2. Linearized limits: the energy is split equally between U and K, so
    q^2 = -1 + sqrt(1 + 2E) for U(q) = q^2/2 + q^4/4 = E/2, and u'(q) = i'(phi) = 1 + 3 q^2; leapfrog on the circuit is
    stable for dt < 2/sqrt(u'i'), and on a ring for a Courant number dt sqrt(u'i') < 1 (a linear inductor: i' = 1)."""
    cb = CJ0 / math.sqrt(1 + V_BIAS / PHI_BI)
    t0 = math.sqrt(L_IND * cb)
    du = lambda E: 1 + 3 * (-1 + math.sqrt(1 + 2 * E))  # noqa: E731
    q3 = (-1 + math.sqrt(1 + 4 * ALPHA * 3.0)) / (2 * ALPHA)  # q at 3 V
    return {"C_b_pF": cb * 1e12, "t0_ps": t0 * 1e12, "Z0_ohm": math.sqrt(L_IND / cb), "dt_ps": DT * t0 * 1e12,
            "alpha_exact": 1 / (4 * (PHI_BI + V_BIAS)), "alpha_code": ALPHA,
            "circuit_limit": 2 / du(1.0), "line_limit": 1 / du(0.5), "line_courant_max": DT * du(0.5),
            "varactor_fraction_3V": DT * math.sqrt(1 + 2 * ALPHA * q3)}


# ---- export: the per-run values and single values from the runs and the outputs

def free_rows(source: str, seeds, root: str, lrs=LD.LRS, models=None):
    ds, names, sub, tag, key = TEXT_FREE[source]
    for short, cls in names.items():
        if models is not None and short not in models:
            continue
        for size in LD.SIZES:
            hs = LD.width(cls, size, ds)
            for lr in lrs:
                for seed in seeds:
                    d = Path(root) / ds / LD.RUN_ID.format(lr=lr, seed=seed, tag=tag) / cls / f"hs{hs}"
                    v = LD.t50((d / sub if sub else d) / "ic_generalization_summary.json", key)
                    if v is not None:
                        yield (size, lr), short, seed, v


def drive_rows(directory, prefix: str, key: str, seeds, models=("LF", "SIE", "LF-U", "Res", "RK4-net", "LF-KCL")):
    """letter_data.drive_rows for load and energy; for the other keys, sqrt(key) / amplitude, mean over the signals
    that did not diverge."""
    if key in ("load", "energy"):
        yield from LD.drive_rows(directory, prefix, key, set(seeds), models)
        return
    for m in models:
        f = Path(directory) / f"{prefix}_{LD.SCRIPT_NAME.get(m, m)}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        amp = np.asarray(d["amp"], float)
        for r in d["runs"]:
            if r["seed"] not in seeds:
                continue
            ok = ~np.asarray(r["div"], bool)
            if ok.any():
                yield ("medium", r["lr"]), m, r["seed"], float((np.sqrt(np.asarray(r[key], float)) / amp)[ok].mean())


def drive_seeds(source: str):
    return SECOND if source.endswith("-2") else THIRD if source.endswith("-3") else FIRST


def ring_third_rows(path):
    """The training ring's test initial conditions in thirds of the squared differences between adjacent sections
    (expo_f + expo_q of nltlv_diag_20261022.py --mode ring); per run, the mean charge error over a third (errors of 100
    or more left out)."""
    d = json.load(open(path))
    expo = np.asarray(d["expo_f"], float) + np.asarray(d["expo_q"], float)
    lo, hi = np.quantile(expo, [1 / 3, 2 / 3])
    bins = {"ring-third-low": expo <= lo, "ring-third-mid": (expo > lo) & (expo <= hi), "ring-third-high": expo > hi}
    for r in d["runs"]:
        e = np.asarray(r["err"], float)
        for source, b in bins.items():
            ok = (e <= 100) & b
            if ok.any():
                yield source, ("medium", r["lr"]), LD.LETTER_NAME.get(r["model"], r["model"]), r["seed"], float(e[ok].mean())


def pulse_rows(directory):
    """Per network of the varactor line (seeds 0-9: LF, SIE, LF-U, Res; seeds 10-19: LF, SIE, LF-U): the dc load voltage
    with the sources off over the prediction 320 (dq0 + dphi0) of the matched line (networks whose free-running line
    diverged left out), and |mean slope| of the arrival shifts (steps) over block crossings 1, 3, 5, 7, 9 under the
    reflected pulses, times 1.27 ps per step, in ns."""
    for name, models in (("old", ("LF", "SIE", "LF-U", "Res")), ("new", ("LF", "SIE", "LF-U"))):
        for m in models:
            f = Path(directory) / f"{name}_{m}.json"
            if not f.exists():
                continue
            for r in json.load(open(f))["runs"]:
                key, seed = ("medium", r["lr"]), int(r["seed"])
                if not r["off_div_m"]:
                    yield "dc-offset", key, m, seed, r["dc_m"] / (320.0 * (r["dq0"] + r["dphi0"]))
                ok = ~np.asarray(r["div_r"], bool)
                sh = np.asarray([[np.nan if v is None else v for v in row] for row in r["shift"]], float)[ok]
                slope = abs(float(np.mean([np.polyfit([1, 3, 5, 7, 9], row, 1)[0] for row in sh]))) * 1.27e-3
                if math.isfinite(slope):
                    yield "arrival-drift", key, m, seed, slope


def text_rows(root: str, outputs: str):
    """Rows (source, stratum, model, seed, value) of results/text_runs.csv."""
    out = Path(outputs)
    have = {(s, r[1]) for s, rows in LD.read_runs().items() for r in rows}
    for source, seeds in MORE_MODELS.items():
        for key, model, seed, v in LD.rows(source, seeds, LD.SIZES, root, outputs):
            if (source, model) not in have:
                yield source, key, model, seed, v
    for key, model, seed, v in LD.vi_voltage_rows(out / "vi_voltage.json", "circuit2", set(FIRST)):
        yield "voltage-both", key, model, seed, v
    for source in ("line-32-dt0.2", "circuit-extrap", "line-extrap"):
        models = T4_MODELS + ("LF-Sep",) if source == "line-extrap" else None
        for key, model, seed, v in free_rows(source, FIRST, root, models=models):
            yield source, key, model, seed, v
    for source in ("circuit-lr", "circuit-lr-val"):
        for key, model, seed, v in free_rows(source, FIRST, root, LR_GRID, LR_MODELS):
            yield source, key, model, seed, v
    for source, (prefix, key) in TEXT_DRIVES.items():
        for k, model, seed, v in drive_rows(out / "drives", prefix, key, drive_seeds(source)):
            yield source, k, model, seed, v
    paths = sorted(glob.glob(str(out / "embedded" / "*.json")))
    for key, model, seed, v in LD.embedded_rows(paths, "T10000", "drift_med", set(FIRST)):
        yield "embedded-worst", key, model, seed, v
    charge = paths + sorted(glob.glob(str(out / "embedded_charge" / "*.json")))
    for source, k in (("embedded-charge", "dq"), ("embedded-flux", "dphi")):
        for key, model, seed, v in LD.embedded_rows(charge, "T10000", k, set(FIRST)):
            yield source, key, model, seed, v
    yield from ring_third_rows(out / "ring_thirds.json")
    yield from pulse_rows(out / "pulse_physics")


def value_range(x) -> list:
    x = np.asarray(x, float)
    return [float(x.min()), float(x.max())]


def signals(outputs: str) -> dict:
    """Amplitudes and times of the driven line's 64 signals per drive (the display signal of pulses.json and
    reflected.json left out), and the durations, in V, ps, ns, GHz, and Gb/s."""
    out, t0 = Path(outputs), constants()["t0_ps"]
    p, r = json.load(open(out / "pulses.json")), json.load(open(out / "reflected.json"))
    z0 = constants()["Z0_ohm"]
    res = {"pulses": {"amplitude_V": value_range(p["A"][:64]), "edge_ps": value_range(np.asarray(p["tr"][:64]) * t0),
                      "duration_ns": p["steps"] * p["dt"] * t0 * 1e-3},
           "reflected": {"amplitude_V": value_range(r["a"][:64]), "edge_ps": value_range(np.asarray(r["tr"][:64]) * t0),
                         "source_ohm": r["rs"] * z0, "load_ohm": r["rl"] * z0,
                         "duration_ns": r["steps"] * r["dt"] * t0 * 1e-3}}
    for prefix in ("train", "prbs", "cw", "train_new", "cw_new"):
        d = json.load(open(out / "drives" / f"{prefix}_LF.json"))
        x = {"amplitude_V": value_range(d["amp"]), "duration_ns": d["steps"] * DT * t0 * 1e-3}
        if "period" in d:
            x["period_ns"] = value_range(np.asarray(d["period"]) * t0 * 1e-3)
        if "tb" in d:
            x["bit_rate_Gbps"] = value_range(1e3 / (np.asarray(d["tb"]) * t0))
        if "f" in d:
            x["frequency_GHz"] = value_range(np.asarray(d["f"]) / t0 * 1e3)
        res[prefix] = x
    return res


def rise_times(outputs: str) -> dict:
    """Median 10-90% rise times over the 64 matched pulses: of the incident raised-cosine edge, (acos(-0.8) -
    acos(0.8)) / pi times the edge time, and of the all-classical load voltage (edges_classical of
    nltlv_open_20261022.py --reference), in ps."""
    ref, t0 = json.load(open(Path(outputs) / "pulses_reference.json")), constants()["t0_ps"]
    edges = np.asarray(ref["edges_classical"], float)
    k = (math.acos(-0.8) - math.acos(0.8)) / math.pi
    return {"incident_ps": float(np.median(np.asarray(ref["tr"][:64]) * k)) * t0,
            "classical_load_ps": float(np.median(edges[:, 2] - edges[:, 0])) * t0, "pulses": len(edges)}


def crossings(quiet: float = 0.05, merge: int = 300) -> dict:
    """Block crossings of the reflected pulses in the all-classical line: per pulse, the first two times at which the
    load voltage starts to exceed 5% of the amplitude (spans less than 300 steps apart merged); crossings = 1 +
    (end - first arrival) / (half the interval between the arrivals), with the medians over the 64 pulses."""
    import torch
    import nltlv_reflect_20261022 as RF
    a, tr = RF.pulses()
    with torch.no_grad():
        uc = RF.run(None, "stag", torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(tr, dtype=torch.float32),
                    record=True)["uLc"][:-1]
    first, gap = [], []
    for b in range(len(uc)):
        idx = np.flatnonzero(np.abs(uc[b]) > quiet * a[b])
        starts = np.r_[idx[0], idx[np.flatnonzero(np.diff(idx) > merge) + 1]]
        first.append(starts[0])
        gap.append(starts[1] - starts[0])
    f, g = float(np.median(first)), float(np.median(gap))
    return {"first_arrival_t0": f * DT, "arrival_interval_t0": g * DT, "steps": RF.STEPS,
            "crossings": 1 + (RF.STEPS - f) / (g / 2)}


def diverged_drive_runs(outputs: str) -> dict:
    """Runs of the driven line in which any signal diverged: [file, model, lr, seed, diverged, signals]."""
    out, found, files = Path(outputs), [], 0
    for f in sorted(glob.glob(str(out / "drives" / "*.json"))):
        files += 1
        for r in json.load(open(f))["runs"]:
            n = int(np.sum(r["div"]))
            if n:
                found.append([Path(f).name, r["model"], r["lr"], r["seed"], n, len(r["div"])])
    for name in ("pulses.json", "reflected.json"):
        files += 1
        for r in json.load(open(out / name))["runs"]:
            if r["n_div"]:
                found.append([name, r["model"], r["lr"], r["seed"], int(r["n_div"]), 64])
    return {"files": files, "runs": found}


# free-run sources at dt = 0.05 whose rollout errors the letter reports (those of Fig. 5 at coarser steps are in
# results/fig5_runs.csv)
FREE_AT_005 = ("circuit", "line", "line-32", "leaky", "leaky-32", "circuit-rollout", "line-rollout", "circuit-noise0.1",
               "line-noise0.1", "circuit-noise1", "line-noise1", "ring", "vi-cap", "vi-both", "vi-ind", "vi-both-e1",
               "vi-ind-cpu")


def free_divergence(root: str) -> dict:
    """{source: {model: [runs, runs with any diverged test initial condition, largest number diverged]}}."""
    res = {}
    for source in FREE_AT_005:
        cnt = {}
        for _, model, _, v in LD.free_rows(source, range(50), LD.SIZES, root, None, DIVERGED):
            c = cnt.setdefault(model, [0, 0, 0])
            c[0] += 1
            c[1] += int(v > 0)
            c[2] = max(c[2], int(v))
        res[source] = cnt
    return res


def charge_check(outputs: str) -> dict:
    """The embedded_charge/ runs against the same runs in embedded/: largest relative difference of T10000 drift."""
    out = Path(outputs)
    a = {(k, m, s): v for k, m, s, v in LD.embedded_rows(sorted(glob.glob(str(out / "embedded" / "*.json"))))}
    b = {(k, m, s): v for k, m, s, v in LD.embedded_rows(sorted(glob.glob(str(out / "embedded_charge" / "*.json"))))}
    common = [k for k in b if k in a]
    worst = max((abs(b[k] - a[k]) / max(abs(a[k]), 1e-30) for k in common), default=None)
    return {"runs": len(b), "matched": len(common), "max_rel_diff": worst}


def empty_signals(outputs: str) -> dict:
    """Signals of each first-set drive whose block energy stays zero in the all-classical line (e_scale), which the
    block-energy deviation leaves out; the same in every run."""
    res = {}
    for prefix in ("open", "reflect", "train", "prbs", "cw"):
        runs = json.load(open(Path(outputs) / "drives" / f"{prefix}_LF.json"))["runs"]
        zero = {tuple(np.flatnonzero(np.asarray(r["e_scale"], float) <= 1e-9)) for r in runs}
        assert len(zero) == 1, prefix
        res[prefix] = len(zero.pop())
    return res


def text_values(root: str, outputs: str) -> dict:
    out = Path(outputs)
    steps = json.load(open(out / "classical_steps.json"))
    line = json.load(open(out / "classical_line.json"))
    div = json.loads(open(out / "classical_div.txt").read().strip().splitlines()[-1])
    return {
        "classical": {"circuit_extrap": steps["nlqf3_v1_dtd2"]["t50s"][EXTRAP],
                      "line_extrap": line["nltl8_v1_dtd2"]["t50s"][EXTRAP],
                      "vi_voltage": json.load(open(out / "vi_voltage.json"))["classical"],
                      "centered": json.load(open(out / "classical_centered.json")),
                      "diverged": div},
        "signals": signals(outputs), "rise_times": rise_times(outputs), "reflected_crossings": crossings(),
        "empty_energy_signals": empty_signals(outputs), "diverged_drive_runs": diverged_drive_runs(outputs),
        "free_run_divergence": free_divergence(root), "embedded_charge_check": charge_check(outputs),
    }


def export(root: str, outputs: str) -> int:
    items = list(text_rows(root, outputs))
    seen = set()
    for s, k, m, sd, _ in items:
        assert (s, k, m, sd) not in seen, f"duplicate run {s} {k} {m} {sd}"
        seen.add((s, k, m, sd))
    LD.write_runs(TEXT_RUNS_CSV, items)
    json.dump(text_values(root, outputs), open(TEXT_VALUES, "w"), indent=1)
    return len(items)


# ---- printing: the numbers from the saved values

class Runs:
    """The rows of results/letter_runs.csv and results/text_runs.csv together."""

    def __init__(self, paths=(LD.RUNS_CSV, TEXT_RUNS_CSV)):
        self.rows = {}
        for p in paths:
            for source, rows in LD.read_runs(p).items():
                self.rows.setdefault(source, []).extend(rows)

    def get(self, source: str, seeds=FIRST, sizes=LD.SIZES) -> dict:
        seeds = set(seeds)
        return LD.nest(r for r in self.rows.get(source, []) if r[2] in seeds and r[0][0] in sizes)


def fig5_groups(path=FIG5_CSV):
    """{(panel, dt): {stratum: {model: {seed: error}}}} and {(panel, dt): classical error} of results/fig5_runs.csv."""
    g, ref = {}, {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        key = (r["panel"], float(r["dt"]))
        if r["model"] == "classical":
            ref[key] = float(r["error"])
        else:
            g.setdefault(key, {}).setdefault((r["size"], r["lr"]), {}).setdefault(r["model"], {})[int(r["seed"])] = float(r["error"])
    return g, ref


def est(groups: dict, a: str, b: str, name: str):
    return LS.estimate(LS.log_ratios(groups, a, b), name)


def fr(r) -> str:
    return "--" if r is None else f"{r['ratio']:.3g} [{r['lo']:.3g}, {r['hi']:.3g}] ({r['runs']} pairs)"


def median(groups: dict, model: str):
    v = [x for cell in groups.values() for x in cell.get(model, {}).values()]
    return float(np.median(v)) if v else None


def only(groups: dict, size: str) -> dict:
    return {k: v for k, v in groups.items() if k[0] == size}


def span(values) -> str:
    v = [x for x in values if x is not None]
    return f"{min(v):.3g}-{max(v):.3g}" if v else "--"


def read_plan() -> list:
    return list(csv.DictReader(open(PLAN, encoding="utf-8")))


def planned(leave_out: str = "both") -> dict:
    """The planned tests computed by scripts/letter_tables.py from results/letter_runs.csv."""
    import letter_tables as LT
    old, LS.LEAVE_OUT = LS.LEAVE_OUT, leave_out
    try:
        return LT.run_tests(LT.SavedData())
    finally:
        LS.LEAVE_OUT = old


STEP_SOURCE = {("circuit", 0.05): "circuit", ("circuit", 0.1): "circuit-dt0.1", ("circuit", 0.2): "circuit-dt0.2",
               ("circuit", 0.4): "circuit-dt0.4", ("line", 0.05): "line", ("line", 0.1): "line-dt0.1",
               ("line", 0.2): "line-dt0.2", ("line", 0.4): "line-dt0.4"}
OFF_CENTER_SOURCES = {"circuit", "line", "line-32", "leaky", "leaky-32", "circuit-rollout", "line-rollout",
                      "circuit-noise0.1", "line-noise0.1", "circuit-noise1", "line-noise1", "ring", "embedded"}


def sect3(R: Runs, V: dict, plan: list, p) -> None:
    c = constants()
    p("## Sect. 3, Tables II and III\n")
    p(f"- Time step 0.05 'about 8% of the circuit's stability limit linearized at the highest energy': limit "
      f"{c['circuit_limit']:.4g}, 0.05 / limit = {DT / c['circuit_limit']:.4f}; 'Courant number 0.05 for small signals "
      f"to 0.11': 0.05 and {c['line_courant_max']:.4f} (line limit {c['line_limit']:.4g})")
    d = V["classical"]["diverged"]
    p(f"- Classical scheme, true laws, initial conditions of 512 diverged (t = 50, t = 500): "
      + "; ".join(f"{k} {v['t50']}, {v['t500']}" for k, v in d.items()))
    val, test = R.get("circuit-lr-val"), R.get("circuit-lr")
    best = {}
    for m in LR_MODELS:
        for size in LD.SIZES:
            meds = {lr: median({k: x for k, x in val.items() if k == (size, lr)}, m) for lr in LR_GRID
                    if len(val.get((size, lr), {}).get(m, {})) >= 8}
            best[m, size] = min(meds, key=meds.get)
    p("- Learning rate chosen per model and size by the validation median (runs >= 8): "
      + "; ".join(f"{m} {s} {lr}" for (m, s), lr in best.items()))
    sel = {}
    for (m, size), lr in best.items():
        sel.setdefault((size, "selected"), {})[m] = test.get((size, lr), {}).get(m, {})
    fixed = R.get("circuit")
    for m in ("Res", "RK4-net", "SIE-M"):
        p(f"  - {m}/LF with the chosen rates {fr(est(sel, m, 'LF', f'text/lr/{m}'))}; with 1e-2 and 3e-2 "
          f"{fr(est(fixed, m, 'LF', f'text/lr-fixed/{m}'))}")
    fd = V["free_run_divergence"]
    bad = [f"{s} {m}: {n[1]} of {n[0]} runs, up to {n[2]}" for s in FREE_AT_005 for m, n in fd[s].items() if n[1]]
    p(f"- Free runs at dt = 0.05 with any diverged test initial condition ({len(FREE_AT_005)} sources, "
      f"{sum(n[0] for s in FREE_AT_005 for n in fd[s].values())} runs): " + ("; ".join(bad) if bad else "none"))
    fam = {}
    for r in plan:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    p(f"- Planned tests: {len(plan)} in {len(fam)} families of {min(fam.values())}-{max(fam.values())}; "
      f"repository only {sum(r['letter'] == 'repository only' for r in plan)}\n")
    p(f"- Varactor line: C_b = {c['C_b_pF']:.4g} pF, t0 = {c['t0_ps']:.4g} ps, Z0 = {c['Z0_ohm']:.4g} Ohm, "
      f"dt = {c['dt_ps']:.4g} ps; u(q) = q + alpha q^2 with alpha = 1/(4 (phi + V_bias)) = {c['alpha_exact']:.6g} "
      f"({c['alpha_code']} in the code); dt at 3 V: {c['varactor_fraction_3V']:.4f} of the stability limit")
    s = V["signals"]
    p(f"- Matched pulses: {s['pulses']['amplitude_V'][0]:.4g}-{s['pulses']['amplitude_V'][1]:.4g} V, edges "
      f"{s['pulses']['edge_ps'][0]:.4g}-{s['pulses']['edge_ps'][1]:.4g} ps, {s['pulses']['duration_ns']:.4g} ns; median "
      f"10-90% rise {V['rise_times']['incident_ps']:.4g} ps incident, {V['rise_times']['classical_load_ps']:.4g} ps at "
      f"the all-classical load")
    x = V["reflected_crossings"]
    p(f"- Reflecting terminations: {s['reflected']['source_ohm']:.4g} and {s['reflected']['load_ohm']:.4g} Ohm, "
      f"{s['reflected']['amplitude_V'][0]:.4g}-{s['reflected']['amplitude_V'][1]:.4g} V, "
      f"{s['reflected']['duration_ns']:.4g} ns; first arrival {x['first_arrival_t0']:.4g} t0, arrivals "
      f"{x['arrival_interval_t0']:.4g} t0 apart: {x['crossings']:.3g} block crossings")
    for k, lab in (("train", "pulse trains"), ("prbs", "bit streams"), ("cw", "sinusoids"),
                   ("train_new", "pulse trains, 2nd set"), ("cw_new", "sinusoids, 2nd set")):
        y = s[k]
        rest = "; ".join(f"{n} {y[n][0]:.4g}-{y[n][1]:.4g}" for n in ("period_ns", "bit_rate_Gbps", "frequency_GHz") if n in y)
        p(f"- {lab.capitalize()}: {y['amplitude_V'][0]:.4g}-{y['amplitude_V'][1]:.4g} V; {rest}; {y['duration_ns']:.4g} ns")
    p("- Signals left out of the block-energy deviation, with no block energy in the all-classical line: "
      + ", ".join(f"{k} {n}" for k, n in V["empty_energy_signals"].items()))
    p("")


def sect41(R: Runs, tests: dict, p) -> None:
    p("## Sect. 4.1\n")
    c, ln, l32 = R.get("circuit"), R.get("line"), R.get("line-32")
    p(f"- 32 sections: RK4-net/LF {fr(est(l32, 'RK4-net', 'LF', 'text/l32/rk4'))}; LF/LF-KCL "
      f"{fr(est(l32, 'LF', 'LF-KCL', 'text/l32/kcl'))}")
    p(f"- Res2/Res: circuit {fr(est(c, 'Res2', 'Res', 'text/res2/c'))}, line {fr(est(ln, 'Res2', 'Res', 'text/res2/l'))}")
    off = [(k, r) for k, r in tests.items() if r["model_b"] == "LF" and r["model_a"] in ("SIE", "SIE-M", "LF-U")
           and r["source"] in OFF_CENTER_SOURCES]
    kept = [(k, r) for k, r in off if r["ratio"] is not None]
    p(f"- Planned off-center ablation / LF comparisons in charge and flux, free runs and the embedded ring: {len(off)}, "
      f"{len(kept)} with a stratum left, {sum(r['dagger'] for _, r in kept)} with a dagger, "
      f"{sum(r['model_a'] == 'SIE-M' for _, r in kept)} against SIE-M: "
      + ", ".join(f"{k[0]}/{k[1]} {r['model_a']}/LF {r['ratio']:.3g}{'†' if r['dagger'] else ''}" for k, r in kept))
    for lab, seeds in (("first set", FIRST), ("second set", SECOND)):
        cells = []
        for src in ("circuit", "line"):
            g = R.get(src, seeds)
            for m in ("SIE", "LF-U"):
                cells.append(f"{src} {m}/LF {fr(est(g, m, 'LF', f'text/{lab}/{src}/{m}'))}")
                if seeds == SECOND:
                    cells[-1] += " (" + ", ".join(f"{z} {est(only(g, z), m, 'LF', f'text/{lab}/{src}/{m}/{z}')['ratio']:.3g}"
                                                  for z in LD.SIZES) + ")"
        p(f"- {lab.capitalize()}: " + "; ".join(cells))
    rows, narrowed = [], 0
    for src, g in (("circuit", c), ("line N=8", ln), ("line N=32", l32)):
        for m in ("Res", "RK4-net", "SIE", "LF-U", "SIE-M"):
            s, mm = (est(only(g, z), m, "LF", f"text/margin/{src}/{m}/{z}") for z in LD.SIZES)
            narrowed += mm["ratio"] < s["ratio"]
            rows.append((f"{src} {m}", s["ratio"], mm["ratio"]))
    p(f"- X/LF at about 700 and 2400 parameters: narrowed in {narrowed} of {len(rows)}; smallest at 2400 "
      f"{min(r[2] for r in rows):.4g}; " + "; ".join(f"{n} {a:.3g} -> {b:.3g}" for n, a, b in rows))
    p(f"- Line N=8 medians at 2400 parameters: " + ", ".join(f"{m} {median(only(ln, 'medium'), m):.3g}" for m in T4_MODELS)
      + "; at 700: " + ", ".join(f"{m} {median(only(ln, 'small'), m):.3g}" for m in T4_MODELS))
    p(f"- LF-Sep/LF in the line (1 / line L6): {1 / est(ln, 'LF', 'LF-Sep', 'text/lfsep')['ratio']:.3g}\n")


def sect42(R: Runs, p) -> None:
    p("## Sect. 4.2 (and README.md)\n")
    div = R.get("embedded-div")
    cells = []
    for m in ("Res", "RK4-net", "SIE", "LF-U", "SIE-M"):
        pairs = [(cell["LF"][s], cell[m][s]) for cell in div.values() if "LF" in cell and m in cell
                 for s in cell["LF"] if s in cell[m]]
        cells.append(f"{m} {sum(a < b for a, b in pairs)} of {len(pairs)}")
    p("- Pairs in which LF lost fewer initial conditions to divergence: " + ", ".join(cells))
    q, f = R.get("embedded-charge"), R.get("embedded-flux")
    p("- Total charge and flux changed by t = 500 (medians, % of the initial norms): "
      + "; ".join(f"{m} {100 * median(q, m):.4g} and {100 * median(f, m):.4g}" for m in ("LF", "Res", "RK4-net", "LF-KCL",
                                                                                      "SIE-M", "LF-Sep", "SIE", "LF-U", "Res2") if median(q, m) is not None))
    w = R.get("embedded-worst")
    p("- Energy change with divergence counted as the worst (median over initial conditions), X/LF: "
      + "; ".join(f"{m} {fr(est(w, m, 'LF', f'text/worst/{m}'))}" for m in ("Res", "RK4-net", "SIE", "SIE-M", "LF-U")))
    p("")


DRIVE_ROWS = [("Pulses, matched", FIRST, ["pulses"]), ("Pulses, reflected", FIRST, ["reflected"]),
              ("Pulse trains", FIRST, ["trains"]), ("Bit streams", FIRST, ["bits"]), ("Sinusoids", FIRST, ["sinusoids"]),
              ("Pulses, 2nd set", SECOND, ["pulses-2", "reflected-2", "trains-2"]),
              ("Pulses, 3rd set", THIRD, ["pulses-3", "reflected-3", "trains-3"]),
              ("Sinusoids, 2nd set", SECOND, ["sinusoids-2"])]


def variant(source: str, tag: str) -> str:
    """'pulses-2' -> 'pulses-<tag>-2'; 'pulses' -> 'pulses-<tag>'."""
    name, _, n = source.partition("-")
    return f"{name}-{tag}" + (f"-{n}" if n else "")


def pooled(R: Runs, sources, seeds, a: str, name: str):
    parts = [LS.log_ratios(R.get(s, seeds), a, "LF") for s in sources]
    return LS.estimate(parts[0] if len(parts) == 1 else LS.pooled(parts), name)


def sect43(R: Runs, V: dict, p) -> None:
    p("## Sect. 4.3 (and README.md)\n")
    meds = []
    table_v = ("pulses", "reflected", "trains", "bits", "sinusoids", "sinusoids-2")
    for _, seeds, sources in DRIVE_ROWS:
        meds += [(s, median(R.get(s, seeds), "LF")) for s in sources]
    p("- LF's median load-voltage error (% of the amplitude): " + ", ".join(f"{s} {100 * v:.4g}" for s, v in meds)
      + f"; range {100 * min(v for _, v in meds):.3g}-{100 * max(v for _, v in meds):.3g}, without the 3rd set "
        f"{100 * min(v for s, v in meds if not s.endswith('-3')):.3g}-{100 * max(v for s, v in meds if not s.endswith('-3')):.3g}, "
        f"in the rows of Table V {100 * min(v for s, v in meds if s in table_v):.3g}-{100 * max(v for s, v in meds if s in table_v):.3g}")
    dv = V["diverged_drive_runs"]
    p(f"- Runs with diverged signals in the {dv['files']} files of the driven line: "
      + ("; ".join(f"{f} {m} {lr} seed {s}: {n} of {k}" for f, m, lr, s, n, k in dv["runs"]) or "none"))
    e = [(s, m, est(R.get(f"{s}-energy"), m, "LF", f"text/energy/{s}/{m}"))
         for s in ("pulses", "reflected", "trains", "bits", "sinusoids") for m in ("Res", "RK4-net")]
    p("- Block energy, first set (README.md): " + "; ".join(f"{s} {m}/LF {r['ratio']:.4g}" for s, m, r in e)
      + f"; range {span(r['ratio'] for _, _, r in e)}")
    k = [("ring", "charge"), *[(s, "load") for s in ("pulses", "reflected", "trains", "bits", "sinusoids")],
         *[(f"{s}-energy", "energy") for s in ("pulses", "reflected", "trains", "bits", "sinusoids")]]
    ks = [(s, w, est(R.get(s), "LF-KCL", "LF", f"text/kcl/{s}")) for s, w in k]
    p("- LF-KCL/LF: " + "; ".join(f"{s} {r['ratio']:.3g}" for s, _, r in ks if r)
      + f"; load voltage and the ring {span(r['ratio'] for _, w, r in ks if r and w != 'energy')}, block energy "
        f"{span(r['ratio'] for _, w, r in ks if r and w == 'energy')}")
    p("- Uniform-state increments subtracted (README.md; each drive as in README.md; load voltage, then block energy):")
    for label, seeds, sources in DRIVE_ROWS:
        cells = []
        for measure, tag in (("load", "rc"), ("energy", "rc-energy")):
            for a in ("SIE", "LF-U", "Res", "RK4-net"):
                r = pooled(R, [variant(s, tag) for s in sources], seeds, a, f"text/{tag}/{label}/{a}")
                if r is not None:
                    cells.append(f"{measure} {a}/LF {fr(r)}")
        p(f"  - {label}: " + "; ".join(cells))
    d = R.get("arrival-drift", range(20))
    p(f"- Arrival-time drift per block crossing (40 networks per model): SIE/LF {fr(est(d, 'SIE', 'LF', 'text/drift/SIE'))}; "
      f"LF-U/LF {fr(est(d, 'LF-U', 'LF', 'text/drift/LF-U'))}")
    dc = [x for cell in R.get("dc-offset", range(20)).values() for m in cell for x in cell[m].values()]
    p(f"- Measured / predicted dc load voltage, matched line: median {np.median(dc):.3g} over {len(dc)} networks "
      f"(10-90%: {np.percentile(dc, 10):.3g}-{np.percentile(dc, 90):.3g})")
    for src, lab in (("busy", "busy signals"), ("busy-low", "busy signals, low frequency"), ("burst", "bursts")):
        g = R.get(src)
        p(f"- {lab.capitalize()}, load voltage: SIE/LF {fr(est(g, 'SIE', 'LF', f'text/{src}/SIE'))}; "
          f"LF-U/LF {fr(est(g, 'LF-U', 'LF', f'text/{src}/LF-U'))}")
    for t in ("low", "mid", "high"):
        g = R.get(f"ring-third-{t}")
        p(f"- Training ring, {t} third: SIE/LF {fr(est(g, 'SIE', 'LF', f'text/ring/{t}/SIE'))}; "
          f"LF-U/LF {fr(est(g, 'LF-U', 'LF', f'text/ring/{t}/LF-U'))}")
    for drive in ("pulses", "reflected"):
        cells = []
        for m, lab in (("charge-block", "charge in the block"), ("flux-block", "flux in the block"),
                       ("charge-all", "charge on all sections"), ("energy", "block energy")):
            g = R.get(f"{drive}-{m}")
            cells.append(f"{lab}: SIE/LF {fr(est(g, 'SIE', 'LF', f'text/{drive}/{m}/SIE'))}, "
                         f"LF-U/LF {fr(est(g, 'LF-U', 'LF', f'text/{drive}/{m}/LF-U'))}")
        p(f"- {drive.capitalize()}, other measures; " + "; ".join(cells))
    p("")


def sect44(R: Runs, V: dict, p) -> None:
    p("## Sect. 4.4, Table VI, Fig. 5, and Sect. 5\n")
    g5, ref = fig5_groups()
    at = {}
    for key, src in STEP_SOURCE.items():
        g = {k: dict(v) for k, v in g5.get(key, {}).items()}
        for k, cell in R.get(src).items():
            for m, x in cell.items():
                g.setdefault(k, {}).setdefault(m, x)
        at[key] = g
    diff = max(abs(g5[key][k]["LF"][s] - x) for key, src in STEP_SOURCE.items() for k, cell in R.get(src).items()
               if "LF" in cell for s, x in cell["LF"].items() if key in g5 and s in g5[key].get(k, {}).get("LF", {}))
    p(f"- results/fig5_runs.csv and the runs files agree for LF (largest difference {diff:.1e})")
    for dt in (0.1, 0.2, 0.4):
        g = at["circuit", dt]
        lf = median(g, "LF")
        p(f"- Circuit, dt = {dt}: LF-KCL/LF {fr(est(g, 'LF-KCL', 'LF', f'text/fig5/kcl/{dt}'))}, ratio of medians "
          f"{median(g, 'LF-KCL') / lf:.4g}; classical / LF's median {ref['circuit', dt] / lf:.4g}")
    p(f"- Line, dt = 0.2: LF median {median(at['line', 0.2], 'LF'):.4g}, classical {ref['line', 0.2]:.4g}; "
      f"LF-KCL/LF {fr(est(at['line', 0.2], 'LF-KCL', 'LF', 'text/fig5/kcl/line'))}")
    c = constants()
    p(f"- dt = 0.4 in the circuit: {0.4 / c['circuit_limit']:.3f} of the linearized limit; dt = 0.2 in the line: "
      f"{0.2 / c['line_limit']:.3f}")
    for panel, coarse in (("circuit", 0.4), ("line", 0.2)):
        cells = []
        for m in ("Res", "LF-U", "SIE-M", "SIE"):
            rr = LS.estimate(LS.ratio_of_ratios(at[panel, coarse], at[panel, 0.05], m, "LF"), f"text/shrink/{panel}/{m}")
            cells.append(f"{m}/LF {est(at[panel, 0.05], m, 'LF', f'text/fine/{panel}/{m}')['ratio']:.3g} -> "
                         f"{est(at[panel, coarse], m, 'LF', f'text/coarse/{panel}/{m}')['ratio']:.3g} (fine / coarse "
                         f"{1 / rr['ratio']:.3g})")
        meds = ", ".join(f"{m} {median(at[panel, coarse], m):.3g}" for m in ("LF", "Res", "RK4-net", "LF-U", "SIE", "SIE-M"))
        p(f"- {panel.capitalize()}, dt = 0.05 -> {coarse}: " + "; ".join(cells) + f"; medians at {coarse}: {meds}")
    p("- Line, dt = 0.4, medians: " + ", ".join(f"{m} {median(at['line', 0.4], m):.3g}" for m in ("LF", "Res", "RK4-net", "LF-U", "SIE", "SIE-M", "Res2", "LF-Sep")))
    l32 = R.get("line-32-dt0.2")
    p(f"- 32 sections, dt = 0.2: RK4-net/LF {fr(est(l32, 'RK4-net', 'LF', 'text/l32dt/rk4'))}")
    cells = []
    for src in ("circuit-dt0.4", "line-dt0.2", "line-32-dt0.2", "circuit", "line", "line-32"):
        g = {}
        for (size, lr), cell in R.get(src).items():
            m = {"small": "RK4-net", "medium": "LF"}[size]
            if m in cell:
                g.setdefault(("cost", lr), {})[m] = cell[m]
        cells.append(f"{src} {fr(est(g, 'RK4-net', 'LF', f'text/cost/{src}'))}")
    p("- RK4-net with about 700 parameters / LF with about 2400 (matched by seed and learning rate): " + "; ".join(cells))
    for src in ("circuit-noise1", "line-noise1", "circuit-extrap", "line-extrap"):
        g = R.get(src)
        ms = sorted({m for cell in g.values() for m in cell})
        p(f"- {src} medians: " + ", ".join(f"{m} {median(g, m):.4g}" for m in ms))
    cl = V["classical"]
    p(f"- Classical, extrapolation: circuit {cl['circuit_extrap']:.4g}, line {cl['line_extrap']:.4g}")
    vi = {k: R.get(f"voltage-{k}") for k in ("cubic", "ind", "cap", "both")}
    p("- V-I / charge-flux training, voltage error: " + "; ".join(
        f"{k} LF {fr(est(g, 'LF (V-I)', 'LF (q-phi)', f'text/vi/LF/{k}'))}, SIE {fr(est(g, 'SIE (V-I)', 'SIE (q-phi)', f'text/vi/SIE/{k}'))}"
        for k, g in vi.items()))
    p("- SIE (V-I) / LF (V-I): " + "; ".join(
        f"{k} ratio of medians {median(g, 'SIE (V-I)') / median(g, 'LF (V-I)'):.4g}, matched "
        f"{est(g, 'SIE (V-I)', 'LF (V-I)', f'text/vi/both/{k}')['ratio']:.4g}" for k, g in vi.items()))
    cv = cl["centered"]["cubic"]
    p(f"- Classical V-I update, cubic circuit: {cl['vi_voltage']['cubic']['vi_in_V']:.4g} (vi_voltage.json), "
      f"{cv['vi_lf'][0]:.4g}, and {cv['vi_centered'][0]:.4g} with iteration (classical_centered_20261022.py)")
    cells = []
    for key in sorted(at):
        g = at[key]
        if median(g, "LF") >= LS.FAIL:
            cells.append(f"{key[0]} {key[1]}: LF failed")
            continue
        cells.append(f"{key[0]} {key[1]}: " + ", ".join(f"{m}/LF {est(g, m, 'LF', f'text/sect5/{key}/{m}')['ratio']:.3g}"
                                                         for m in ("LF-U", "SIE")))
    p("- Off-center ablations by time step (Sect. 5): " + "; ".join(cells))
    p("")


def readme(R: Runs, tests: dict, p) -> None:
    p("## README.md\n")
    ln = R.get("line")
    p(f"- RK4-KCL/LF-KCL in the line: {fr(est(ln, 'RK4-KCL', 'LF-KCL', 'text/rk4kcl'))}")
    with_p = {k: r for k, r in tests.items() if "ordering" not in r and r["ratio"] is not None}
    keys = list(with_p)
    hr = LS.holm([with_p[k]["p_runs"] for k in keys])
    hs = LS.holm([with_p[k]["p_seeds"] for k in keys])
    dag = [i for i, k in enumerate(keys) if with_p[k]["dagger"]]
    lost = [keys[i] for i in dag if hr[i] >= LS.ALPHA]
    p(f"- Holm over all {len(keys)} planned tests with p-values ({len(tests) - len(keys)} without): of the {len(dag)} "
      f"with a dagger, {sum(hr[i] < LS.ALPHA for i in dag)} keep p < 0.05 over runs and {sum(hs[i] < LS.ALPHA for i in dag)} "
      f"over seed means; losing over runs: " + ", ".join(f"{f}/{t}" for f, t in lost))
    anyr = planned("any")
    h6a = anyr[("tan-circuit", "H6a")]
    p(f"- --leave-out any: H6a {fr(h6a) if h6a['ratio'] is not None else 'no stratum left'}, Holm p "
      f"{h6a['p_runs_holm']:.2g} and {h6a['p_seeds_holm']:.2g}; no stratum left in "
      + ", ".join(f"{f}/{t}" for (f, t), r in anyr.items() if "ordering" not in r and r["ratio"] is None
                  and tests[(f, t)]["ratio"] is not None)
      + "; daggers lost: " + (", ".join(f"{f}/{t}" for (f, t), r in anyr.items() if "ordering" not in r
                                        and tests[(f, t)].get("dagger") and not r.get("dagger")) or "none")
      + f"; E1 {anyr[('embedded', 'E1')]['ratio']:.3g} (both: {tests[('embedded', 'E1')]['ratio']:.3g})")
    p("")


def report() -> str:
    R = Runs()
    V = json.load(open(TEXT_VALUES))
    tests = planned()
    lines = []
    p = lines.append
    p("# Numbers in the letter's text and README.md\n")
    p("Computed by `python scripts/letter_text.py` from results/letter_runs.csv, results/text_runs.csv, "
      "results/text_values.json, results/fig5_runs.csv, and docs/planned_tests.csv. A/B [95% seed interval] "
      "(pairs) as in scripts/letter_stats.py.\n")
    sect3(R, V, read_plan(), p)
    sect41(R, tests, p)
    sect42(R, p)
    sect43(R, V, p)
    sect44(R, V, p)
    readme(R, tests, p)
    p("## Sources of results/text_runs.csv\n")
    p("Further models of " + ", ".join(MORE_MODELS) + " (scripts/letter_data.py), and:\n")
    for s, text in TEXT_SOURCES.items():
        p(f"- {s}: {text}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", choices=["print", "export"], default="print")
    ap.add_argument("--runs-root", default=LD.RUNS_ROOT)
    ap.add_argument("--outputs", default=LD.OUTPUTS)
    args = ap.parse_args()
    if args.what == "export":
        n = export(args.runs_root, args.outputs)
        print(f"wrote {n} runs to {TEXT_RUNS_CSV} and the single values to {TEXT_VALUES}")
        return 0
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
