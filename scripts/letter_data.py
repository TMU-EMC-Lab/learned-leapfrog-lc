"""Per-run values of the letter's comparisons, read from the outputs of the training and evaluation scripts.

Every loader returns {stratum: {model: {seed: value}}} with strata (size, learning rate) and the letter's model names
(RK4-net, LF-KCL, ...). A source is one set of runs and one measure; FREE, DRIVES, WAVEFORMS, and OTHER list them, and
docs/planned_tests.csv names the source of each planned test. Values are errors (smaller is better); a non-finite error
or one of 1000 or more is set to 1000, as in the evaluation.

results/letter_runs.csv keeps the values that the planned tests and Tables IV and V use, one row per run and source
(columns source, size, lr, model, seed, value), and results/letter_classical.csv the classical scheme's values in
Table IV; `letter_tables.py export` writes both from the outputs below, and from_runs() reads them back as load() does.

Free runs are read from the run folders, <runs root>/<dataset>/<run id>/<class>/hs<width>/[<subfolder>/]
ic_generalization_summary.json, row t50s: the relative RMS error of the first variable (charge, voltage in V-I
variables, or position in the systems other than LC circuits) up to t = 50 averaged over the test initial conditions
(test_interp_x_rmse_rel_mean). The other sources are read from the outputs of the scripts in scripts/, collected in one
folder (OUTPUTS below; README.md gives the commands that write them):

  embedded/*.json          hybrid_line_20261021.py --block 8 --steps 10000            (any split of models and seeds)
  embedded_references.json hybrid_line_20261021.py --block 8 --steps 10000 --references
                           (both with --horizons 100 200 500 1000 2000 5000 10000 for Fig. 4, letter_figures.py)
  leaky_embedded/*.json    hybrid_line_20261021.py --train-ds nltlg8_v1_dtd2 --test-ds nltlg32_v1_dtd2 --loss-g 0.03
                           --block 8 --steps 1000                                     (any split of models and seeds)
  vi_voltage.json          vi_voltage_error.py
  pulses.json              nltlv_open_20261022.py
  reflected.json           nltlv_reflect_20261022.py
  drives/<drive>_<model>.json  nltlv_diag_20261022.py (and nltlv_diag_20261023f_restfix.py --rest-correct for *_rc_*),
                           with <drive> one of open, reflect, train, prbs, cw (first set), open_new, reflect_new,
                           train_new, cw_new (seeds 10-19), open_third, reflect_third, train_third (seeds 20-49),
                           open_rc, train_rc, prbs_rc (rest-corrected first set), and <model> the script's name (RK4
                           for RK4-net)
  waveforms/<drive>_<model>.json  nltlv_waveform_diag_20261023.py --seeds 10 19 (--sig-seed 20261031 for train), with
                           <drive> one of open_new, train_new, reflect_new and <model> one of LF, SIE, LF-U
  classical_steps.json     dt_classical_ref_20261021.py nlqf3_v1_dtd2 nltl8_v1_dtd2 (and the datasets with the
                           other time steps of Fig. 5, letter_figures.py)
  classical_line.json      line_classical_ref.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from lllc_paths import REPO, RUNS_ROOT, WORK_DIR  # noqa: E402

OUTPUTS = os.environ.get("LLLC_OUTPUTS", str(Path(WORK_DIR) / "outputs"))
RUNS_CSV = REPO / "results" / "letter_runs.csv"
CLASSICAL_CSV = REPO / "results" / "letter_classical.csv"
RUN_COLUMNS = ("source", "size", "lr", "model", "seed", "value")
LRS = ("1e2", "3e2")
SIZES = ("small", "medium")
RUN_ID = "abs_mse_lr{lr}_bs64_gc1_num_ltctau_cos_s{seed}{tag}"
CAP = 1e3

CIRCUIT = {"LF": "LF_Free_MLP", "LF-U": "LF_Par_MLP", "LF-KCL": "LF_Sep_MLP", "LF-C": "LF_Cen_MLP", "Res": "Res_MLP_state",
           "RK4-net": "RK4_MLP_state", "Res2": "Res2_MLP_state", "SIE": "SIE_MLP_state", "SIE-M": "SIEMid_MLP_state"}
LINE = {"LF": "LF_Free_Loc", "LF-U": "LF_Par_Loc", "LF-Sep": "LF_Sep_Loc", "LF-KCL": "LF_KCL_Loc", "LF-KCLD": "LF_KCLD_Loc",
        "LF-SepD": "LF_SepD_Loc", "RK4-KCL": "RK4_KCL_Loc", "Res": "Res_Loc_state", "RK4-net": "RK4_Loc_state",
        "Res2": "Res2_Loc_state", "SIE": "SIE_Loc_state", "SIE-M": "SIEMid_Loc_state"}
SCRIPT_NAME = {"RK4-net": "RK4"}  # names in the outputs of scripts/*.py that differ from the letter's
LETTER_NAME = {v: k for k, v in SCRIPT_NAME.items()}

# Free runs: source -> (training dataset, models, subfolder of the evaluation, end of the run id)
FREE = {
    "circuit": ("nlqf3_v1_dtd2", CIRCUIT, "", "_gpu"),
    "circuit-tan": ("nllcqf_v1_dtd2", CIRCUIT, "", "_gpu"),
    "line": ("nltl8_v1_dtd2", LINE, "", "_gpu"),
    "line-32": ("nltl8_v1_dtd2", LINE, "eval_nltl32_v1_dtd2", "_gpu"),
    "leaky": ("nltlg8_v1_dtd2", LINE, "", "_gpu"),
    "leaky-32": ("nltlg8_v1_dtd2", LINE, "eval_nltlg32_v1_dtd2", "_gpu"),
    "circuit-dt0.1": ("nlqf3_v1_dt0p1", CIRCUIT, "", "_gpu"),
    "circuit-dt0.2": ("nlqf3_v1_dt0p2", CIRCUIT, "", "_gpu"),
    "circuit-dt0.4": ("nlqf3_v1_dt0p4", CIRCUIT, "", "_gpu"),
    "line-dt0.1": ("nltl8_v1_dt0p1", LINE, "", "_gpu"),
    "line-dt0.2": ("nltl8_v1_dt0p2", LINE, "", "_gpu"),
    "line-dt0.4": ("nltl8_v1_dt0p4", LINE, "", "_gpu"),
    "circuit-rollout": ("nlqf3_v1_dtd2", CIRCUIT, "", "_ro10_gpu"),
    "line-rollout": ("nltl8_v1_dtd2", LINE, "", "_ro10_gpu"),
    "circuit-noise1": ("nlqf3_v1_dtd2", CIRCUIT, "", "_nz1_gpu"),
    "line-noise1": ("nltl8_v1_dtd2", LINE, "", "_nz1_gpu"),
    "circuit-noise0.1": ("nlqf3_v1_dtd2", CIRCUIT, "", "_nz01_gpu"),
    "line-noise0.1": ("nltl8_v1_dtd2", LINE, "", "_nz01_gpu"),
    "vi-cap": ("nlcap_v1_dtd2", CIRCUIT, "", "_gpu"),
    "vi-both": ("nllc_v1_dtd2", CIRCUIT, "", "_gpu"),
    "vi-ind": ("nlind_v1_dtd2", CIRCUIT, "", "_gpu"),
    "vi-both-e1": ("nllc_e1_v1_dtd2", CIRCUIT, "", "_gpu"),
    "vi-ind-cpu": ("nlind_v1_dtd2", CIRCUIT, "", "_cpu"),
    "ring": ("nltlv8_v1_dtd2", LINE, "", "_gpu"),
    # systems other than LC circuits (dx/dt = v), trained on CPUs; only planned tests given only in this repository
    "duffing": ("duffing_v1_dtd2", CIRCUIT, "", "_cpu"),
    "double-well": ("dwell_v1_dtd2", CIRCUIT, "", "_cpu"),
    "morse": ("morse_v1_dtd2", CIRCUIT, "", "_cpu"),
    "fpu": ("fpu4_v1_dtd2", CIRCUIT, "", "_cpu"),
    "charged-b1": ("charged_v1_dtd2", CIRCUIT, "", "_cpu"),
    "charged-b10": ("charged_B10_v1_dtd2", CIRCUIT, "", "_cpu"),
    "charged-b20": ("charged_B20_v1_dtd2", CIRCUIT, "", "_cpu"),
    "duffing-b0.1": ("duffing_b0p1_v1_dtd2", CIRCUIT, "", "_cpu"),
    "duffing-b0.3": ("duffing_b0p3_v1_dtd2", CIRCUIT, "", "_cpu"),
    "duffing-b3": ("duffing_b3_v1_dtd2", CIRCUIT, "", "_cpu"),
}
# Driven varactor line: source -> (file prefix in drives/, measure)
DRIVES = {
    "sinusoids": ("cw", "load"), "bits": ("prbs", "load"), "trains": ("train", "load"),
    "pulses-rc": ("open_rc", "load"), "bits-rc": ("prbs_rc", "load"), "trains-rc": ("train_rc", "load"),
    "sinusoids-2": ("cw_new", "load"), "pulses-2": ("open_new", "load"), "reflected-2": ("reflect_new", "load"),
    "trains-2": ("train_new", "load"), "pulses-3": ("open_third", "load"), "reflected-3": ("reflect_third", "load"),
    "trains-3": ("train_third", "load"),
    "sinusoids-energy": ("cw", "energy"), "bits-energy": ("prbs", "energy"), "trains-energy": ("train", "energy"),
    "pulses-energy": ("open", "energy"), "reflected-energy": ("reflect", "energy"), "sinusoids-energy-2": ("cw_new", "energy"),
    "pulses-energy-2": ("open_new", "energy"), "reflected-energy-2": ("reflect_new", "energy"),
    "trains-energy-2": ("train_new", "energy"), "pulses-energy-3": ("open_third", "energy"),
    "reflected-energy-3": ("reflect_third", "energy"), "trains-energy-3": ("train_third", "energy"),
}
# Components of the driven varactor line's errors, seeds 10-19: source -> (file prefix in waveforms/, key)
WAVEFORMS = {
    "pulses-active-2": ("open_new", "e_act"), "trains-active-2": ("train_new", "e_act"),
    "pulses-quiet-2": ("open_new", "e_qui"), "trains-quiet-2": ("train_new", "e_qui"),
    "pulses-delivered-2": ("open_new", "e_W"), "trains-delivered-2": ("train_new", "e_W"),
    "reflected-delivered-2": ("reflect_new", "e_W"),
}
WAVEFORM_MEASURE = {"e_act": "load-voltage error while a pulse is at the load / amplitude",
                    "e_qui": "load-voltage error at the other times / amplitude",
                    "e_W": "relative error of the energy delivered to the load"}
OTHER = {
    "embedded": "energy change over the last 10^3 of 10^4 steps of the 32-section classical ring with an eight-section "
                "learned block (embedded/*.json, T10000 drift)",
    "leaky-embedded": "deviation from the classical solution over 1000 steps of the leaky ring with an eight-section "
                      "learned block (leaky_embedded/*.json, T1000 dev)",
    "voltage-cap": "voltage error, circuit with only C(V) nonlinear, V-I (nlcap) and q-phi (nlcapqf) networks (vi_voltage.json)",
    "voltage-ind": "voltage error, circuit with only L(I) nonlinear, V-I (nlind) and q-phi (nlindqf) networks (vi_voltage.json)",
    "voltage-cubic": "voltage error, cubic circuit, V-I (nlqf3vi) and q-phi (nlqf3) networks (vi_voltage.json)",
    "pulses": "matched line, 64 pulses: RMS load-voltage error / amplitude (pulses.json)",
    "reflected": "reflecting terminations, 64 pulses, 0-10 ns: RMS load-voltage error / amplitude (reflected.json)",
    "reflected-early": "the same over 0-2.54 ns (reflected.json, err_early)",
    "reflected-late": "the same over 2.54-10 ns (reflected.json, err_late)",
    "embedded-div": "initial conditions that diverged in 10^4 steps of the 32-section classical ring with an eight-section "
                    "learned block (embedded/*.json, T10000 n_div)",
}
VOLTAGE_PAIR = {"voltage-cap": "circuit1", "voltage-ind": "circuit3", "voltage-cubic": "cubic"}


def describe(source: str) -> str:
    if source in FREE:
        ds, _, sub, tag = FREE[source]
        where = f"{ds} networks on {sub[5:]}" if sub else ds
        return f"free run, {where}, runs ending in {tag}: relative RMS error of the first variable up to t = 50"
    if source in DRIVES:
        prefix, measure = DRIVES[source]
        what = "RMS load-voltage error / amplitude" if measure == "load" else "block-energy deviation / classical maximum"
        return f"driven varactor line, drives/{prefix}_<model>.json: {what}, mean over the signals"
    if source in WAVEFORMS:
        prefix, key = WAVEFORMS[source]
        return f"driven varactor line, waveforms/{prefix}_<model>.json: {WAVEFORM_MEASURE[key]}, mean over the signals"
    return OTHER[source]


def capped(v):
    return v if v is not None and math.isfinite(v) and v < CAP else CAP


def t50(path: Path, key: str = "test_interp_x_rmse_rel_mean"):
    """The error up to t = 50 in one evaluation summary, or None if the file or the value is missing."""
    if not path.exists():
        return None
    row = next((r for r in json.load(open(path))["rows"] if r["t_eval_tag"] == "t50s"), None)
    if row is None or row.get(key) is None:
        return None
    return capped(row[key])


def state_dim(dataset: str) -> int:
    """2 for a single circuit or oscillator, twice the number of sections for a ring (nltl8, nltlg8, nltlv8: 16), 4 for
    the charged particle, and 8 for the chain of four masses."""
    if dataset.startswith("nltl"):
        return 2 * int(dataset.split("_")[0].lstrip("nltlgv"))
    if dataset.startswith("charged"):
        return 4
    if dataset.startswith("fpu4"):
        return 8
    return 2


@lru_cache(maxsize=None)
def width(cls: str, size: str, dataset: str) -> int:
    from harmonic_models import memoryless_width
    return memoryless_width(cls, size, state_dim(dataset))


def nest(rows) -> dict:
    """{stratum: {model: {seed: value}}} from rows (stratum, model, seed, value), in the order of the rows."""
    g = {}
    for key, model, seed, v in rows:
        g.setdefault(key, {}).setdefault(model, {})[seed] = v
    return g


def free_rows(source: str, seeds, sizes=SIZES, root: str = RUNS_ROOT, models=None,
              key: str = "test_interp_x_rmse_rel_mean"):
    ds, names, sub, tag = FREE[source]
    for short, cls in names.items():
        if models is not None and short not in models:
            continue
        for size in sizes:
            hs = width(cls, size, ds)
            for lr in LRS:
                for seed in seeds:
                    d = Path(root) / ds / RUN_ID.format(lr=lr, seed=seed, tag=tag) / cls / f"hs{hs}"
                    v = t50((d / sub if sub else d) / "ic_generalization_summary.json", key)
                    if v is not None:
                        yield (size, lr), short, seed, v


def free_runs(source: str, seeds, sizes=SIZES, root: str = RUNS_ROOT, models=None) -> dict:
    return nest(free_rows(source, seeds, sizes, root, models))


def embedded_rows(paths, horizon: str = "T10000", key: str = "drift", seeds=None):
    """hybrid_line_20261021.py outputs: one value per run, from several files if the runs were split."""
    for p in paths:
        for r in json.load(open(p))["runs"]:
            v = r.get(horizon, {}).get(key)
            if v is None or (seeds is not None and int(r["seed"]) not in seeds):
                continue
            yield (r["size"], r["lr"]), LETTER_NAME.get(r["model"], r["model"]), int(r["seed"]), capped(v)


def embedded(paths, horizon: str = "T10000", key: str = "drift", seeds=None) -> dict:
    return nest(embedded_rows(paths, horizon, key, seeds))


def vi_voltage_rows(path, pair: str, seeds=None):
    """vi_voltage_error.py output; models are '<model> (V-I)' and '<model> (q-phi)'."""
    runs = json.load(open(path))["runs"][pair]
    for var, label in (("vi", "V-I"), ("qf", "q-phi")):
        for model, d in runs.get(var, {}).items():
            for k, v in d.items():
                size, lr, seed = k.split("|")
                if seeds is None or int(seed) in seeds:
                    yield (size, lr), f"{LETTER_NAME.get(model, model)} ({label})", int(seed), v


def vi_voltage(path, pair: str, seeds=None) -> dict:
    return nest(vi_voltage_rows(path, pair, seeds))


def line_rows(path, key: str = "err", seeds=None):
    """nltlv_open_20261022.py and nltlv_reflect_20261022.py outputs (one size: about 2400 parameters)."""
    for r in json.load(open(path))["runs"]:
        if seeds is None or r["seed"] in seeds:
            yield ("medium", r["lr"]), LETTER_NAME.get(r["model"], r["model"]), r["seed"], min(r[key], CAP)


def line_runs(path, key: str = "err", seeds=None) -> dict:
    return nest(line_rows(path, key, seeds))


def drive_rows(directory, prefix: str, measure: str = "load", seeds=None,
               models=("LF", "SIE", "LF-U", "Res", "RK4-net", "LF-KCL")):
    """nltlv_diag_20261022.py outputs drives/<prefix>_<model>.json. Per run, the mean over the signals that did not
    diverge of the RMS load-voltage error divided by the amplitude (measure "load"), or of the RMS block-energy
    deviation divided by the classical block energy's maximum (measure "energy"; signals whose classical block energy
    is zero, two bit streams, are left out)."""
    for m in models:
        name = SCRIPT_NAME.get(m, m)
        f = Path(directory) / f"{prefix}_{name}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        amp = np.asarray(d["amp"], float)
        for r in d["runs"]:
            if r.get("model", name) != name or (seeds is not None and r["seed"] not in seeds):
                continue
            ok = ~np.asarray(r["div"], bool)
            if measure == "energy":
                scale = np.asarray(r["e_scale"], float)
                ok &= scale > 1e-9
                x = np.sqrt(np.asarray(r["e_blk"], float)) / np.maximum(scale, 1e-12)
            else:
                x = np.sqrt(np.asarray(r["load"], float)) / amp
            if ok.any():
                yield ("medium", r["lr"]), m, r["seed"], float(x[ok].mean())


def drive(directory, prefix: str, measure: str = "load", seeds=None,
          models=("LF", "SIE", "LF-U", "Res", "RK4-net", "LF-KCL")) -> dict:
    return nest(drive_rows(directory, prefix, measure, seeds, models))


def waveform_rows(directory, prefix: str, key: str, seeds=None, models=("LF", "SIE", "LF-U")):
    """nltlv_waveform_diag_20261023.py outputs waveforms/<prefix>_<model>.json. Per run, the mean over the signals that
    did not diverge of one component (WAVEFORM_MEASURE): e_act and e_qui split the RMS load-voltage error between the
    times when the classical load voltage exceeds 5 % of the amplitude and the other times (their squares add up to the
    square of the whole error), and e_W is the relative error of the energy delivered to the load."""
    for m in models:
        f = Path(directory) / f"{prefix}_{SCRIPT_NAME.get(m, m)}.json"
        if not f.exists():
            continue
        for r in json.load(open(f))["runs"]:
            if seeds is not None and r["seed"] not in seeds:
                continue
            x = np.asarray(r[key], float)
            ok = ~np.asarray(r["div"], bool) & np.isfinite(x)
            if ok.any():
                yield ("medium", r["lr"]), LETTER_NAME.get(r["model"], r["model"]), r["seed"], float(x[ok].mean())


def rows(source: str, seeds, sizes=SIZES, root: str = RUNS_ROOT, outputs: str = OUTPUTS):
    """Rows (stratum, model, seed, value) of one source with the given seeds (and sizes, for free runs)."""
    seeds = set(seeds)
    out = Path(outputs)
    if source in FREE:
        return free_rows(source, sorted(seeds), sizes, root)
    if source in DRIVES:
        prefix, measure = DRIVES[source]
        return drive_rows(out / "drives", prefix, measure, seeds)
    if source in WAVEFORMS:
        prefix, key = WAVEFORMS[source]
        return waveform_rows(out / "waveforms", prefix, key, seeds)
    if source in ("embedded", "embedded-div"):
        key = "drift" if source == "embedded" else "n_div"
        return embedded_rows(sorted(glob.glob(str(out / "embedded" / "*.json"))), "T10000", key, seeds)
    if source == "leaky-embedded":
        return embedded_rows(sorted(glob.glob(str(out / "leaky_embedded" / "*.json"))), "T1000", "dev", seeds)
    if source in VOLTAGE_PAIR:
        return vi_voltage_rows(out / "vi_voltage.json", VOLTAGE_PAIR[source], seeds)
    if source == "pulses":
        return line_rows(out / "pulses.json", "err", seeds)
    if source.startswith("reflected"):
        key = {"reflected": "err", "reflected-early": "err_early", "reflected-late": "err_late"}[source]
        return line_rows(out / "reflected.json", key, seeds)
    raise KeyError(f"unknown source {source!r}")


def load(source: str, seeds, sizes=SIZES, root: str = RUNS_ROOT, outputs: str = OUTPUTS) -> dict:
    """The runs of one source with the given seeds (and sizes, for free runs)."""
    return nest(rows(source, seeds, sizes, root, outputs))


def classical(outputs: str = OUTPUTS) -> dict:
    """The classical leapfrog scheme with the true element laws in Table IV: one deterministic value per column."""
    out = Path(outputs)
    key = "test_interp_x_rmse_rel_mean"
    steps = json.load(open(out / "classical_steps.json"))
    line = json.load(open(out / "classical_line.json"))
    refs = json.load(open(out / "embedded_references.json"))["classical"]["10000"]
    return {"Circuit": steps["nlqf3_v1_dtd2"]["t50s"][key], "N=8": steps["nltl8_v1_dtd2"]["t50s"][key],
            "N=32": line["nltl32_v1_dtd2"]["t50s"][key], "Div.": refs["n_div"], "Energy": refs["drift"]}


# ---- the per-run values saved in results/ (written by `letter_tables.py export`)

def write_runs(path, items):
    """items: (source, stratum, model, seed, value); one row each, in the given order."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(RUN_COLUMNS)
        for source, (size, lr), model, seed, v in items:
            w.writerow([source, size, lr, model, seed, repr(float(v))])


def read_runs(path=RUNS_CSV) -> dict:
    """{source: [rows (stratum, model, seed, value)]} in the order of the file."""
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out.setdefault(r["source"], []).append(((r["size"], r["lr"]), r["model"], int(r["seed"]), float(r["value"])))
    return out


def from_runs(saved: dict, source: str, seeds, sizes=SIZES) -> dict:
    """What load() returns, from the rows of read_runs(): the given seeds, and the given sizes for free runs only."""
    if not (source in FREE or source in DRIVES or source in WAVEFORMS or source in OTHER):
        raise KeyError(f"unknown source {source!r}")
    seeds = set(seeds)
    return nest(r for r in saved.get(source, []) if r[2] in seeds and (source not in FREE or r[0][0] in sizes))


def write_classical(path, values: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("column", "value"))
        for col, v in values.items():
            w.writerow([col, repr(float(v))])


def read_classical(path=CLASSICAL_CSV) -> dict:
    return {r["column"]: float(r["value"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def seed_range(text: str) -> range:
    """'0-9' -> range(0, 10)."""
    lo, hi = text.split("-")
    return range(int(lo), int(hi) + 1)
