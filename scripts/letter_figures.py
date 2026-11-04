#!/usr/bin/env python3
"""The values plotted in Figs. 4 and 5 of the letter, computed from the per-run values.

  python scripts/letter_figures.py           the plotted medians and the legend of Fig. 4 (results/letter_figures.md)
  python scripts/letter_figures.py export    rewrite results/fig4_runs.csv and results/fig5_runs.csv

The per-run values are read from results/fig4_runs.csv and results/fig5_runs.csv, or with --raw from the trained runs
(--runs-root, default $LLLC_RUNS_ROOT) and the evaluation outputs (--outputs, default $LLLC_OUTPUTS; layout in
scripts/letter_data.py); export always reads the latter. Each plotted point is a median over the runs of one model,
learning rates 1e-2 and 3e-2, both sizes, and seeds 0-9 (40 runs).

Fig. 4 (fig4_runs.csv: steps, model, size, lr, seed, energy_change, diverged): the 32-section classical ring with an
eight-section learned block, from embedded/*.json and embedded_references.json written with --horizons 100 200 500
1000 2000 5000 10000. energy_change is the relative energy change averaged over the initial conditions that did not
diverge, and diverged the number of the 512 initial conditions that diverged; the legend gives the median of diverged
at 10^4 steps. The rows "classical" (all sections classical) and "true laws, full steps" have no size, lr, or seed.

Fig. 5 (fig5_runs.csv: panel, dt, model, size, lr, seed, error, diverged): the free-run error up to t = 50 of Table IV
against the time step, in the cubic circuit (nlqf3) and in the eight-section line (nltl8). error is averaged over the
test initial conditions whose error stayed finite and at most 100, and diverged counts the others, of 512. The rows
"classical" are the classical scheme of classical_steps.json (dt_classical_ref_20261021.py). The line at 0.4 is not
plotted; the caption reports that all generic networks failed there.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import letter_data as LD  # noqa: E402

FIG4_CSV = LD.REPO / "results" / "fig4_runs.csv"
FIG5_CSV = LD.REPO / "results" / "fig5_runs.csv"
FIG4_COLUMNS = ("steps", "model", "size", "lr", "seed", "energy_change", "diverged")
FIG5_COLUMNS = ("panel", "dt", "model", "size", "lr", "seed", "error", "diverged")
DIVERGED = "test_interp_x_rmse_rel_n_diverged"  # key of the evaluation summaries and of classical_steps.json
MODELS = ("LF-KCL", "LF", "SIE", "LF-U", "RK4-net", "Res")  # in the circuit, LF-KCL is LF_Sep_MLP (letter_data.CIRCUIT)
HORIZONS = (100, 200, 500, 1000, 2000, 5000, 10000)
REFERENCES = {"classical": "classical", "exact_full": "true laws, full steps"}  # keys of embedded_references.json
# Fig. 5: panel -> [(time step, free-run source in letter_data.FREE, dataset of the classical value)]
STEPS = {"circuit": [(0.05, "circuit", "nlqf3_v1_dtd2"), (0.1, "circuit-dt0.1", "nlqf3_v1_dt0p1"),
                     (0.2, "circuit-dt0.2", "nlqf3_v1_dt0p2"), (0.4, "circuit-dt0.4", "nlqf3_v1_dt0p4")],
         "line": [(0.05, "line", "nltl8_v1_dtd2"), (0.1, "line-dt0.1", "nltl8_v1_dt0p1"),
                  (0.2, "line-dt0.2", "nltl8_v1_dt0p2"), (0.4, "line-dt0.4", "nltl8_v1_dt0p4")]}
NOT_PLOTTED = {("line", 0.4)}
SEEDS = range(10)


def fig4_rows(outputs: str = LD.OUTPUTS):
    """Rows (steps, model, size, lr, seed, energy_change, diverged); the references have size, lr, and seed ''."""
    out = Path(outputs)
    paths = sorted(glob.glob(str(out / "embedded" / "*.json")))
    for T in HORIZONS:
        div = {(key, m, s): v for key, m, s, v in LD.embedded_rows(paths, f"T{T}", "n_div", set(SEEDS))}
        for (size, lr), model, seed, v in LD.embedded_rows(paths, f"T{T}", "drift", set(SEEDS)):
            if model in MODELS:
                yield T, model, size, lr, seed, v, div[(size, lr), model, seed]
    refs = json.load(open(out / "embedded_references.json"))
    for key, name in REFERENCES.items():
        for T in HORIZONS:
            r = refs[key][str(T)]
            yield T, name, "", "", "", r["drift"], r["n_div"]


def fig5_rows(root: str = LD.RUNS_ROOT, outputs: str = LD.OUTPUTS):
    """Rows (panel, dt, model, size, lr, seed, error, diverged); the classical rows have size, lr, and seed ''."""
    steps = json.load(open(Path(outputs) / "classical_steps.json"))
    for panel, points in STEPS.items():
        for dt, source, ds in points:
            div = {(key, m, s): v for key, m, s, v in LD.free_rows(source, SEEDS, LD.SIZES, root, MODELS, DIVERGED)}
            for (size, lr), model, seed, v in LD.free_rows(source, SEEDS, LD.SIZES, root, MODELS):
                yield panel, dt, model, size, lr, seed, v, div[(size, lr), model, seed]
            c = steps[ds]["t50s"]
            yield panel, dt, "classical", "", "", "", c["test_interp_x_rmse_rel_mean"], c[DIVERGED]


def as_text(row) -> list[str]:
    """A row as the CSV files hold it; the last two entries are the value (repr of a float) and the diverged count."""
    *head, value, diverged = row
    return [str(v) for v in head] + [repr(float(value)), str(int(diverged))]


def write_csv(path, columns, rows) -> int:
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow(as_text(r))
            n += 1
    return n


def read_csv(path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8")))


def export(root: str, outputs: str) -> tuple[int, int]:
    FIG4_CSV.parent.mkdir(parents=True, exist_ok=True)
    return (write_csv(FIG4_CSV, FIG4_COLUMNS, fig4_rows(outputs)),
            write_csv(FIG5_CSV, FIG5_COLUMNS, fig5_rows(root, outputs)))


def as_dicts(columns, rows) -> list[dict]:
    """The rows as read_csv() returns them."""
    return [dict(zip(columns, as_text(r))) for r in rows]


def summary(rows: list[dict], keys, value: str, how=np.median) -> dict:
    """{key tuple: (how(values), number of rows)}."""
    g = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(float(r[value]))
    return {k: (float(how(v)), len(v)) for k, v in g.items()}


def cell(m, key, reference: bool) -> str:
    if key not in m:
        return "--"
    v, n = m[key]
    return f"{v:.3g}" + ("" if reference else f" ({n})")


def print_fig4(rows: list[dict]):
    energy = summary(rows, ("model", "steps"), "energy_change")
    div = summary(rows, ("model", "steps"), "diverged")
    print("Relative energy change of the ring by number of steps: medians over the runs of each model (run counts in "
          "parentheses) of the mean over the initial conditions that did not diverge. Diverged: median number of the "
          "512 initial conditions that diverged by 10^4 steps (the legend).")
    print("| Model | " + " | ".join(str(T) for T in HORIZONS) + " | Diverged |")
    print("|---|" + "---|" * (len(HORIZONS) + 1))
    for m in list(MODELS) + list(REFERENCES.values()):
        ref = m in REFERENCES.values()
        cells = [cell(energy, (m, str(T)), ref) for T in HORIZONS]
        print(f"| {m} | " + " | ".join(cells) + f" | {cell(div, (m, str(HORIZONS[-1])), True)} |")


def print_fig5(rows: list[dict]):
    err = summary(rows, ("panel", "model", "dt"), "error")
    div = summary(rows, ("panel", "model", "dt"), "diverged", np.sum)
    titles = {"circuit": "(a) cubic circuit", "line": "(b) 8-section line"}
    print("Rollout error up to t = 50 by time step: medians over the runs of each model (run counts in parentheses). "
          "Diverged: initial conditions left out of the error (non-finite or above 100), of 512 per run, summed over "
          "the runs, at each time step in order.")
    for panel, points in STEPS.items():
        dts = [repr(dt) for dt, _, _ in points]
        head = [d + (" (not plotted)" if (panel, float(d)) in NOT_PLOTTED else "") for d in dts]
        print(f"\n{titles[panel]}\n")
        print("| Model | " + " | ".join(head) + " | Diverged |")
        print("|---|" + "---|" * (len(dts) + 1))
        for m in list(MODELS) + ["classical"]:
            n_div = ", ".join(f"{div[panel, m, d][0]:.0f}" if (panel, m, d) in div else "--" for d in dts)
            print(f"| {m} | " + " | ".join(cell(err, (panel, m, d), m == "classical") for d in dts) + f" | {n_div} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", choices=["print", "export"], default="print")
    ap.add_argument("--raw", action="store_true", help="read the trained runs and the evaluation outputs, not results/")
    ap.add_argument("--runs-root", default=LD.RUNS_ROOT)
    ap.add_argument("--outputs", default=LD.OUTPUTS)
    args = ap.parse_args()
    if args.what == "export":
        n4, n5 = export(args.runs_root, args.outputs)
        print(f"wrote {n4} rows to {FIG4_CSV} and {n5} rows to {FIG5_CSV}")
        return 0
    if args.raw:
        fig4 = as_dicts(FIG4_COLUMNS, fig4_rows(args.outputs))
        fig5 = as_dicts(FIG5_COLUMNS, fig5_rows(args.runs_root, args.outputs))
    else:
        fig4, fig5 = read_csv(FIG4_CSV), read_csv(FIG5_CSV)
    print("## Fig. 4\n")
    print_fig4(fig4)
    print("\n## Fig. 5\n")
    print_fig5(fig5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
