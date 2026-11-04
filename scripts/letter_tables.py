#!/usr/bin/env python3
"""The planned tests, Table IV, Table V, and Fig. 3 of the letter, computed from the per-run values.

  python scripts/letter_tables.py tests [--family NAME ...] [--json OUT.json]   every test of docs/planned_tests.csv
  python scripts/letter_tables.py table4                                         medians of Table IV
  python scripts/letter_tables.py table5                                         Table V, with the block energy and the later pulse
                                                                                 sets that only this repository gives
  python scripts/letter_tables.py fig3                                           ratios, seed intervals, and daggers of Fig. 3
  python scripts/letter_tables.py all                                            all of the above (results/letter_tables.md)
  python scripts/letter_tables.py export                                         rewrite results/letter_runs.csv and
                                                                                 results/letter_classical.csv

In docs/planned_tests.csv, a source 'X / Y' is a ratio of ratios (condition X over condition Y), 'X + Y + Z' averages the
log ratios over the sources, and 'X < Y < Z' is an ordering over conditions without a p-value; models 'A1/B1' and
'A2/B2' compare two pairs' ratios in one condition.

The per-run values are read from results/letter_runs.csv and results/letter_classical.csv, or with --raw from the
trained runs (--runs-root, default $LLLC_RUNS_ROOT) and the evaluation outputs (--outputs, default $LLLC_OUTPUTS;
layout in scripts/letter_data.py); export always reads the latter. The procedure is in scripts/letter_stats.py: ratios
are geometric means over runs matched by seed, size, and learning rate, and a dagger marks a comparison significant in
both one-sided sign-flip tests (over runs and over seed means) after Holm's correction within its planned family.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import letter_data as LD  # noqa: E402
import letter_stats as LS  # noqa: E402

PLAN = HERE.parent / "docs" / "planned_tests.csv"


def read_plan(path: Path = PLAN) -> dict:
    """{family: [rows]} in the order of the file."""
    fams = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        fams.setdefault(r["family"], []).append(r)
    return fams


class Data:
    """The trained runs and the evaluation outputs; loads each source once."""

    def __init__(self, root: str, outputs: str):
        self.root, self.outputs, self.cache = root, outputs, {}

    def get(self, source: str, seeds, sizes=LD.SIZES) -> dict:
        key = (source, tuple(seeds), tuple(sizes))
        if key not in self.cache:
            self.cache[key] = LD.load(source, seeds, sizes, self.root, self.outputs)
        return self.cache[key]

    def classical(self) -> dict:
        return LD.classical(self.outputs)


class SavedData:
    """The per-run values in results/letter_runs.csv and results/letter_classical.csv."""

    def __init__(self, runs=LD.RUNS_CSV, classical=LD.CLASSICAL_CSV):
        self.saved, self.ref = LD.read_runs(runs), LD.read_classical(classical)

    def get(self, source: str, seeds, sizes=LD.SIZES) -> dict:
        return LD.from_runs(self.saved, source, seeds, sizes)

    def classical(self) -> dict:
        return dict(self.ref)


def log_ratios_for(data: Data, source: str, a: str, b: str, seeds, sizes=LD.SIZES) -> dict:
    """D for one comparison. Source 'X + Y + Z': mean over the sources, each of which may be 'X / Y', a ratio of ratios,
    condition X over condition Y. Models 'A1/B1' and 'A2/B2': the ratio of the two pairs' ratios in one condition."""
    if " + " in source:
        return LS.pooled([log_ratios_for(data, s, a, b, seeds, sizes) for s in source.split(" + ")])
    if " / " in source:
        s1, s2 = source.split(" / ")
        return LS.ratio_of_ratios(data.get(s1, seeds, sizes), data.get(s2, seeds, sizes), a, b)
    if "/" in a:
        (a1, b1), (a2, b2) = a.split("/"), b.split("/")
        return LS.interaction(data.get(source, seeds, sizes), a1, b1, a2, b2)
    return LS.log_ratios(data.get(source, seeds, sizes), a, b)


def run_family(data: Data, rows: list[dict]) -> dict:
    """The tests of one family with Holm's correction; an ordering 'X < Y < Z' gives its estimates and outcome only."""
    res, orderings = {}, {}
    for r in rows:
        seeds, sizes, name = LD.seed_range(r["seeds"]), tuple(r["sizes"].split()), f"{r['family']}/{r['test']}"
        if " < " in r["source"]:
            orderings[r["test"]] = [LS.estimate(log_ratios_for(data, s, r["model_a"], r["model_b"], seeds, sizes), f"{name}/{s}")
                                    for s in r["source"].split(" < ")]
            continue
        d = log_ratios_for(data, r["source"], r["model_a"], r["model_b"], seeds, sizes)
        res[r["test"]] = LS.compare(d, name)
    out = LS.family(res) if res else {}
    for k, parts in orderings.items():
        out[k] = {"ratio": None, "ordering": parts, "held": LS.ordered(parts)}
    return {r["test"]: out[r["test"]] for r in rows}


def run_tests(data: Data, families=None) -> dict:
    """{(family, test): row of the plan with the results} for the given families (all by default)."""
    out = {}
    for fam, rows in read_plan().items():
        if families and fam not in families:
            continue
        res = run_family(data, rows)
        for r in rows:
            out[(fam, r["test"])] = {**r, **res[r["test"]]}
    return out


def fmt_p(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def comparison(r: dict) -> str:
    a, b = r["model_a"], r["model_b"]
    return f"({a})/({b})" if "/" in a else f"{a}/{b}"


def print_tests(res: dict):
    print("| Family / test | In the letter | A/B | Ratio [95% seed interval] | Runs | Seeds | p runs (Holm) | p seed means (Holm) | Dagger |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (fam, test), r in res.items():
        comp = comparison(r)
        if "ordering" in r:
            parts = r["ordering"]
            est = " < ".join("--" if x is None else f"{x['ratio']:.3g} [{x['lo']:.3g}, {x['hi']:.3g}]" for x in parts)
            runs = " + ".join(str(x["runs"]) if x else "0" for x in parts)
            seeds = max((x["seeds"] for x in parts if x), default=0)
            print(f"| {fam} / {test} | {r['letter']} | {comp} | {est} | {runs} | {seeds} | ordering {'held' if r['held'] else 'not held'} "
                  f"(no p) | -- | |")
            continue
        if r["ratio"] is None:
            print(f"| {fam} / {test} | {r['letter']} | {comp} | no stratum left | 0 | 0 | 1 ({fmt_p(r['p_runs_holm'])}) | "
                  f"1 ({fmt_p(r['p_seeds_holm'])}) | |")
            continue
        print(f"| {fam} / {test} | {r['letter']} | {comp} | {r['ratio']:.3g} [{r['lo']:.3g}, {r['hi']:.3g}] | {r['runs']} | {r['seeds']} | "
              f"{fmt_p(r['p_runs'])} ({fmt_p(r['p_runs_holm'])}) | {fmt_p(r['p_seeds'])} ({fmt_p(r['p_seeds_holm'])}) | "
              f"{'†' if r['dagger'] else ''} |")


# Fig. 3: (label, group, family, test); group 0: one-network baselines, 1: off-center ablations
FIG3 = [
    ("Cubic circuit", 0, "circuit", "C4"), ("Cubic circuit", 0, "circuit", "C5"),
    ("Line (N=8)", 0, "line", "L4"), ("Line (N=8)", 0, "line", "L5"), ("Line (N=32)", 0, "line", "L7"),
    ("Leaky ring", 0, "leaky", "G2"), ("Leaky ring", 0, "leaky", "G3"),
    ("Embedded ring, energy", 0, "embedded", "E1"), ("Embedded ring, energy", 0, "embedded", "E2"),
    ("Cubic circuit", 1, "circuit", "C1"), ("Cubic circuit, 2nd set", 1, "second-set", "R1"),
    ("Cubic circuit, 2nd set", 1, "second-set", "R2"), ("Circuit, rollout loss", 1, "training", "T3-M-circuit"),
    ("Circuit, 0.1% noise", 1, "input-noise", "N2-SIEM-circuit"), ("Line (N=8)", 1, "line", "L1"),
    ("Line (N=8), 2nd set", 1, "second-set", "R3"), ("Line (N=8), 2nd set", 1, "second-set", "R4"),
    ("Line, rollout loss", 1, "training", "T3-M-line"), ("Line, 0.1% noise", 1, "input-noise", "N2-SIEM-line"),
    ("Leaky ring", 1, "leaky", "G4"), ("Embedded ring, energy", 1, "embedded", "E3"),
]


def fig3(data: Data) -> list[dict]:
    res = run_tests(data, {f for _, _, f, _ in FIG3})
    out = []
    for label, group, fam, test in FIG3:
        r = res[(fam, test)]
        out.append({"label": label, "group": group, "family": fam, "test": test, "a": r["model_a"], "b": r["model_b"],
                    "ratio": r["ratio"], "lo": r["lo"], "hi": r["hi"], "dagger": r["dagger"]})
    return out


def print_fig3(rows: list[dict]):
    print("| Group | Row | A/B | Ratio | 95% seed interval | Dagger | Test |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        grp = "one-network baselines" if r["group"] == 0 else "off-center ablations"
        print(f"| {grp} | {r['label']} | {r['a']}/{r['b']} | {r['ratio']:.3g} | [{r['lo']:.3g}, {r['hi']:.3g}] | "
              f"{'†' if r['dagger'] else ''} | {r['family']} / {r['test']} |")


# Table V: rows (label, seeds, cells); a cell is (column, planned test or None, source, model A); B is always LF.
# The letter's table has T4_LETTER_COLUMNS and not T4_REPOSITORY_ROWS; the rest is given only in this repository.
T4_LETTER_COLUMNS = ["Load Res", "Load RK4-net", "Load SIE", "Load LF-U"]
T4_REPOSITORY_COLUMNS = ["Load SIE", "Load LF-U", "Energy SIE", "Energy LF-U"]
T4_REPOSITORY_ROWS = ("Pulses, 2nd set", "Pulses, 3rd set")
T4 = [
    ("Training ring", "0-9", [("Load Res", ("varactor", "V2"), "ring", "Res"), ("Load RK4-net", ("varactor", "V3"), "ring", "RK4-net"),
                              ("Load SIE", ("varactor", "V1"), "ring", "SIE"), ("Load LF-U", None, "ring", "LF-U")]),
    ("Pulses, matched", "0-9", [("Load Res", ("varactor", "V5"), "pulses", "Res"), ("Load RK4-net", ("varactor", "V6"), "pulses", "RK4-net"),
                                ("Load SIE", ("varactor", "V4"), "pulses", "SIE"), ("Load LF-U", None, "pulses", "LF-U"),
                                ("Energy SIE", None, "pulses-energy", "SIE"), ("Energy LF-U", None, "pulses-energy", "LF-U")]),
    ("Pulses, reflected", "0-9", [("Load Res", ("varactor-reflected", "R3"), "reflected", "Res"),
                                  ("Load RK4-net", ("varactor-reflected", "R4"), "reflected", "RK4-net"),
                                  ("Load SIE", ("varactor-reflected", "R1"), "reflected", "SIE"),
                                  ("Load LF-U", ("varactor-reflected", "R2"), "reflected", "LF-U"),
                                  ("Energy SIE", None, "reflected-energy", "SIE"), ("Energy LF-U", None, "reflected-energy", "LF-U")]),
    ("Pulse trains", "0-9", [(c, None, "trains", m) for c, m in (("Load Res", "Res"), ("Load RK4-net", "RK4-net"), ("Load SIE", "SIE"),
                                                                ("Load LF-U", "LF-U"))]
     + [("Energy SIE", None, "trains-energy", "SIE"), ("Energy LF-U", None, "trains-energy", "LF-U")]),
    ("Bit streams", "0-9", [(c, None, "bits", m) for c, m in (("Load Res", "Res"), ("Load RK4-net", "RK4-net"), ("Load SIE", "SIE"),
                                                             ("Load LF-U", "LF-U"))]
     + [("Energy SIE", None, "bits-energy", "SIE"), ("Energy LF-U", None, "bits-energy", "LF-U")]),
    ("Sinusoids", "0-9", [("Load Res", None, "sinusoids", "Res"), ("Load RK4-net", None, "sinusoids", "RK4-net"),
                          ("Load SIE", ("varactor-drives", "P1"), "sinusoids", "SIE"), ("Load LF-U", ("varactor-drives", "P2"), "sinusoids", "LF-U"),
                          ("Energy SIE", None, "sinusoids-energy", "SIE"), ("Energy LF-U", None, "sinusoids-energy", "LF-U")]),
    ("Pulses, 2nd set", "10-19", [("Load SIE", None, "pulses-2 + reflected-2 + trains-2", "SIE"),
                                  ("Load LF-U", None, "pulses-2 + reflected-2 + trains-2", "LF-U"),
                                  ("Energy SIE", ("varactor-energy-2", "E2"), "pulses-energy-2 + reflected-energy-2 + trains-energy-2", "SIE"),
                                  ("Energy LF-U", ("varactor-energy-2", "E1"), "pulses-energy-2 + reflected-energy-2 + trains-energy-2", "LF-U")]),
    ("Pulses, 3rd set", "20-49", [("Load SIE", None, "pulses-3 + reflected-3 + trains-3", "SIE"),
                                  ("Energy SIE", ("varactor-energy-3", "F1"), "pulses-energy-3 + reflected-energy-3 + trains-energy-3", "SIE")]),
    ("Sinusoids, 2nd set", "10-19", [("Load SIE", ("varactor-sinusoids-2", "S1"), "sinusoids-2", "SIE"),
                                     ("Load LF-U", ("varactor-sinusoids-2", "S2"), "sinusoids-2", "LF-U"),
                                     ("Energy SIE", None, "sinusoids-energy-2", "SIE"), ("Energy LF-U", None, "sinusoids-energy-2", "LF-U")]),
]


def table5(data: Data) -> list[dict]:
    fams = {t[0] for _, _, cells in T4 for _, t, _, _ in cells if t}
    res = run_tests(data, fams)
    out = []
    for label, seeds, cells in T4:
        row = {"row": label}
        for col, test, source, a in cells:
            if test:
                r = res[test]
                assert r["model_a"] == a and r["model_b"] == "LF" and r["source"] == source, f"Table V cell {label}/{col} != {test}"
                row[col] = {"ratio": r["ratio"], "tested": True, "dagger": r["dagger"], "test": "/".join(test),
                            "p": max(r["p_runs_holm"], r["p_seeds_holm"])}
            else:
                d = log_ratios_for(data, source, a, "LF", LD.seed_range(seeds), ("medium",))
                D = np.array([v for x in d.values() for v in x.values()])
                row[col] = {"ratio": math.exp(D.mean()) if len(D) else None, "tested": False, "dagger": False}
        out.append(row)
    return out


def print_t4_cells(rows: list[dict], columns: list[str]):
    print("| Row | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for r in rows:
        cells = []
        for c in columns:
            x = r.get(c)
            if x is None or x["ratio"] is None:
                cells.append("--")
            elif x["tested"]:
                cells.append(f"{x['ratio']:.3g}" + ("†" if x["dagger"] else "*") + f" ({fmt_p(x['p'])})")
            else:
                cells.append(f"{x['ratio']:.3g}")
        print(f"| {r['row']} | " + " | ".join(cells) + " |")


def print_table5(rows: list[dict]):
    print("Ratios to LF in load voltage (charge on the training ring). †: both Holm-adjusted p < 0.05; *: planned test without †; "
          "in parentheses, the larger of the two Holm-adjusted p; plain: not tested; Load Res: Euler-net.")
    print_t4_cells([r for r in rows if r["row"] not in T4_REPOSITORY_ROWS], T4_LETTER_COLUMNS)
    print("\n### SIE and LF-U in load voltage and block energy, every drive (the block energy and the rows 'Pulses, 2nd set' and "
          "'Pulses, 3rd set' are given only in this repository)\n")
    print_t4_cells([r for r in rows if r["row"] != "Training ring"], T4_REPOSITORY_COLUMNS)


T3_MODELS = ["LF", "Res", "RK4-net", "LF-U", "SIE", "SIE-M", "LF-KCL"]
# columns of Table IV: medians over the runs of one source
T3_SOURCES = {"Circuit": "circuit", "N=8": "line", "N=32": "line-32", "Div.": "embedded-div", "Energy": "embedded"}


def median_of(groups: dict, model: str):
    v = [e for cell in groups.values() for e in cell.get(model, {}).values()]
    return (float(np.median(v)), len(v)) if v else (None, 0)


def table4(data) -> dict:
    seeds = range(10)
    out = {m: {} for m in T3_MODELS + ["Classical"]}
    for col, source in T3_SOURCES.items():
        g = data.get(source, seeds)
        for m in T3_MODELS:
            out[m][col] = median_of(g, m)
    # the classical leapfrog scheme with the true element laws (one deterministic run per column)
    ref = data.classical()
    for col in T3_SOURCES:
        out["Classical"][col] = (ref[col], 1)
    return out


def print_table4(t: dict):
    cols = ["Circuit", "N=8", "N=32", "Div.", "Energy"]
    print("Medians over the runs of each model (run counts in parentheses).")
    print("| Model | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for m, row in t.items():
        cells = []
        for c in cols:
            v, n = row.get(c, (None, 0))
            if v is None:
                cells.append("--")
            else:
                cells.append(f"{v:.3g}" + (f" ({n})" if m != "Classical" else ""))
        print(f"| {m} | " + " | ".join(cells) + " |")


class Recorder(Data):
    """Data that also notes the seeds and sizes of each source that the tables use."""

    def __init__(self, root: str, outputs: str):
        super().__init__(root, outputs)
        self.used = {}

    def get(self, source: str, seeds, sizes=LD.SIZES) -> dict:
        s, z = self.used.setdefault(source, (set(), set()))
        s.update(seeds)
        z.update(sizes)
        return super().get(source, seeds, sizes)


def compared_models() -> dict:
    """{source: models} that the planned tests and Tables IV and V compare."""
    need = {}

    def add(source, models):
        names = {m for x in models for m in x.split("/")}  # 'A1/B1': the two models of one pair
        for s in source.replace(" / ", " + ").replace(" < ", " + ").split(" + "):
            need.setdefault(s, set()).update(names)

    for rows in read_plan().values():
        for r in rows:
            add(r["source"], (r["model_a"], r["model_b"]))
    for _, _, cells in T4:
        for _, _, source, a in cells:
            add(source, (a, "LF"))
    for source in T3_SOURCES.values():
        add(source, T3_MODELS)
    return need


def export(root: str, outputs: str, runs=LD.RUNS_CSV, classical=LD.CLASSICAL_CSV) -> int:
    """Writes the values of the compared models in every source, seed, and size that the tests and tables use, in the
    order in which load() reads them, so that the saved values give the same output."""
    rec = Recorder(root, outputs)
    run_tests(rec)
    table4(rec)
    table5(rec)
    need = compared_models()
    items = []
    for source, (seeds, sizes) in rec.used.items():
        for key, model, seed, v in LD.rows(source, seeds, [z for z in LD.SIZES if z in sizes], root, outputs):
            if model in need[source]:
                items.append((source, key, model, seed, v))
    Path(runs).parent.mkdir(parents=True, exist_ok=True)
    LD.write_runs(runs, items)
    LD.write_classical(classical, rec.classical())
    return len(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["tests", "table4", "table5", "fig3", "all", "export"])
    ap.add_argument("--raw", action="store_true", help="read the trained runs and the evaluation outputs, not results/")
    ap.add_argument("--runs-root", default=LD.RUNS_ROOT)
    ap.add_argument("--outputs", default=LD.OUTPUTS)
    ap.add_argument("--family", nargs="+", default=None, help="only these families (tests)")
    ap.add_argument("--json", default=None, help="also write the results of 'tests' to this file")
    ap.add_argument("--leave-out", choices=["both", "any"], default="both",
                    help="leave a stratum out when both models failed (the letter) or when either did (the plans of H1-H6)")
    args = ap.parse_args()
    LS.LEAVE_OUT = args.leave_out
    if args.what == "export":
        n = export(args.runs_root, args.outputs)
        print(f"wrote {n} runs to {LD.RUNS_CSV} and the classical values to {LD.CLASSICAL_CSV}")
        return 0
    data = Data(args.runs_root, args.outputs) if args.raw else SavedData()
    if args.what in ("tests", "all"):
        res = run_tests(data, set(args.family) if args.family else None)
        print("## Planned tests\n")
        print_tests(res)
        if args.json:
            json.dump([{"family": f, **v} for (f, _), v in res.items()], open(args.json, "w"), indent=1)
    if args.what in ("table4", "all"):
        print("\n## Table IV\n")
        print_table4(table4(data))
    if args.what in ("table5", "all"):
        print("\n## Table V\n")
        print_table5(table5(data))
    if args.what in ("fig3", "all"):
        print("\n## Fig. 3\n")
        print_fig3(fig3(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
