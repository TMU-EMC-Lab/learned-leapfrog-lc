#!/usr/bin/env python3
"""Effect of the different number of evaluation points on the free-run errors (README, "Evaluation points").

system_eval.evaluate scores the staggered models on steps - 1 points and the integer-time models on steps points, where
steps = round(t/dt) + 1, because a staggered rollout of steps points gives its time-centered flux only up to the step
before the last. This script reruns the integer-time models of the cubic circuit and the eight-section line at each time
step of Fig. 5 and scores each run both ways, up to t = 50. It writes results/eval_points_runs.csv (panel, dt, model,
size, lr, seed, error with the final time, error without it) and prints the changes.

  python scripts/eval_points_check.py --out results/eval_points_runs.csv
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import ic_generalization_eval as ev  # noqa: E402
import letter_data as LD  # noqa: E402
from harmonic_ic_dataset import split_scales, split_to_array  # noqa: E402
from system_eval import _agg, _per_ic, cached_reference, rollout_state, steps_from_t_end, system_from_manifest  # noqa: E402

SOURCES = (("circuit", 0.05, "circuit"), ("circuit", 0.1, "circuit-dt0.1"), ("circuit", 0.2, "circuit-dt0.2"),
           ("circuit", 0.4, "circuit-dt0.4"), ("line", 0.05, "line"), ("line", 0.1, "line-dt0.1"),
           ("line", 0.2, "line-dt0.2"), ("line", 0.4, "line-dt0.4"))
INTEGER_TIME = ("Res", "RK4-net", "Res2", "SIE", "SIE-M")
SEEDS = range(10)
T_EVAL, SPLIT = 50.0, "test_interp"
KEY = "test_interp_x_rmse_rel_mean"


def rows(root: str):
    for panel, dt, source in SOURCES:
        ds, names, sub, tag = LD.FREE[source]
        manifest = json.load(open(Path(HERE) / ".." / "datasets" / ds / "manifest.json"))
        assert float(manifest["dt_direct"]) == dt
        sysm = system_from_manifest(manifest)
        steps = steps_from_t_end(T_EVAL, dt)
        ics, scale = split_to_array(manifest, SPLIT), split_scales(manifest, SPLIT)
        ref = cached_reference(sysm, manifest, SPLIT, ics, dt, steps)
        for model in INTEGER_TIME:
            cls = names[model]
            for size in LD.SIZES:
                hs = LD.width(cls, size, ds)
                for lr in LD.LRS:
                    for seed in SEEDS:
                        d = Path(root) / ds / LD.RUN_ID.format(lr=lr, seed=seed, tag=tag) / cls / f"hs{hs}"
                        saved = LD.t50(d / "ic_generalization_summary.json", KEY)
                        p = ev.find_model_checkpoint(d, cls, hs)
                        if saved is None or not p.exists():
                            continue
                        net, _ = ev.load_model_for_eval(cls, hs, d, p)
                        pred = rollout_state(net, ics, steps)
                        both = _agg(_per_ic(sysm, ref, pred, scale, dt, T_EVAL)["x_rmse_rel"])["mean"]
                        short = _agg(_per_ic(sysm, ref[:, :-1], pred[:, :-1], scale, dt, T_EVAL)["x_rmse_rel"])["mean"]
                        print(panel, dt, model, size, lr, seed, f"{both:.5g} (saved {saved:.5g}) {short:.5g}", flush=True)
                        yield panel, dt, model, size, lr, seed, both, short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=LD.RUNS_ROOT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    all_rows = list(rows(args.root))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("panel", "dt", "model", "size", "lr", "seed", "error_with_final_time", "error_without_final_time"))
        w.writerows((*r[:6], repr(float(r[6])), repr(float(r[7]))) for r in all_rows)
    for panel, dt, _ in SOURCES:
        x = np.array([r[7] / r[6] for r in all_rows if r[0] == panel and r[1] == dt and np.isfinite(r[6]) and np.isfinite(r[7])])
        print(f"{panel} dt = {dt}: {len(x)} runs, without / with final time: median {np.median(x):.5f}, "
              f"range {x.min():.5f}-{x.max():.5f}")


if __name__ == "__main__":
    main()
