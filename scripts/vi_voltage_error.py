#!/usr/bin/env python3
"""Voltage error of circuit networks trained in voltage-current (V-I) and in charge-flux (q-phi) variables (Sect. 4.4).

  python scripts/vi_voltage_error.py --out vi_voltage.json [--pairs circuit1 circuit3 cubic circuit2] [--seeds 0 9]

The same circuit is trained in both pairs of variables, with the same initial conditions. Each trained network is run
for 1000 steps (t = 50) from the test initial conditions, and its voltage is compared with the reference voltage:
the RMS error over time of each initial condition, divided by the larger of 0.05 and the reference's maximum |V|, is
averaged over the initial conditions with finite error of at most 100 (as in system_eval). Networks in q-phi give the
voltage through the capacitor law of their circuit. The classical leapfrog scheme with the true laws, in both pairs of
variables, is evaluated in the same way.

Output: {"runs": {pair: {"vi" | "qf": {model: {"size|lr|seed": error}}}}, "classical": {pair: {...}}}; the models use the
names of the evaluation scripts (RK4 for RK4-net, LF-Sep for the letter's LF-KCL in a single circuit).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import ic_generalization_eval as ev  # noqa: E402
import system_eval as se  # noqa: E402
from harmonic_ic_dataset import split_to_array  # noqa: E402
from harmonic_models import MODEL_REGISTRY, memoryless_width  # noqa: E402
from lllc_paths import RUNS_ROOT  # noqa: E402
from systems import system_from_manifest  # noqa: E402

# pair -> (q-phi dataset, V-I dataset, voltage of the capacitor as a function of the charge)
PAIRS = {
    "circuit1": ("nlcapqf_v1_dtd2", "nlcap_v1_dtd2", lambda q: np.tan(q)),  # only C(V) nonlinear
    "circuit3": ("nlindqf_v1_dtd2", "nlind_v1_dtd2", lambda q: q / 2.0),  # only L(I) nonlinear (C0 = 2)
    "cubic": ("nlqf3_v1_dtd2", "nlqf3vi_v1_dtd2", lambda q: q + q ** 3),
    "circuit2": ("nllcqf_v1_dtd2", "nllc_e1_v1_dtd2", lambda q: np.tan(q)),  # both nonlinear
}
MODELS = {"LF_Free_MLP": "LF", "LF_Sep_MLP": "LF-Sep", "SIE_MLP_state": "SIE", "SIEMid_MLP_state": "SIE-M",
          "Res_MLP_state": "Res", "RK4_MLP_state": "RK4"}
SIZES, LRS = ("small", "medium"), ("1e2", "3e2")
RUN_ID = "abs_mse_lr{lr}_bs64_gc1_num_ltctau_cos_s{seed}_gpu"
T_EVAL = 50.0


def manifest_of(ds):
    return json.load(open(os.path.join(HERE, "..", "datasets", ds, "manifest.json")))


def v_error(pred_v, ref_v):
    scale = np.maximum(np.abs(ref_v).max(axis=1), 0.05)
    with np.errstate(all="ignore"):
        return np.sqrt(np.mean((pred_v - ref_v) ** 2, axis=1)) / scale


def agg(e):
    ok = e[np.isfinite(e) & (e <= 100)]
    return float(ok.mean()) if len(ok) else 1e3


def rollout_pos(model, family, sysm, ics, dt, steps):
    z0 = se.initial_states(sysm, ics, dt, family, None)
    if family == "lf":
        xs, _ = ev._rollout_lf_batch(model, z0[:, 0], z0[:, 1], steps)
        return np.asarray(xs, dtype=np.float64)[:, :-1]
    return se.rollout_state(model, z0, steps)[:, :-1, 0]


def classical_lf(sysm, ics, dt, steps):
    """The classical leapfrog scheme with the true laws, in the order of LF: v_{n+1/2} = v_{n-1/2} + dt f_v(x_n, v_{n-1/2}),
    x_{n+1} = x_n + dt f_x(x_n, v_{n+1/2})."""
    z0 = se.initial_states(sysm, ics, dt, "lf", None)
    x, v = z0[:, 0].astype(np.float64), z0[:, 1].astype(np.float64)
    out = [x.copy()]
    for _ in range(steps - 1):
        v = v + dt * sysm.rhs(np.stack([x, v], 1))[:, 1]
        x = x + dt * sysm.rhs(np.stack([x, v], 1))[:, 0]
        out.append(x.copy())
    return np.stack(out, 1)[:, :-1]


def rescore(root, pairs, seeds):
    runs, classical = {}, {}
    for key in pairs:
        ds_qf, ds_vi, to_v = PAIRS[key]
        m_qf, m_vi = manifest_of(ds_qf), manifest_of(ds_vi)
        s_qf, s_vi = system_from_manifest(m_qf), system_from_manifest(m_vi)
        i_qf, i_vi = split_to_array(m_qf, "test_interp"), split_to_array(m_vi, "test_interp")
        assert np.allclose(to_v(i_qf[:, 0]), i_vi[:, 0], rtol=1e-6, atol=1e-8), f"{key}: initial conditions differ"
        dt = float(m_qf["dt_lf"])
        steps = se.steps_from_t_end(T_EVAL, dt)
        ref_v = se.cached_reference(s_vi, m_vi, "test_interp", i_vi, dt, steps)[:, :-1, 0]
        ref_q = se.cached_reference(s_qf, m_qf, "test_interp", i_qf, dt, steps)[:, :-1, 0]
        assert np.allclose(to_v(ref_q), ref_v, rtol=1e-3, atol=1e-3), f"{key}: reference solutions differ"
        classical[key] = {"qf_in_V": agg(v_error(to_v(classical_lf(s_qf, i_qf, dt, steps)), ref_v)),
                          "vi_in_V": agg(v_error(classical_lf(s_vi, i_vi, dt, steps), ref_v))}
        runs[key] = {"qf": {}, "vi": {}}
        for ds, var, sysm, ics in ((ds_qf, "qf", s_qf, i_qf), (ds_vi, "vi", s_vi, i_vi)):
            for name, short in MODELS.items():
                fam = "lf" if MODEL_REGISTRY[name]["family"] == "lf" else "direct_state"
                for size in SIZES:
                    hs = memoryless_width(name, size, 2)
                    for lr in LRS:
                        for seed in seeds:
                            hd = Path(root) / ds / RUN_ID.format(lr=lr, seed=seed) / name / f"hs{hs}"
                            p = ev.find_model_checkpoint(hd, name, hs)
                            if not p.exists():
                                continue
                            model, _ = ev.load_model_for_eval(name, hs, hd, p)
                            model = model.to(ev.DEVICE).eval()
                            with torch.no_grad():
                                pos = rollout_pos(model, fam, sysm, ics, dt, steps)
                            e = agg(v_error(to_v(pos) if var == "qf" else pos, ref_v))
                            runs[key][var].setdefault(short, {})[f"{size}|{lr}|{seed}"] = e
            print(key, var, {m: len(d) for m, d in runs[key][var].items()}, flush=True)
    return runs, classical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS))
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 9])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs, classical = rescore(args.root, args.pairs, range(args.seeds[0], args.seeds[1] + 1))
    json.dump({"runs": runs, "classical": classical}, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
