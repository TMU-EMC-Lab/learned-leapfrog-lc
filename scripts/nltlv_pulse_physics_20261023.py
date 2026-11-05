#!/usr/bin/env python3
"""Per-network dc offset and pulse arrival times of the driven varactor line (sources dc-offset and arrival-drift of
letter_text.py). It only evaluates trained networks, the block of 32 sections of nltlv_open_20261022.py and
nltlv_reflect_20261022.py.

Each batch holds the 64 pulses and a last row without a source, which gives the free-running line.

dq0, dphi0   the block's increments in one step at the bias point (all states 0). In the matched line they predict the
             dc load voltage 320 (dq0 + dphi0): 32 sections, each a dc current dq0/dt into a node and a dc series
             voltage dphi0/dt, with 1-ohm (50 ohm) source and load
dc_m         matched line, sources off: mean load voltage over the last 400 of 2000 steps
off_div_m    whether that free-running line diverged
shift        reflecting line: for each pulse, the shift of the network's load voltage (steps; negative: the network's
             wave arrives early) that minimizes the squared difference from the conventional line, in each of five time
             windows that hold the arrivals after block crossings 1, 3, 5, 7, and 9
div_r        whether each pulse's reflecting line diverged

  python scripts/nltlv_pulse_physics_20261023.py --set old --models LF --out $O/pulse_physics/old_LF.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import nltlv_diag_20261022 as D  # noqa: E402
import nltlv_waveform_diag_20261023 as W  # noqa: E402
from lllc_paths import RUNS_ROOT  # noqa: E402

DT = 0.05
SETS = {"old": (0, 9), "new": (10, 19)}
SMAX, FINE = 15, 0.05  # integer shifts up to +-15 steps, then +-1 around the best in steps of 0.05
# Starts of the arrival windows (t0). The conventional load voltage first exceeds 5% of the amplitude at a median of
# 38, 115, 195, 275, and 356 t0, so each window starts about 20 t0 before that.
WINDOWS_T0 = (0.0, 95.0, 175.0, 255.0, 335.0)


def r6(x):
    return [float(f"{v:.6g}") if np.isfinite(v) else None for v in np.asarray(x, dtype=np.float64).ravel()]


def bias_increments(model, kind, dev):
    z = torch.zeros(1, D.N, device=dev)
    if kind == "stag":
        return float(D.rest_inc_stag(model, z, z, False)[0, 0]), float(D.rest_inc_stag(model, z, z, True)[0, 0])
    d = D.rest_inc_full(model, z, z)
    return float(d[0, 0]), float(d[0, D.N])


def window_masks(B, T):
    bounds = [int(round(g / DT)) for g in WINDOWS_T0] + [T]
    masks = np.zeros((len(WINDOWS_T0), B, T), bool)
    for k in range(len(WINDOWS_T0)):
        masks[k, :, max(20, bounds[k]): min(T - 20, bounds[k + 1])] = True
    return masks


def best_shift(x, y, masks):
    """The s minimizing the sum over each mask of (x(t + s) - y(t))^2, with linear interpolation and the end values
    beyond the ends. x, y: (B, T); masks: (R, B, T). Returns (B, R)."""
    B, T = x.shape
    pad = SMAX + 3
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    mf = masks.astype(np.float64)
    ints = np.arange(-SMAX, SMAX + 1)
    e_int = np.empty((len(ints), B, masks.shape[0]))
    for i, s in enumerate(ints):
        e_int[i] = np.einsum("bt,rbt->br", (xp[:, pad + s: pad + s + T] - y) ** 2, mf)
    s1 = ints[e_int.argmin(0)].astype(np.float64)
    best = np.full_like(s1, np.inf)
    sb = np.zeros_like(s1)
    tt = np.arange(T)[None, :]
    for r in range(masks.shape[0]):
        for dl in np.arange(-1.0, 1.0 + 1e-9, FINE):
            s = s1[:, r] + dl
            k = np.floor(s + 1e-9).astype(int)
            f = (s - k)[:, None]
            idx = tt + pad + k[:, None]
            xs = (1 - f) * np.take_along_axis(xp, idx, 1) + f * np.take_along_axis(xp, idx + 1, 1)
            e = (((xs - y) ** 2) * mf[r]).sum(1)
            better = e < best[:, r]
            best[better, r] = e[better]
            sb[better, r] = s[better]
    return sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), required=True, help="old: seeds 0-9; new: seeds 10-19")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--lrs", nargs="+", default=["1e2", "3e2"])
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--size", default="medium")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    O, RF = D.O, D.RF
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f32 = dict(device=dev, dtype=torch.float32)
    # O.pulses() and RF.pulses() end with a display signal, which is replaced by the row without a source
    a_m, tr_m = O.pulses()
    a_m_t, tr_m_t = torch.as_tensor(np.append(a_m[:-1], 0.0), **f32), torch.as_tensor(np.append(tr_m[:-1], 8.0), **f32)
    a_r, tr_r = RF.pulses()
    a_r_t, tr_r_t = torch.as_tensor(np.append(a_r[:-1], 0.0), **f32), torch.as_tensor(np.append(tr_r[:-1], 8.0), **f32)

    def src_m(t):
        return 2.0 * a_m_t * O.p_of(t, tr_m_t)

    def src_r(t):
        return a_r_t * (1.0 + RF.RS) * O.p_of(t, tr_r_t)

    lo, hi = SETS[args.set]
    res = {"set": args.set, "runs": []}
    for short in args.models:
        for lr in args.lrs:
            for seed in range(lo, hi + 1):
                model, kind = D.load(short, lr, seed, args.root, args.size, dev, D.N)
                if model is None:
                    print("missing", short, lr, seed, flush=True)
                    continue
                with torch.no_grad():
                    dq, dphi = bias_increments(model, kind, dev)
                    m = W.run_trace(model, kind, src_m, 1.0, 1.0, 2000, dev)
                    r = W.run_trace(model, kind, src_r, RF.RS, RF.RL, RF.STEPS, dev)
                u_l, u_c = r["uL"][:-1], r["uLc"][:-1]
                shift = best_shift(u_l, u_c, window_masks(*u_l.shape))
                rec = {"model": short, "lr": lr, "seed": seed, "dq0": dq, "dphi0": dphi,
                       "off_div_m": bool(m["div"][-1]), "dc_m": float(m["uL"][-1][-400:].mean()),
                       "div_r": r["div"][:-1].tolist(), "shift": [r6(row) for row in shift]}
                res["runs"].append(rec)
                print(short, lr, seed, f"dq0 {dq:.3e} dphi0 {dphi:.3e} dc_m {rec['dc_m']:.4e}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
