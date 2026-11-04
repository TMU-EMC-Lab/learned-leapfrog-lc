#!/usr/bin/env python3
"""真の特性の古典のリープフロッグで環の線路を回し，発散する試験の初期条件を数える.

  python scripts/classical_div_20261022.py nltl32_v1_dt0p2 nltl32_v1_dt0p1 nltl8_v1_dt0p2 > classical_div_20261022.txt

φ_{−1/2} は参照の式を刻み 1e-4 の RK4 で半ステップ前まで積分して得る（本文の評価と同じ始め方）。
発散は，状態が有限でないか max|q| > 100 になったもの。t = 50 と t = 500 の時点で数える。
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def u(q):
    return q + q ** 3


def i(p):
    return p + p ** 3


def rhs(q, p):
    dq = i(p) - np.roll(i(p), -1, axis=1)  # q̇_k = i(φ_k) − i(φ_{k+1})
    dp = np.roll(u(q), 1, axis=1) - u(q)  # φ̇_k = u(q_{k−1}) − u(q_k)
    return dq, dp


def rk4(q, p, T, h=1e-4):
    m = int(round(abs(T) / h))
    h = T / m
    for _ in range(m):
        k1 = rhs(q, p)
        k2 = rhs(q + h / 2 * k1[0], p + h / 2 * k1[1])
        k3 = rhs(q + h / 2 * k2[0], p + h / 2 * k2[1])
        k4 = rhs(q + h * k3[0], p + h * k3[1])
        q = q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        p = p + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return q, p


def count(ds):
    man = json.load(open(os.path.join(ROOT, "datasets", ds, "manifest.json")))
    n, dt = man["params"]["n"], float(man["dt_lf"])
    assert man["var_names"][0] == "q0" and len(man["var_names"]) == 2 * n
    X = np.array(man["splits"]["test_interp"]["ics"], dtype=np.float64)
    q, p0 = X[:, :n].copy(), X[:, n:].copy()
    _, ph = rk4(q.copy(), p0, -dt / 2)  # φ_{−1/2}
    div = np.zeros(len(q), bool)
    out = {}
    with np.errstate(all="ignore"):
        for step in range(1, int(round(500 / dt)) + 1):
            ph = ph + dt * (np.roll(u(q), 1, axis=1) - u(q))
            q = q + dt * (i(ph) - np.roll(i(ph), -1, axis=1))
            div |= ~np.isfinite(q).all(1) | (np.abs(q).max(1) > 100)
            q[div], ph[div] = 0.0, 0.0
            t = step * dt
            if abs(t - 50) < 1e-9 or abs(t - 500) < 1e-9:
                out[f"t{t:.0f}"] = int(div.sum())
                print(f"{ds} dt={dt} t={t:.0f}: diverged {div.sum()} of {len(div)}", flush=True)
    return out


if __name__ == "__main__":
    res = {ds: count(ds) for ds in sys.argv[1:]}
    print(json.dumps(res))
