#!/usr/bin/env python3
"""真の特性を入れた古典のリープフロッグ（シュテルマー・ベルレ法）の t = 50 の誤差を，
刻みを変えたデータセットで学習したモデルと同じ評価（system_eval.evaluate，単精度）で測る.

  python scripts/dt_classical_ref_20261021.py DATASET [DATASET ...] [--out OUT.json]

nltl（輪の線路）は line_classical_ref.ExactLeapfrog，nlqf3（3 次の回路，1 自由度）は φ_{n+1/2} = φ_{n−1/2} − Δt u(q_n)，q_{n+1} = q_n + Δt i(φ_{n+1/2})。
出力は t = 10，20，50 ごとの誤差の平均と，発散した初期条件の数（誤差が有限でないか 100 を超えたもの。2026-10-29 に足した）。
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import system_eval as se  # noqa: E402
from line_classical_ref import ExactLeapfrog  # noqa: E402


class ExactCubic(nn.Module):
    hidden_size = 1

    def __init__(self, dt: float) -> None:
        super().__init__()
        self.dt = dt

    def forward(self, x, h_pp, h_p, is_velocity_step):
        q, w = x[:, :1], x[:, 1:]
        if is_velocity_step:
            return w - self.dt * (q + q ** 3), h_p, h_pp
        return q + self.dt * (w + w ** 3), h_p, h_pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    for ds in args.datasets:
        manifest = json.load(open(os.path.join(HERE, "..", "datasets", ds, "manifest.json")))
        dt = float(manifest["dt_lf"])
        model = (ExactLeapfrog(int(manifest["params"]["n"]), dt) if manifest["system"] == "nltl" else ExactCubic(dt)).to(se.ev.DEVICE)
        splits = ["test_interp"] + (["test_extrap"] if not ds.startswith("nltl32") else [])
        with tempfile.TemporaryDirectory() as d:
            rows = se.evaluate(model, "lf", manifest, splits, [10.0, 20.0, 50.0], Path(d), {}, teacher=None)
        keys = [f"{s}_x_rmse_rel_{k}" for s in splits for k in ("mean", "n_diverged")]
        res[ds] = {r["t_eval_tag"]: {k: r[k] for k in keys} for r in rows}
        print(ds, json.dumps(res[ds]["t50s"]), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
