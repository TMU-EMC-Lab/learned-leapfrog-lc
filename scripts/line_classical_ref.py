#!/usr/bin/env python3
"""伝送線路（nltl）で，真の特性 v(q) = q + q³，i(φ) = φ + φ³ をそのまま入れた古典のリープフロッグ（時間方向は 1 次元の FDTD と同じ組み立て）
のロールアウトの誤差を，学習したモデルと同じ評価（system_eval.evaluate，Δt = 0.05，単精度）で測る（比較の基準）.

  BHIC_NO_REF_CACHE=1 python scripts/line_classical_ref.py [--out OUT.json]

初期の磁束 φ_{−1/2} は学習したモデルの評価と同じく，参照解を半ステップ戻して作る（system_eval.initial_states）。
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
sys.path.insert(0, os.path.join(HERE, ".."))

import system_eval as se  # noqa: E402


class ExactLeapfrog(nn.Module):
    """φ_{n+1/2} = φ_{n−1/2} + Δt [v(q_{k−1}) − v(q_k)]，q_{n+1} = q_n + Δt [i(φ_k) − i(φ_{k+1})]（輪）."""

    hidden_size = 1

    def __init__(self, n: int, dt: float) -> None:
        super().__init__()
        self.n, self.dt = n, dt

    def forward(self, x, h_pp, h_p, is_velocity_step):
        q, w = x[:, :self.n], x[:, self.n:]
        if is_velocity_step:
            vq = q + q ** 3
            out = w + self.dt * (torch.roll(vq, 1, dims=1) - vq)
        else:
            iw = w + w ** 3
            out = q + self.dt * (iw - torch.roll(iw, -1, dims=1))
        return out, h_p, h_pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    for ds, splits in (("nltl8_v1_dtd2", ["test_interp", "test_extrap"]), ("nltl16_v1_dtd2", ["test_interp"]),
                       ("nltl32_v1_dtd2", ["test_interp"])):
        manifest = json.load(open(os.path.join(HERE, "..", "datasets", ds, "manifest.json")))
        model = ExactLeapfrog(int(manifest["params"]["n"]), float(manifest["dt_lf"])).to(se.ev.DEVICE)
        with tempfile.TemporaryDirectory() as d:
            rows = se.evaluate(model, "lf", manifest, splits, [10.0, 20.0, 50.0], Path(d), {}, teacher=None)
        res[ds] = {r["t_eval_tag"]: {f"{s}_x_rmse_rel_mean": r[f"{s}_x_rmse_rel_mean"] for s in splits} for r in rows}
        print(ds, json.dumps(res[ds]["t50s"]), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
