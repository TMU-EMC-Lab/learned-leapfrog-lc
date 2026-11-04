#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""記憶なし・増分型のモデル（2026-10-09）のテスト.

pytest でも，`python tests/test_memoryless.py` でも実行できる。
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harmonic_ic_dataset import DATASETS_DIR, save_manifest  # noqa: E402
from harmonic_models import build_model, hidden_size_of  # noqa: E402

LF_NAMES = ("LF_Sym_MLP", "LF_Sym_LTC", "LF_Free_MLP", "LF_Free_LTC")
STATE_NAMES = ("Res_MLP_state", "Res_LTC_state")


def _lf_step(model, z):
    """LF 系の 1 ステップ (x_n, v_{n−1/2}) → (x_{n+1}, v_{n+1/2})（ロールアウトと同じ呼び方）."""
    x, v = z[:, :1], z[:, 1:2]
    h = torch.zeros(z.shape[0], hidden_size_of(model))
    v1, h1, hp = model(torch.cat([x, v], 1), h, h, is_velocity_step=True)
    x1, _, _ = model(torch.cat([x, v1], 1), hp, h1, is_velocity_step=False)
    return torch.cat([x1, v1], 1)


def _jac_det(f, z):
    J = torch.autograd.functional.jacobian(lambda u: f(u.unsqueeze(0))[0], z)
    return torch.det(J).item()


def test_symplectic_det_one():
    """③ LF_Sym_* はどんな重みでもヤコビ行列の行列式が 1（自由度 1 の厳密なシンプレクティック性）."""
    torch.manual_seed(0)
    for name in ("LF_Sym_MLP", "LF_Sym_LTC"):
        for _ in range(3):
            m = build_model(name, 8).double()
            for p in m.parameters():
                torch.nn.init.normal_(p, std=1.0)  # 学習前の初期値でなく，任意の重みでも成り立つことを見る
            for z in torch.randn(5, 2, dtype=torch.float64) * 2:
                d = _jac_det(lambda u: _lf_step(m, u), z)
                assert abs(d - 1.0) < 1e-10, (name, d)


def test_others_not_symplectic():
    """② LF_Free_* と ① Res_*_state は一般に行列式が 1 にならない（比較の対照として正しい）."""
    torch.manual_seed(1)
    for name in ("LF_Free_MLP", "LF_Free_LTC"):
        m = build_model(name, 8).double()
        for p in m.parameters():
            torch.nn.init.normal_(p, std=1.0)
        dets = [_jac_det(lambda u: _lf_step(m, u), z) for z in torch.randn(5, 2, dtype=torch.float64)]
        assert max(abs(d - 1) for d in dets) > 1e-4, (name, dets)
    for name in STATE_NAMES:
        m = build_model(name, 8).double()
        for p in m.parameters():
            torch.nn.init.normal_(p, std=1.0)
        f = lambda u: m(u, torch.zeros(u.shape[0], hidden_size_of(m), dtype=u.dtype))[0]  # noqa: E731
        dets = [_jac_det(f, z) for z in torch.randn(5, 2, dtype=torch.float64)]
        assert max(abs(d - 1) for d in dets) > 1e-4, (name, dets)


def test_harmonic_exact_kick():
    """③ に調和振動子の真の力 F(x) = −x を入れると Störmer–Verlet 法そのものになる（位置の段は x + Δt·v）."""
    m = build_model("LF_Sym_MLP", 4)
    class _Neg(torch.nn.Module):
        def forward(self, x):
            return -x

    m.F = _Neg()
    z = torch.tensor([[0.7, 0.2]])
    out = _lf_step(m, z)
    dt = m.delta_t
    v1 = 0.2 - dt * 0.7
    assert torch.allclose(out, torch.tensor([[0.7 + dt * v1, v1]]), atol=1e-7)


def test_no_memory():
    """隠れ状態を持たない：返す隠れ状態は常に 0，出力は前のステップに依存しない."""
    torch.manual_seed(2)
    for name in LF_NAMES:
        m = build_model(name, 6)
        x = torch.randn(4, 2)
        h = torch.randn(4, hidden_size_of(m))
        o1, h1, _ = m(x, h, h, is_velocity_step=True)
        o2, _, _ = m(x, torch.zeros_like(h), torch.zeros_like(h), is_velocity_step=True)
        assert torch.allclose(o1, o2) and torch.count_nonzero(h1) == 0
    for name in STATE_NAMES:
        m = build_model(name, 6)
        xv = torch.randn(4, 2)
        o1, h1 = m(xv, torch.randn(4, hidden_size_of(m)))
        o2, _ = m(xv, torch.zeros(4, hidden_size_of(m)))
        assert torch.allclose(o1, o2) and torch.count_nonzero(h1) == 0


def test_train_eval_smoke():
    """調和振動子（刻み 0.05，解析解の教師）で学習・評価が最後まで通る."""
    src = DATASETS_DIR / "_ampmin_smoke" / "manifest.json"
    if not src.exists():
        print("  (skip: datasets/_ampmin_smoke なし)")
        return
    base = json.loads(src.read_text())
    m = dict(base, dataset_id="_test_memless_dtd2", dt_direct=0.05)
    did = m["dataset_id"]
    try:
        save_manifest(m, DATASETS_DIR / did)
        for name in LF_NAMES + STATE_NAMES:
            hs = build_model(name, 8) and {"LF_Sym_MLP": 25, "LF_Sym_LTC": 24, "LF_Free_MLP": 16, "LF_Free_LTC": 16,
                                           "Res_MLP_state": 24, "Res_LTC_state": 23}[name]
            r = subprocess.run([sys.executable, "-u", "train_ic_generalization.py", "--model", name, "--dataset-id", did,
                                "--run-id", "_t", "--epochs", "2", "--batch-size", "128", "--lr", "1e-2", "--loss-type", "mse",
                                "--teacher", "analytic", "--seed", "0", "--no-progress"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, (name, r.stderr[-1500:])
            r = subprocess.run([sys.executable, "-u", "ic_generalization_eval.py", "--model", name, "--dataset-id", did,
                                "--run-id", "_t", "--t-evals", "2", "--no-progress"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, (name, r.stderr[-1500:])
            d = ROOT / "picture" / "ic_gen" / did / "_t" / name / f"hs{hs}"
            row = json.loads((d / "ic_generalization_summary.json").read_text())["rows"][0]
            assert math.isfinite(row["test_interp_x_rmse_rel_mean"]), (name, row)
            cfg = json.loads((d / "config.json").read_text())
            assert cfg["teacher"] == "analytic" and abs(cfg["dt"] - 0.05) < 1e-12, cfg
    finally:
        shutil.rmtree(DATASETS_DIR / did, ignore_errors=True)
        shutil.rmtree(ROOT / "picture" / "ic_gen" / did, ignore_errors=True)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
