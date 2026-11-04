#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""保存系の追加（2026-10-15）：二重井戸・モース型・磁場中の荷電粒子・FPU 鎖と，記憶なし 3 段階の多自由度化のテスト.

pytest でも，`python tests/test_new_systems.py` でも実行できる。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harmonic_ic_dataset import DATASETS_DIR, save_manifest, split_scales, split_to_array  # noqa: E402
from harmonic_losses import lf_one_step_loss  # noqa: E402
from harmonic_models import MEMLESS_TARGET_PARAMS, MEMLESS_W2, build_model, hidden_size_of, memoryless_width  # noqa: E402
from systems import DEFAULT_REGIONS, make_system  # noqa: E402

NEW = ("doublewell", "morse", "charged", "fpu")


def test_energy_conserved():
    """参照解（刻みの細かい RK4）で 50 s のエネルギーの相対変化が 1e-9 未満."""
    rng = np.random.default_rng(0)
    for name in NEW:
        s = make_system(name)
        lo, hi = DEFAULT_REGIONS[name]["train"]
        ics = s.sample_ics(16, lo, hi, rng)
        ref = s.integrate(ics, 0.05, 1001)
        E = s.energy(ref)
        drift = np.abs(E - E[:, :1]).max() / max(np.abs(E[:, 0]).min(), 1e-3)
        assert drift < 1e-9, (name, drift)


def test_sampling_and_scales():
    rng = np.random.default_rng(1)
    for name in NEW:
        s = make_system(name)
        for lo, hi in (DEFAULT_REGIONS[name]["train"], DEFAULT_REGIONS[name]["extrap"]):
            ics = s.sample_ics(64, lo, hi, rng)
            rv = s.region_value(ics)
            assert ics.shape == (64, s.dim)
            assert (rv > lo - 1e-9).all() and (rv <= hi + 1e-9).all(), (name, rv.min(), rv.max())
        sc = s.scales(ics)
        assert sc.shape == ics.shape and (sc > 0).all()
        k = s.n_dof
        # 位置の尺度は位置の最大振れ以上（参照解の範囲で）
        ref = s.integrate(ics[:4], 0.05, 201)
        assert (np.abs(ref[:, :, :k]).max(axis=1) <= sc[:4, :k] * 1.001 + 1e-9).all(), name


def test_half_step_velocity_shapes():
    for name, k in (("doublewell", 1), ("charged", 2), ("fpu", 4)):
        s = make_system(name)
        ics = s.sample_ics(3, *DEFAULT_REGIONS[name]["train"], np.random.default_rng(2))
        v = s.half_step_velocity(ics, 0.05, 5)
        assert v.shape == ((3, 5) if k == 1 else (3, 5, k))


def test_charged_force_depends_on_velocity():
    """荷電粒子は同じ位置でも速度で加速度が変わる（③ の「力は位置だけ」では表せない）."""
    s = make_system("charged")
    a1 = s.rhs(np.array([[0.3, 0.2, 0.0, 0.0]]))[0, 2:]
    a2 = s.rhs(np.array([[0.3, 0.2, 0.5, -0.4]]))[0, 2:]
    assert np.abs(a1 - a2).max() > 0.1


def _lf_step(model, z, k):
    x, v = z[:, :k], z[:, k:]
    h = torch.zeros(z.shape[0], hidden_size_of(model), dtype=z.dtype)
    v1, h1, hp = model(torch.cat([x, v], 1), h, h, is_velocity_step=True)
    x1, _, _ = model(torch.cat([x, v1], 1), hp, h1, is_velocity_step=False)
    return torch.cat([x1, v1], 1)


def test_multidof_symplectic():
    """多自由度でも ③ はヤコビ行列が厳密にシンプレクティック（JᵀΩJ = Ω），② と ① は一般に違う."""
    torch.manual_seed(0)
    for dim in (4, 8):
        k = dim // 2
        Om = torch.zeros(dim, dim, dtype=torch.float64)
        Om[:k, k:], Om[k:, :k] = torch.eye(k), -torch.eye(k)
        for name, sym in (("LF_Sym_MLP", True), ("LF_Free_MLP", False)):
            m = build_model(name, 8, dim=dim).double()
            for p in m.parameters():
                torch.nn.init.normal_(p, std=1.0)
            z = torch.randn(dim, dtype=torch.float64)
            J = torch.autograd.functional.jacobian(lambda u: _lf_step(m, u.unsqueeze(0), k)[0], z)
            err = (J.T @ Om @ J - Om).abs().max().item()
            assert (err < 1e-10) if sym else (err > 1e-3), (name, dim, err)
        r = build_model("Res_MLP_state", 8, dim=dim)
        out, _ = r(torch.randn(3, dim), torch.zeros(3, 1))
        assert out.shape == (3, dim)


def test_lf_loss_same_for_1dof_shapes():
    """1 自由度で (B, T) と (B, T, 1) の損失が一致（既存の結果を変えない）."""
    torch.manual_seed(0)
    m = build_model("LF_Free_MLP", 8)
    x, v, a = torch.randn(4, 6), torch.randn(4, 6), torch.ones(4)
    l1 = lf_one_step_loss(m, x, v, a, 1.0, "mse")
    l2 = lf_one_step_loss(m, x.unsqueeze(-1), v.unsqueeze(-1), a, 1.0, "mse")
    assert torch.allclose(l1, l2)


def test_memoryless_width():
    for name, ws in MEMLESS_W2.items():
        for sz, w in ws.items():
            assert memoryless_width(name, sz, 2) == w
    for dim in (4, 8):
        for name in ("LF_Sym_MLP", "LF_Free_MLP", "Res_MLP_state"):
            for sz, target in MEMLESS_TARGET_PARAMS.items():
                w = memoryless_width(name, sz, dim)
                n = sum(p.numel() for p in build_model(name, w, dim=dim).parameters())
                assert abs(n - target) / target < 0.12, (name, dim, sz, w, n)


def _tiny_dataset(system: str, did: str) -> None:
    args = Namespace(system=system, dataset_id=did, n_train=8, n_val=4, n_test_interp=4, n_test_extrap=4,
                     train_lo=None, train_hi=None, extrap_lo=None, extrap_hi=None, omega=None, omega0=None, mu=None,
                     duff_alpha=None, duff_beta=None, duff_delta=None, fpu_n=None, bfield=None,
                     sigma=None, rho=None, beta=None, t_end=4.0, dt_direct=0.05, dt_lf=None, scale_floor=0.05, seed=1)
    import system_dataset
    save_manifest(system_dataset.build(args), DATASETS_DIR / did)


def test_train_eval_smoke():
    """データセット → 学習 → 評価（val も）が 4 系 × 3 段階で通る."""
    made = []
    try:
        for system in NEW:
            did = f"_test_new_{system}"
            _tiny_dataset(system, did)
            made.append(did)
            m = json.loads((DATASETS_DIR / did / "manifest.json").read_text())
            dim = m["dim"]
            assert split_to_array(m, "train").shape == (8, dim) and split_scales(m, "train").shape == (8, dim)
            for model in ("LF_Sym_MLP", "LF_Free_MLP", "Res_MLP_state"):
                hs = memoryless_width(model, "tiny", dim)
                base = [sys.executable, "-u"]
                r = subprocess.run(base + ["train_ic_generalization.py", "--model", model, "--dataset-id", did,
                                           "--run-id", "_t", "--hidden-size", str(hs), "--epochs", "2", "--batch-size", "4",
                                           "--lr", "1e-2", "--loss-type", "mse", "--seed", "0", "--no-progress"],
                                   cwd=ROOT, capture_output=True, text=True)
                assert r.returncode == 0, r.stderr[-1500:]
                for extra in ([], ["--splits", "val", "--out-subdir", "eval_val"]):
                    r = subprocess.run(base + ["ic_generalization_eval.py", "--model", model, "--hidden-size", str(hs),
                                               "--dataset-id", did, "--run-id", "_t", "--t-evals", "2", "--no-progress",
                                               *extra], cwd=ROOT, capture_output=True, text=True)
                    assert r.returncode == 0, r.stderr[-1500:]
                d = ROOT / "picture" / "ic_gen" / did / "_t" / model / f"hs{hs}"
                row = json.loads((d / "ic_generalization_summary.json").read_text())["rows"][0]
                assert np.isfinite(row["test_interp_x_rmse_rel_mean"]), (system, model)
                assert "val_x_rmse_rel_mean" in json.loads((d / "eval_val" / "ic_generalization_summary.json").read_text())["rows"][0]
    finally:
        for did in made:
            shutil.rmtree(DATASETS_DIR / did, ignore_errors=True)
            shutil.rmtree(ROOT / "picture" / "ic_gen" / did, ignore_errors=True)


def test_perfect_lf_model_multidof():
    """FPU 鎖で，力に真の力を入れた ③ は参照解に刻み Δt² の精度で一致する（多自由度のロールアウトの確認）."""
    import system_eval

    s = make_system("fpu")
    k = s.n_dof

    class Exact(torch.nn.Module):
        hidden_size, n_dof, delta_t = 1, k, 0.01

        def forward(self, x, hpp, hp, is_velocity_step):
            q, v = x[:, :k].double().cpu().numpy(), x[:, k:].double().cpu().numpy()
            if is_velocity_step:
                a = s.rhs(np.concatenate([q, v], 1))[:, k:]
                out = v + self.delta_t * a
            else:
                out = q + self.delta_t * v
            return torch.as_tensor(out, dtype=torch.float32, device=x.device), hpp, hp

    ics = s.sample_ics(4, 0.5, 1.0, np.random.default_rng(3))
    dt, steps = 0.01, 501
    v0 = s.half_step_velocity(ics, dt, 1)[:, 0]
    xs, vs = system_eval.rollout_lf_multi(Exact(), ics[:, :k], v0, steps)
    ref = s.integrate(ics, dt, steps)
    err = np.abs(xs - ref[:, :, :k]).max()
    assert err < 5e-3, err


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
