"""学習した段を古典のリープフロッグの線路に埋め込む試験（scripts/hybrid_line_20261021.py）の仕組みの確かめ（2026-10-21 夜）."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import hybrid_line_20261021 as H  # noqa: E402


ExactStag, ExactFull = H.ExactStag, H.ExactFull


def _ics(n_ic=16, n=32, seed=0):
    rng = np.random.default_rng(seed)
    from systems import ChargeFluxLine
    sysm = ChargeFluxLine(n)
    ics = sysm.sample_ics(n_ic, 0.05, 0.4, rng)
    fh = sysm.half_step_velocity(ics, 0.05, 1)[:, 0].reshape(n_ic, n)
    return torch.as_tensor(ics[:, :n]), torch.as_tensor(fh), torch.as_tensor(ics[:, n:])


def test_exact_staggered_block_reproduces_classical():
    q0, fh0, f0 = _ics()
    mask = torch.zeros(32, dtype=torch.bool)
    mask[:8] = True
    r = H.run(ExactStag(), "stag", q0, fh0, f0, mask, 1000, 0.05)
    assert r["1000"]["dev"] < 1e-4  # 単精度の丸めだけ
    rc = H.run(None, "stag", q0, fh0, f0, mask, 1000, 0.05)
    assert abs(r["1000"]["drift"] - rc["1000"]["drift"]) < 1e-4


def test_exact_full_step_block_is_close_and_classical_drift_is_small():
    q0, fh0, f0 = _ics()
    mask = torch.zeros(32, dtype=torch.bool)
    mask[:8] = True
    r = H.run(ExactFull(), "full", q0, fh0, f0, mask, 1000, 0.05)
    assert r["1000"]["dev"] < 0.05  # つなぎ方の差は小さい
    rc = H.run(None, "stag", q0, fh0, f0, mask, 2000, 0.05)
    assert rc["2000"]["drift"] < 1e-2 and rc["1000"]["dev"] < 1e-4


def test_lossy_line_energy_rate_and_exact_block():
    """損失のある線路（nltlg）：dH/dt = −G Σ u(q)²。真の式のずらした部品は損失のある古典の解と一致する."""
    from systems import ChargeFluxLossyLine
    rng = np.random.default_rng(1)
    sysm = ChargeFluxLossyLine(32, G=0.03)
    ics = sysm.sample_ics(8, 0.05, 0.4, rng)
    h = 1e-6
    grad = np.stack([(sysm.energy(ics + h * e) - sysm.energy(ics - h * e)) / (2 * h) for e in np.eye(64)], axis=1)
    dHdt = (grad * sysm.rhs(ics)).sum(1)
    q = ics[:, :32]
    assert np.allclose(dHdt, -0.03 * ((q + q ** 3) ** 2).sum(1), rtol=1e-6, atol=1e-9)
    old = H.LOSS_G
    try:
        H.LOSS_G = 0.03
        fh = sysm.half_step_velocity(ics, 0.05, 1)[:, 0].reshape(8, 32)
        mask = torch.zeros(32, dtype=torch.bool)
        mask[:8] = True
        q0, fh0, f0 = torch.as_tensor(ics[:, :32]), torch.as_tensor(fh), torch.as_tensor(ics[:, 32:])
        r = H.run(ExactStag(), "stag", q0, fh0, f0, mask, 1000, 0.05)
        rc = H.run(None, "stag", q0, fh0, f0, mask, 1000, 0.05)
        assert r["1000"]["dev"] < 1e-4 and abs(r["1000"]["edev"] - rc["1000"]["edev"]) < 1e-6
    finally:
        H.LOSS_G = old


def test_learned_interface_and_coarse_step_with_exact_parts():
    """境目も部品で更新する版（--interface learned）と粗い刻み（Δt = 0.2）でも，真の式のずらした部品は古典の解と一致し，
    輪の総電荷・総磁束はほぼ保存される（2026-10-21 夜）."""
    old = H.DT
    try:
        H.DT = 0.2
        q0, fh0, f0 = _ics()
        n = 32
        from systems import ChargeFluxLine
        sysm = ChargeFluxLine(n)
        fh0 = torch.as_tensor(sysm.half_step_velocity(np.concatenate([q0.numpy(), f0.numpy()], 1), 0.2, 1)[:, 0].reshape(len(q0), n))
        mask = torch.zeros(n, dtype=torch.bool)
        mask[:8] = True
        mask_x = mask.clone()
        mask_x[n - 1] = True
        mask_v = mask.clone()
        mask_v[8] = True
        old_h = H.HORIZONS
        H.HORIZONS = (50, 250)
        r = H.run(ExactStag(), "stag", q0, fh0, f0, mask_v, 250, 0.2, mask_x=mask_x)
        assert r["250"]["dev"] < 1e-4
        assert r["250"]["dq"] < 1e-5 and r["250"]["dphi"] < 1e-5
        rf = H.run(ExactFull(), "full", q0, fh0, f0, mask_v, 250, 0.2, mask_x=mask_x)
        assert np.isfinite(rf["250"]["drift"])
    finally:
        H.DT = old
        H.HORIZONS = old_h
