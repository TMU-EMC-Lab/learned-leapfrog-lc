"""時刻をまとめて計算する 1 ステップ損失（--fast-onestep）が，元の時刻のループの損失と値・勾配とも一致するか（2026-10-21）."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harmonic_losses as HL  # noqa: E402
from harmonic_models import build_model  # noqa: E402

LF_MODELS = ["LF_Free_MLP", "LF_Par_MLP", "LF_Kick_MLP", "LF_Sym_MLP", "LF_Sep_MLP", "LF_Cen_MLP"]
STATE_MODELS = ["Res_MLP_state", "RK4_MLP_state", "Res2_MLP_state", "SIE_MLP_state", "SIEMid_MLP_state", "RK4Kin_MLP_state"]


def _grads(model, loss):
    model.zero_grad()
    loss.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


@pytest.mark.parametrize("name", LF_MODELS)
@pytest.mark.parametrize("dim", [2, 4])
@pytest.mark.parametrize("loss_type", ["mse", "amp_norm_mse"])
def test_lf_flat_matches_loop(name, dim, loss_type):
    torch.manual_seed(0)
    model = build_model(name, 8, dim=dim).double()
    assert HL.supports_flat_onestep(model)
    k = dim // 2
    x = torch.randn(5, 7, k, dtype=torch.float64)
    v = torch.randn(5, 7, k, dtype=torch.float64)
    if k == 1:
        x, v = x[..., 0], v[..., 0]
    amp = torch.rand(5, dtype=torch.float64) + 0.5
    l0 = HL.lf_one_step_loss(model, x, v, amp, 1.3, loss_type)
    g0 = _grads(model, l0)
    l1 = HL.lf_one_step_loss_flat(model, x, v, amp, 1.3, loss_type)
    g1 = _grads(model, l1)
    assert torch.allclose(l0, l1, rtol=1e-12, atol=1e-14)
    assert torch.allclose(g0, g1, rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("name", STATE_MODELS)
@pytest.mark.parametrize("dim", [2, 4])
@pytest.mark.parametrize("loss_type", ["mse", "amp_norm_mse"])
@pytest.mark.parametrize("pos_w", [1.0, 100.0])
def test_state_flat_matches_loop(name, dim, loss_type, pos_w, monkeypatch):
    monkeypatch.setattr(HL, "POS_LOSS_WEIGHT", pos_w)
    torch.manual_seed(1)
    model = build_model(name, 8, dim=dim).double()
    assert HL.supports_flat_onestep(model)
    s = torch.randn(5, 7, dim, dtype=torch.float64)
    scale = torch.rand(5, dim, dtype=torch.float64) + 0.5
    l0 = HL.state_one_step_loss(model, s, scale, loss_type)
    g0 = _grads(model, l0)
    l1 = HL.state_one_step_loss_flat(model, s, scale, loss_type)
    g1 = _grads(model, l1)
    assert torch.allclose(l0, l1, rtol=1e-12, atol=1e-14)
    assert torch.allclose(g0, g1, rtol=1e-10, atol=1e-13)


def test_recurrent_models_are_not_flattened():
    assert not HL.supports_flat_onestep(build_model("LF_LTCs", 8))


def test_charge_flux_circuit_matches_voltage_current():
    """電荷・磁束の版（nllcqf）は，電圧・電流の版（nllc）と同じ初期条件・同じ軌道になる（q = arctan V，φ = arctan I）."""
    import numpy as np
    from systems import make_system

    vi_sys, qf_sys = make_system("nllc"), make_system("nllcqf")
    vi = vi_sys.sample_ics(64, 0.0025, 1.0, np.random.default_rng(42))
    qf = qf_sys.sample_ics(64, 0.0025, 1.0, np.random.default_rng(42))
    assert np.allclose(qf, np.arctan(vi))
    assert np.allclose(qf_sys.energy(qf), vi_sys.energy(vi))
    tv = vi_sys.integrate(vi, 0.05, 201)
    tq = qf_sys.integrate(qf, 0.05, 201)
    assert np.allclose(np.tan(tq), tv, rtol=1e-7, atol=1e-8)
    E = qf_sys.energy(tq)
    assert np.abs(E - E[:, :1]).max() < 1e-8
    sc = qf_sys.scales(qf)
    assert (np.abs(tq[:, :, 0]).max(axis=1) <= sc[:, 0] + 1e-9).all()


def test_cubic_charge_flux_circuit():
    """3 次の電荷・磁束の回路（nlqf3）：エネルギーの保存，領域，尺度."""
    import numpy as np
    from systems import make_system

    sysm = make_system("nlqf3")
    ics = sysm.sample_ics(256, 0.0025, 1.0, np.random.default_rng(0))
    E0 = sysm.energy(ics)
    assert (E0 > 0.0025).all() and (E0 <= 1.0).all()
    traj = sysm.integrate(ics, 0.05, 201)
    E = sysm.energy(traj)
    assert np.abs(E - E[:, :1]).max() < 1e-6
    sc = sysm.scales(ics)
    assert (np.abs(traj[:, :, 0]).max(axis=1) <= sc[:, 0] + 1e-6).all()
    assert (np.abs(traj[:, :, 1]).max(axis=1) <= sc[:, 1] + 1e-6).all()
    assert abs(float(sysm._amp(1.0)) - (5 ** 0.5 - 1) ** 0.5) < 1e-12


def test_lf_sep_is_area_preserving_and_matches_size():
    """LF-Sep：1 自由度の 1 ステップの写像の行列式は 1（ずれの合成），パラメータ数は LF の 5% 以内."""
    torch.manual_seed(3)
    m = build_model("LF_Sep_MLP", 16).double()
    lf = build_model("LF_Free_MLP", 16)

    def step(z):
        v = m(z.view(1, 2), None, None, is_velocity_step=True)[0]
        x = m(torch.cat([z[:1], v[0]]).view(1, 2), None, None, is_velocity_step=False)[0]
        return torch.cat([x[0], v[0]])

    for z in torch.randn(5, 2, dtype=torch.float64):
        J = torch.autograd.functional.jacobian(step, z)
        assert abs(float(torch.det(J)) - 1.0) < 1e-12
    n_sep, n_lf = (sum(p.numel() for p in mm.parameters()) for mm in (m, lf))
    assert abs(n_sep - n_lf) / n_lf < 0.05


def test_lf_cen_solves_midpoint_equations_and_reduces_to_lf():
    """LF-C：反復のあとの出力が陰的な式（自分の変数も中点）をほぼ満たす。反復 0 回なら同じ重みの LF と一致する."""
    torch.manual_seed(4)
    m = build_model("LF_Cen_MLP", 16).double()
    z = torch.randn(64, 2, dtype=torch.float64) * 0.8
    dt = m.delta_t
    v_new = m(z, None, None, is_velocity_step=True)[0]
    res_v = v_new - z[:, 1:] - dt * m.F(torch.cat([z[:, :1], 0.5 * (z[:, 1:] + v_new)], dim=1))
    zx = torch.cat([z[:, :1], v_new], dim=1)
    x_new = m(zx, None, None, is_velocity_step=False)[0]
    res_x = x_new - z[:, :1] - dt * m.G(torch.cat([0.5 * (z[:, :1] + x_new), v_new], dim=1))
    assert float(res_v.abs().max()) < 1e-6 * float((v_new - z[:, 1:]).abs().max())
    assert float(res_x.abs().max()) < 1e-6 * float((x_new - z[:, :1]).abs().max())
    lf = build_model("LF_Free_MLP", 16).double()
    lf.load_state_dict(m.state_dict())
    m.n_iter = 0
    for flag, inp in ((True, z), (False, zx)):
        assert torch.allclose(m(inp, None, None, is_velocity_step=flag)[0], lf(inp, None, None, is_velocity_step=flag)[0], rtol=0, atol=1e-14)
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in lf.parameters())


LOCAL_LF = ["LF_Free_Loc", "LF_Par_Loc", "LF_Sep_Loc", "LF_SepD_Loc", "LF_SepDx_Loc"]
LOCAL_STATE = ["Res2_Loc_state", "SIE_Loc_state", "SIEMid_Loc_state", "Res_Loc_state", "RK4_Loc_state"]


@pytest.mark.parametrize("name", LOCAL_LF)
def test_local_lf_flat_and_shift(name):
    """段ごとの網：時刻をまとめた損失が元と一致し，段を環状にずらすと出力も同じだけずれる."""
    torch.manual_seed(4)
    model = build_model(name, 8, dim=16).double()
    x, v = torch.randn(3, 6, 8, dtype=torch.float64), torch.randn(3, 6, 8, dtype=torch.float64)
    amp = torch.rand(3, dtype=torch.float64) + 0.5
    l0 = HL.lf_one_step_loss(model, x, v, amp, 1.0, "mse")
    l1 = HL.lf_one_step_loss_flat(model, x, v, amp, 1.0, "mse")
    assert torch.allclose(l0, l1, rtol=1e-12)
    z = torch.cat([x[:, 0], v[:, 0]], 1)
    zs = torch.cat([torch.roll(x[:, 0], 3, 1), torch.roll(v[:, 0], 3, 1)], 1)
    o = model(z, None, None, is_velocity_step=True)[0]
    os_ = model(zs, None, None, is_velocity_step=True)[0]
    assert torch.allclose(torch.roll(o, 3, 1), os_)


@pytest.mark.parametrize("name", LOCAL_STATE)
def test_local_state_flat_and_length(name):
    """段ごとの網の状態入力の型：時刻をまとめた損失が元と一致し，同じ重みで 16 段の線路にも使える."""
    torch.manual_seed(5)
    model = build_model(name, 8, dim=16).double()
    s = torch.randn(3, 6, 16, dtype=torch.float64)
    sc = torch.rand(3, 16, dtype=torch.float64) + 0.5
    assert torch.allclose(HL.state_one_step_loss(model, s, sc, "mse"), HL.state_one_step_loss_flat(model, s, sc, "mse"), rtol=1e-12)
    z8 = torch.randn(2, 16, dtype=torch.float64)
    out8, _ = model(z8, torch.zeros(2, 1, dtype=torch.float64))
    model.set_length(16)
    z16 = torch.cat([z8[:, :8], z8[:, :8], z8[:, 8:], z8[:, 8:]], 1)  # 8 段の状態を 2 周並べた 16 段
    out16, _ = model(z16, torch.zeros(2, 1, dtype=torch.float64))
    assert torch.allclose(out16, torch.cat([out8[:, :8], out8[:, :8], out8[:, 8:], out8[:, 8:]], 1))


def test_nltl_system_conserves():
    """非線形伝送線路：エネルギー・全電荷・全磁束の保存と，1 段あたりのエネルギーの範囲."""
    import numpy as np
    from systems import make_system

    sysm = make_system("nltl", n=8)
    ics = sysm.sample_ics(32, 0.0025, 0.5, np.random.default_rng(0))
    e = sysm.region_value(ics)
    assert (e > 0.0025 - 1e-12).all() and (e <= 0.5 + 1e-12).all()
    traj = sysm.integrate(ics, 0.05, 201)
    E = sysm.energy(traj)
    assert np.abs(E - E[:, :1]).max() < 1e-6 * max(1.0, float(np.abs(E).max()))
    for sl in (slice(0, 8), slice(8, 16)):
        tot = traj[:, :, sl].sum(-1)
        assert np.abs(tot - tot[:, :1]).max() < 1e-9



def test_lf_sepd_solves_midpoint_loss_term():
    """LF-SepD：電荷の更新が q_{n+1} = q_n + Δt[G(φ) + D((q_n + q_{n+1})/2)] をほぼ満たす（反復 3 回）."""
    torch.manual_seed(5)
    m = build_model("LF_SepD_Loc", 12, dim=16).double()
    z = torch.randn(32, 16, dtype=torch.float64) * 0.6
    q_new = m(z, None, None, is_velocity_step=False)[0]
    q, f = z[:, :8], z[:, 8:]
    res = q_new - q - m.delta_t * (m.G(f) + m.D(0.5 * (q + q_new)))
    assert float(res.abs().max()) < 1e-6 * float((q_new - q).abs().max())


def test_lf_sepdx_takes_loss_term_at_start():
    """LF-SepDx：電荷の更新が q_{n+1} = q_n + Δt[G(φ) + D(q_n)]（陽的）．網とパラメータ数は LF-SepD と同じ."""
    torch.manual_seed(6)
    m = build_model("LF_SepDx_Loc", 12, dim=16).double()
    z = torch.randn(32, 16, dtype=torch.float64) * 0.6
    q_new = m(z, None, None, is_velocity_step=False)[0]
    q, f = z[:, :8], z[:, 8:]
    torch.testing.assert_close(q_new, q + m.delta_t * (m.G(f) + m.D(q)))
    n = lambda name: sum(p.numel() for p in build_model(name, 12, dim=16).parameters())
    assert n("LF_SepDx_Loc") == n("LF_SepD_Loc")
