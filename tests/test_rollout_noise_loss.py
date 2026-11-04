"""学習の手順を変えた対照（2026-10-22）：K ステップのロールアウトの損失と，入力の雑音の試験."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harmonic_losses as HL  # noqa: E402
from harmonic_models import build_model  # noqa: E402

LF_MODELS = ["LF_Free_MLP", "LF_Sep_MLP", "LF_Free_Loc", "LF_Sep_Loc"]
STATE_MODELS = ["Res_MLP_state", "RK4_MLP_state", "SIE_MLP_state", "SIEMid_MLP_state", "Res_Loc_state", "SIEMid_Loc_state"]


def _dim(name):
    return 16 if "_Loc" in name else 2


@pytest.mark.parametrize("name", LF_MODELS)
@pytest.mark.parametrize("k_steps", [1, 3])
def test_lf_rollout_matches_manual(name, k_steps, monkeypatch):
    monkeypatch.setattr(HL, "ROLLOUT_STEPS", k_steps)
    torch.manual_seed(0)
    dim = _dim(name)
    k = dim // 2
    model = build_model(name, 8, dim=dim).double()
    x = torch.randn(3, 8, k, dtype=torch.float64)
    v = torch.randn(3, 8, k, dtype=torch.float64)
    amp = torch.ones(3, dtype=torch.float64)
    got = HL.lf_rollout_loss(model, x, v, amp, 1.0, "mse")
    # 手で：区間の始め 0, K, 2K, ...（s + K ≤ 7）から自由に K ステップ進める
    total, starts = 0.0, list(range(0, 8 - k_steps, k_steps))
    for j in range(1, k_steps + 1):
        errs = []
        for s in starts:
            xx, vv = x[:, s], v[:, s]
            for _ in range(j):
                vv = model(torch.cat([xx, vv], 1), None, None, is_velocity_step=True)[0]
                xx = model(torch.cat([xx, vv], 1), None, None, is_velocity_step=False)[0]
            errs.append((((vv - v[:, s + j]) ** 2).mean(), ((xx - x[:, s + j]) ** 2).mean()))
        mv = torch.stack([e[0] for e in errs]).mean()
        mx = torch.stack([e[1] for e in errs]).mean()
        total = total + 0.5 * (mv + mx)
    assert torch.allclose(got, total / k_steps, rtol=1e-10)


@pytest.mark.parametrize("name", STATE_MODELS)
@pytest.mark.parametrize("k_steps", [1, 4])
def test_state_rollout_matches_manual(name, k_steps, monkeypatch):
    monkeypatch.setattr(HL, "ROLLOUT_STEPS", k_steps)
    torch.manual_seed(1)
    dim = _dim(name)
    model = build_model(name, 8, dim=dim).double()
    s = torch.randn(2, 9, dim, dtype=torch.float64)
    scale = torch.ones(2, dim, dtype=torch.float64)
    got = HL.state_rollout_loss(model, s, scale, "mse")
    total, starts = 0.0, list(range(0, 9 - k_steps, k_steps))
    for j in range(1, k_steps + 1):
        errs = []
        for st in starts:
            z = s[:, st]
            for _ in range(j):
                z = model(z, torch.zeros(z.shape[0], 1, dtype=z.dtype))[0]
            errs.append(((z - s[:, st + j]) ** 2).mean())
        total = total + torch.stack(errs).mean()
    assert torch.allclose(got, total / k_steps, rtol=1e-10)


def test_noise_only_in_training_and_zero_noise_unchanged(monkeypatch):
    torch.manual_seed(2)
    model = build_model("LF_Free_MLP", 8, dim=2).double()
    x, v = torch.randn(4, 6, dtype=torch.float64), torch.randn(4, 6, dtype=torch.float64)
    amp = torch.ones(4, dtype=torch.float64)
    base = HL.lf_one_step_loss_flat(model, x, v, amp, 1.0, "mse")
    monkeypatch.setattr(HL, "INPUT_NOISE", (torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)))
    model.train(True)
    assert torch.allclose(HL.lf_one_step_loss_flat(model, x, v, amp, 1.0, "mse"), base, rtol=1e-12)
    monkeypatch.setattr(HL, "INPUT_NOISE", (torch.full((1,), 0.1, dtype=torch.float64), torch.full((1,), 0.1, dtype=torch.float64)))
    model.train(False)
    assert torch.allclose(HL.lf_one_step_loss_flat(model, x, v, amp, 1.0, "mse"), base, rtol=1e-12)  # 検証では雑音なし
    model.train(True)
    assert not torch.allclose(HL.lf_one_step_loss_flat(model, x, v, amp, 1.0, "mse"), base, rtol=1e-6)
    sm = build_model("SIE_MLP_state", 8, dim=2).double()
    s, sc = torch.randn(4, 6, 2, dtype=torch.float64), torch.ones(4, 2, dtype=torch.float64)
    sm.train(False)
    b2 = HL.state_one_step_loss_flat(sm, s, sc, "mse")
    monkeypatch.setattr(HL, "INPUT_NOISE", torch.full((2,), 0.1, dtype=torch.float64))
    assert torch.allclose(HL.state_one_step_loss_flat(sm, s, sc, "mse"), b2, rtol=1e-12)
    sm.train(True)
    assert not torch.allclose(HL.state_one_step_loss_flat(sm, s, sc, "mse"), b2, rtol=1e-6)


@pytest.mark.parametrize("name", ["LF_KCL_Loc", "LF_KCLD_Loc"])
def test_kcl_line_conserves_and_matches_classical_form(name):
    """LF_KCL_Loc は総電荷・総磁束を厳密に保存し，特性を真の式に差し替えると古典のリープフロッグと一致する（2026-10-22）."""
    import harmonic_models as HM
    torch.manual_seed(3)
    n = 8
    model = build_model(name, 6, dim=2 * n).double()
    q, f = torch.randn(4, n, dtype=torch.float64), torch.randn(4, n, dtype=torch.float64)
    f1 = model(torch.cat([q, f], 1), None, None, is_velocity_step=True)[0]
    q1 = model(torch.cat([q, f1], 1), None, None, is_velocity_step=False)[0]
    assert torch.allclose(f1.sum(1), f.sum(1), atol=1e-12)
    if name == "LF_KCL_Loc":
        assert torch.allclose(q1.sum(1), q.sum(1), atol=1e-12)
        # 真の特性 u(q) = q + q³，i(φ) = φ + φ³ に差し替えると古典のリープフロッグ
        model.U.forward = lambda u: u + u ** 3
        model.I.forward = lambda u: u + u ** 3
        dt = model.delta_t
        f2 = model(torch.cat([q, f], 1), None, None, is_velocity_step=True)[0]
        q2 = model(torch.cat([q, f2], 1), None, None, is_velocity_step=False)[0]
        u = q + q ** 3
        fc = f + dt * (torch.roll(u, 1, 1) - u)
        i = fc + fc ** 3
        qc = q + dt * (i - torch.roll(i, -1, 1))
        assert torch.allclose(f2, fc) and torch.allclose(q2, qc)
    n_par = sum(p.numel() for p in build_model(name, HM.memoryless_width(name, "small", 16), dim=16).parameters())
    assert 600 < n_par < 820
