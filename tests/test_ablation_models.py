#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ELEX の対照（2026-10-16）：順番を外した ②（LF_Par_MLP）と RK4 のニューラル ODE（RK4_MLP_state）のテスト.

pytest でも，`python tests/test_ablation_models.py` でも実行できる。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from harmonic_ic_dataset import DATASETS_DIR, save_manifest  # noqa: E402
from harmonic_losses import lf_one_step_loss  # noqa: E402
from harmonic_models import MEMLESS_W2, build_model, memoryless_width  # noqa: E402


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_same_params():
    """順番を外した ② は ② と，RK4 は ① と，同じ幅で同じパラメータ数."""
    for dim in (2, 4, 8):
        for sz in ("tiny", "small", "medium"):
            wf, wr = memoryless_width("LF_Free_MLP", sz, dim), memoryless_width("Res_MLP_state", sz, dim)
            assert memoryless_width("LF_Par_MLP", sz, dim) == wf
            assert memoryless_width("RK4_MLP_state", sz, dim) == wr
            assert _n(build_model("LF_Par_MLP", wf, dim=dim)) == _n(build_model("LF_Free_MLP", wf, dim=dim))
            assert _n(build_model("RK4_MLP_state", wr, dim=dim)) == _n(build_model("Res_MLP_state", wr, dim=dim))
    assert MEMLESS_W2["LF_Par_MLP"] == MEMLESS_W2["LF_Free_MLP"]


def test_kick_model():
    """位置の段を固定した ②：G を持たず，x ← x + Δt·v_new。速度の段は v にも依存する。パラメータ数は目標の ±12%."""
    from harmonic_models import MEMLESS_TARGET_PARAMS

    for dim in (2, 4, 8):
        k = dim // 2
        m = build_model("LF_Kick_MLP", 8, dim=dim)
        assert not hasattr(m, "G")
        x = torch.randn(5, dim)
        out, _, _ = m(x, torch.zeros(5, 1), torch.zeros(5, 1), False)
        assert torch.allclose(out, x[:, :k] + m.delta_t * x[:, k:])
        x2 = x.clone()
        x2[:, k:] += 1.0
        a, _, _ = m(x, torch.zeros(5, 1), torch.zeros(5, 1), True)
        b, _, _ = m(x2, torch.zeros(5, 1), torch.zeros(5, 1), True)
        assert not torch.allclose(a - x[:, k:], b - x2[:, k:])  # 力が速度に依存できる
        for sz, t in MEMLESS_TARGET_PARAMS.items():
            n = _n(build_model("LF_Kick_MLP", memoryless_width("LF_Kick_MLP", sz, dim), dim=dim))
            assert abs(n - t) / t < 0.12, (dim, sz, n)


def _pair(dim):
    torch.manual_seed(0)
    free = build_model("LF_Free_MLP", 8, dim=dim)
    par = build_model("LF_Par_MLP", 8, dim=dim)
    par.load_state_dict(free.state_dict())
    return free, par


def test_par_differs_only_in_order():
    """同じ重みで，速度の段は一致し，位置の段は「更新前の速度」を使う点だけが違う."""
    for dim in (2, 4):
        k = dim // 2
        free, par = _pair(dim)
        x = torch.randn(5, dim)
        pos, vel = x[:, :k], x[:, k:]
        z = torch.zeros(5, k)
        v_f, _, _ = free(x, torch.zeros(5, 1), torch.zeros(5, 1), True)
        v_p, h_new, _ = par(x, z, z, True)
        assert torch.allclose(v_f, v_p)
        assert torch.allclose(h_new, vel)  # 更新前の速度を位置の段へ渡す
        x_p, h2, _ = par(torch.cat([pos, v_p], 1), z, h_new, False)
        assert torch.allclose(x_p, pos + par.delta_t * par.G(torch.cat([pos, vel], 1)))
        x_f, _, _ = free(torch.cat([pos, v_f], 1), torch.zeros(5, 1), torch.zeros(5, 1), False)
        assert torch.allclose(x_f, pos + free.delta_t * free.G(torch.cat([pos, v_f], 1)))
        assert not torch.allclose(x_f, x_p)
        assert torch.count_nonzero(h2) == 0  # ステップをまたぐ記憶はない


def test_par_loss_uses_old_velocity():
    """教師強制の損失でも，位置の段は v_t（更新前）を使う：G を手で計算した損失と一致."""
    torch.manual_seed(1)
    par = build_model("LF_Par_MLP", 8)
    x, v, a = torch.randn(3, 5), torch.randn(3, 5), torch.ones(3)
    got = lf_one_step_loss(par, x, v, a, 1.0, "mse")
    dt, tot = par.delta_t, 0.0
    for t in range(4):
        s = torch.stack([x[:, t], v[:, t]], 1)
        tot = tot + torch.mean((v[:, t] + dt * par.F(s)[:, 0] - v[:, t + 1]) ** 2)
        tot = tot + torch.mean((x[:, t] + dt * par.G(s)[:, 0] - x[:, t + 1]) ** 2)
    assert torch.allclose(got, tot / 8, rtol=1e-5), (got, tot / 8)


def test_rk4_exact_field():
    """RK4 に真のベクトル場 (v, −x) を入れると，調和振動子の 1 ステップが Δt⁵ の精度で合う."""
    m = build_model("RK4_MLP_state", 4)

    class Exact(torch.nn.Module):
        def forward(self, s):
            return torch.stack([s[:, 1], -s[:, 0]], 1)

    m.F = Exact()
    m.delta_t = 0.05
    s0 = torch.tensor([[1.0, 0.0], [0.3, -0.7]], dtype=torch.float64)
    out, _ = m(s0, torch.zeros(2, 1, dtype=torch.float64))
    c, s = np.cos(0.05), np.sin(0.05)
    ref = torch.stack([c * s0[:, 0] + s * s0[:, 1], -s * s0[:, 0] + c * s0[:, 1]], 1)
    assert (out - ref).abs().max() < 1e-8


def test_train_eval_smoke():
    """調和振動子（_ampmin_smoke）と荷電粒子・FPU（小さなデータ）で学習・評価が通る."""
    from test_new_systems import _tiny_dataset

    made = []
    try:
        src = DATASETS_DIR / "_ampmin_smoke" / "manifest.json"
        dsets = []
        if src.exists():
            m = dict(json.loads(src.read_text()), dataset_id="_test_abl_harm", dt_direct=0.05)
            save_manifest(m, DATASETS_DIR / m["dataset_id"])
            made.append(m["dataset_id"])
            dsets.append((m["dataset_id"], ["--teacher", "analytic"]))
        else:
            print("  (skip harmonic: datasets/_ampmin_smoke なし)")
        for system in ("charged", "fpu"):
            did = f"_test_abl_{system}"
            _tiny_dataset(system, did)
            made.append(did)
            dsets.append((did, []))
        for did, extra in dsets:
            dim = int(json.loads((DATASETS_DIR / did / "manifest.json").read_text()).get("dim", 2))
            for model in ("LF_Par_MLP", "RK4_MLP_state", "LF_Kick_MLP", "Res2_MLP_state", "SIE_MLP_state", "SIEMid_MLP_state"):
                hs = memoryless_width(model, "tiny", dim)
                r = subprocess.run([sys.executable, "-u", "train_ic_generalization.py", "--model", model, "--dataset-id", did,
                                    "--run-id", "_t", "--hidden-size", str(hs), "--epochs", "2", "--batch-size", "4",
                                    "--lr", "1e-2", "--loss-type", "mse", "--seed", "0", "--no-progress", *extra],
                                   cwd=ROOT, capture_output=True, text=True)
                assert r.returncode == 0, (did, model, r.stderr[-1500:])
                for ev in ([], ["--splits", "val", "--out-subdir", "eval_val"]):
                    r = subprocess.run([sys.executable, "-u", "ic_generalization_eval.py", "--model", model,
                                        "--hidden-size", str(hs), "--dataset-id", did, "--run-id", "_t", "--t-evals", "2",
                                        "--no-progress", *ev], cwd=ROOT, capture_output=True, text=True)
                    assert r.returncode == 0, (did, model, r.stderr[-1500:])
                d = ROOT / "picture" / "ic_gen" / did / "_t" / model / f"hs{hs}"
                row = json.loads((d / "ic_generalization_summary.json").read_text())["rows"][0]
                assert np.isfinite(row["test_interp_x_rmse_rel_mean"]), (did, model, row)
    finally:
        for did in made:
            shutil.rmtree(DATASETS_DIR / did, ignore_errors=True)
            shutil.rmtree(ROOT / "picture" / "ic_gen" / did, ignore_errors=True)




def test_two_net_controls():
    """2026-10-18 の対照：網を 2 つに分けた ①（Res2）と順番だけを入れた ①（SIE）は LF_Par と同じ網・同じパラメータ数.

    Res2 は位置の網に更新前の速度 v_n を，SIE は更新後の速度 v_{n+1} を渡す。速度の段は同じ重みなら一致する。
    """
    for dim in (2, 4, 8):
        k = dim // 2
        for sz in ("tiny", "small", "medium"):
            w = memoryless_width("LF_Par_MLP", sz, dim)
            for name in ("Res2_MLP_state", "SIE_MLP_state"):
                assert memoryless_width(name, sz, dim) == w
                assert _n(build_model(name, w, dim=dim)) == _n(build_model("LF_Par_MLP", w, dim=dim))
        torch.manual_seed(0)
        res2 = build_model("Res2_MLP_state", 8, dim=dim)
        sie = build_model("SIE_MLP_state", 8, dim=dim)
        sie.load_state_dict(res2.state_dict())
        xv = torch.randn(5, dim)
        h = torch.zeros(5, 1)
        a, _ = res2(xv, h)
        b, _ = sie(xv, h)
        assert torch.allclose(a[:, k:], b[:, k:])  # 速度の段は同じ
        dt = res2.delta_t
        assert torch.allclose(a[:, :k], xv[:, :k] + dt * res2.G(xv))
        assert torch.allclose(b[:, :k], xv[:, :k] + dt * res2.G(torch.cat([xv[:, :k], b[:, k:]], dim=1)))
        assert not torch.allclose(a[:, :k], b[:, :k])


def test_two_net_teacher_step():
    """学習の 1 ステップ（2026-10-18）：SIE の位置の段には参照の v_{n+1} を入れ，位置の損失は F に流れない（LF 系の学習と同じ）.

    Res2 は位置の段が v_n を使うので，学習の 1 ステップは rollout の 1 ステップと同じ。
    """
    for dim in (2, 4):
        k = dim // 2
        torch.manual_seed(1)
        sie = build_model("SIE_MLP_state", 8, dim=dim)
        res2 = build_model("Res2_MLP_state", 8, dim=dim)
        xv, nxt = torch.randn(5, dim), torch.randn(5, dim)
        h = torch.zeros(5, 1)
        dt = sie.delta_t
        out = sie.teacher_step(xv, nxt)
        assert torch.allclose(out[:, :k], xv[:, :k] + dt * sie.G(torch.cat([xv[:, :k], nxt[:, k:]], dim=1)))
        assert torch.allclose(out[:, k:], sie(xv, h)[0][:, k:])
        out[:, :k].pow(2).sum().backward()
        assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in sie.F.parameters())
        assert torch.allclose(res2.teacher_step(xv, nxt), res2(xv, h)[0])


def test_sie_mid_model():
    """中点版（2026-10-19）：SIE と同じ網・同じパラメータ数。位置の網には (v_n + v_{n+1})/2 を渡し，学習では参照の v_{n+1} との平均.

    速度の段は同じ重みなら SIE と一致する。位置の損失は F に流れない（LF 系・SIE の学習と同じ）。
    """
    for dim in (2, 4, 8):
        k = dim // 2
        for sz in ("tiny", "small", "medium"):
            w = memoryless_width("SIE_MLP_state", sz, dim)
            assert memoryless_width("SIEMid_MLP_state", sz, dim) == w
            assert _n(build_model("SIEMid_MLP_state", w, dim=dim)) == _n(build_model("SIE_MLP_state", w, dim=dim))
        torch.manual_seed(2)
        sie = build_model("SIE_MLP_state", 8, dim=dim)
        mid = build_model("SIEMid_MLP_state", 8, dim=dim)
        mid.load_state_dict(sie.state_dict())
        xv, nxt = torch.randn(5, dim), torch.randn(5, dim)
        h = torch.zeros(5, 1)
        dt = mid.delta_t
        a, _ = sie(xv, h)
        b, _ = mid(xv, h)
        assert torch.allclose(a[:, k:], b[:, k:])  # 速度の段は同じ
        v, v_new = xv[:, k:], b[:, k:]
        assert torch.allclose(b[:, :k], xv[:, :k] + dt * mid.G(torch.cat([xv[:, :k], 0.5 * (v + v_new)], dim=1)))
        out = mid.teacher_step(xv, nxt)
        assert torch.allclose(out[:, :k], xv[:, :k] + dt * mid.G(torch.cat([xv[:, :k], 0.5 * (v + nxt[:, k:])], dim=1)))
        assert torch.allclose(out[:, k:], b[:, k:])
        out[:, :k].pow(2).sum().backward()
        assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in mid.F.parameters())


def test_rk4kin_model():
    """2 階の Neural ODE＋RK4（2026-10-18 夜）：LF_Kick_MLP と同じパラメータ数，加速度を真の値にすると RK4 の 1 ステップに一致."""
    import math

    for dim in (2, 4, 8):
        for sz in ("tiny", "small", "medium"):
            w = memoryless_width("RK4Kin_MLP_state", sz, dim)
            assert _n(build_model("RK4Kin_MLP_state", w, dim=dim)) == _n(
                build_model("LF_Kick_MLP", memoryless_width("LF_Kick_MLP", sz, dim), dim=dim))

    class Harm(torch.nn.Module):
        def forward(self, z):
            return -z[:, :1]

    m = build_model("RK4Kin_MLP_state", 8, dim=2)
    m.F = Harm()
    out, _ = m(torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64))
    th = m.delta_t
    rk4 = 1 - th ** 2 / 2 + th ** 4 / 24  # x の RK4 の多項式（調和振動子）
    assert abs(float(out[0, 0]) - rk4) < 1e-12 and abs(float(out[0, 0]) - math.cos(th)) < 1e-8
    assert abs(float(out[0, 1]) + math.sin(th)) < 1e-8


def test_pos_loss_weight():
    """位置の損失の重み（2026-10-18 夜）：既定 1 は従来どおり，W のときは位置の二乗誤差だけ W 倍."""
    import harmonic_losses as L

    torch.manual_seed(0)
    m = build_model("Res_MLP_state", 8, dim=4)
    s = torch.randn(3, 5, 4)
    scale = torch.ones(3, 4)
    base = float(L.state_one_step_loss(m, s, scale, "mse"))
    L.POS_LOSS_WEIGHT = 10.0
    try:
        wtd = float(L.state_one_step_loss(m, s, scale, "mse"))
    finally:
        L.POS_LOSS_WEIGHT = 1.0
    with torch.no_grad():
        ex = ev = 0.0
        for t in range(4):
            e = m(s[:, t], torch.zeros(3, 1))[0] - s[:, t + 1]
            ex += float((e[:, :2] ** 2).sum()) / e.numel()
            ev += float((e[:, 2:] ** 2).sum()) / e.numel()
    assert abs(base - (ex + ev) / 4) < 1e-6 and abs(wtd - (10 * ex + ev) / 4) < 1e-5


def test_large_size():
    """large（2026-10-18）：多自由度でも 2 次元でも，パラメータ数は目標の ±12%."""
    from harmonic_models import MEMLESS_TARGET_PARAMS

    t = MEMLESS_TARGET_PARAMS["large"]
    for dim in (2, 4):
        for name in ("Res_MLP_state", "LF_Free_MLP", "LF_Kick_MLP"):
            n = _n(build_model(name, memoryless_width(name, "large", dim), dim=dim))
            assert abs(n - t) / t < 0.12, (name, dim, n)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
