#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""調和振動子以外の系（systems.py / system_dataset.py / system_eval.py，2026-09-30）のテスト.

pytest でも，`python tests/test_systems.py` でも実行できる。
"""

from __future__ import annotations

import json
import math
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
from harmonic_losses import direct_state_one_step_loss, state_one_step_loss  # noqa: E402
from harmonic_models import build_model  # noqa: E402
from systems import Lorenz, make_system  # noqa: E402


def test_reference_accuracy():
    h = make_system("harmonic")
    ics = np.array([[0.7, -0.4], [0.1, 0.9]])
    ref = h.integrate(ics, 0.025, 401)
    t = np.arange(401) * 0.025
    assert np.abs(ref[:, :, 0] - (ics[:, :1] * np.cos(t) + ics[:, 1:] * np.sin(t))).max() < 1e-10
    th = (np.arange(6) - 0.5) * 0.05
    vh = h.half_step_velocity(ics, 0.05, 6)
    assert np.abs(vh - (-ics[:, :1] * np.sin(th) + ics[:, 1:] * np.cos(th))).max() < 1e-12
    # 単振り子：エネルギー保存
    p = make_system("pendulum")
    r = p.integrate(np.array([[1.0, 0.2], [-0.4, 0.5]]), 0.025, 801)
    E = p.energy(r)
    assert np.abs(E - E[:, :1]).max() < 1e-10
    # ファンデルポール：リミットサイクルの振幅は約 2
    v = make_system("vanderpol")
    rv = v.integrate(np.array([[0.1, 0.0], [2.8, 0.5]]), 0.05, 1201)
    assert np.all(np.abs(np.abs(rv[:, -200:, 0]).max(axis=1) - 2.0) < 0.05)


def test_sampling_regions():
    rng = np.random.default_rng(0)
    p = make_system("pendulum")
    ic = p.sample_ics(300, 0.05, math.pi / 3, rng)
    tm = p.theta_max(ic)
    assert tm.min() > 0.05 and tm.max() <= math.pi / 3 + 1e-12
    ic = p.sample_ics(100, math.pi / 3, math.pi / 2, rng)
    tm = p.theta_max(ic)
    assert tm.min() > math.pi / 3 and tm.max() <= math.pi / 2 + 1e-12
    v = make_system("vanderpol")
    r = v.region_value(v.sample_ics(200, 2.0, 3.0, rng))
    assert r.min() > 2.0 and r.max() <= 3.0
    lz = make_system("lorenz")
    used: set = set()
    a = lz.sample_ics(50, 0, 0, rng, exclude=used)
    b = lz.sample_ics(50, 0, 0, rng, exclude=used)
    assert len(used) == 100 and not (set(map(tuple, a.round(12))) & set(map(tuple, b.round(12))))
    e = lz.sample_ics(30, 0.5, 1.0, rng)
    rv = lz.region_value(e)
    assert rv.min() > 0.5 and rv.max() <= 1.0


def test_generic_loss_matches_2d():
    torch.manual_seed(0)
    m = build_model("RNN_state", 6)
    x, v = torch.randn(4, 7), torch.randn(4, 7)
    s = torch.stack([x, v], dim=2)
    amp = torch.rand(4) + 0.5
    a = direct_state_one_step_loss(m, x, v, amp, 1.0, "mse")
    b = state_one_step_loss(m, s, torch.stack([amp, amp], 1), "mse")
    assert torch.allclose(a, b, atol=1e-6)
    c = direct_state_one_step_loss(m, x, v, amp, 1.0, "amp_norm_mse")
    d = state_one_step_loss(m, s, torch.stack([amp, amp], 1), "amp_norm_mse")
    assert torch.allclose(c, d, atol=1e-6)
    # 3 次元（ローレンツ）でも作れる
    for name in ("RNN_state", "LTCs_state"):
        mm = build_model(name, 5, dim=3)
        loss = state_one_step_loss(mm, torch.randn(2, 5, 3), torch.ones(2, 3), "mse")
        assert torch.isfinite(loss)


def test_perfect_prediction_metrics():
    import system_eval
    p = make_system("pendulum")
    ics = np.array([[0.5, 0.1], [1.0, 0.0]])
    ref = p.integrate(ics, 0.025, 81)
    per = system_eval._per_ic(p, ref, ref.copy(), p.scales(ics), 0.025, 2.0)
    assert np.allclose(per["x_rmse_rel"], 0) and np.allclose(per["vpt"], 2.0) and np.allclose(per["energy_rel"], 0)


def _tiny_dataset(system: str, did: str) -> None:
    args = Namespace(system=system, dataset_id=did, n_train=8, n_val=4, n_test_interp=4, n_test_extrap=4,
                     train_lo=None, train_hi=None, extrap_lo=None, extrap_hi=None, omega=None, omega0=None, mu=None,
                     duff_alpha=None, duff_beta=None, duff_delta=None,
                     sigma=None, rho=None, beta=None, t_end=2.0 if system == "lorenz" else 4.0, dt_direct=None,
                     dt_lf=None, scale_floor=0.05, seed=1)
    import system_dataset
    save_manifest(system_dataset.build(args), DATASETS_DIR / did)


def test_dataset_and_train_eval_smoke():
    """データセット → 学習 → 評価（val も）が通り，要約に既存と同じキーが出る."""
    cases = [("pendulum", "LF_LTCs", 16, ["2"]), ("pendulum", "RNN_state", 23, ["2"]),
             ("vanderpol", "LF_Free_MLP", 16, ["2"]), ("duffing", "LF_RNN", 16, ["2"]),
             ("lorenz", "LTCs_state", 23, ["0.5"])]
    made = []
    try:
        for system, model, hs, te in cases:
            did = f"_test_sys_{system}"
            if did not in made:
                _tiny_dataset(system, did)
                made.append(did)
                m = json.loads((DATASETS_DIR / did / "manifest.json").read_text())
                ics, sc = split_to_array(m, "train"), split_scales(m, "train")
                assert ics.shape == (8, m["dim"]) and sc.shape == (8, m["dim"]) and (sc > 0).all()
            base = [sys.executable, "-u"]
            r = subprocess.run(base + ["train_ic_generalization.py", "--model", model, "--dataset-id", did, "--run-id", "_t",
                                       "--epochs", "2", "--batch-size", "4", "--lr", "1e-2", "--loss-type", "mse", "--seed", "0",
                                       "--no-progress"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-1500:]
            for extra in ([], ["--splits", "val", "--out-subdir", "eval_val"]):
                r = subprocess.run(base + ["ic_generalization_eval.py", "--model", model, "--dataset-id", did, "--run-id", "_t",
                                           "--t-evals", *te, "--no-progress", *extra], cwd=ROOT, capture_output=True, text=True)
                assert r.returncode == 0, r.stderr[-1500:]
            d = ROOT / "picture" / "ic_gen" / did / "_t" / model / f"hs{hs}"
            row = json.loads((d / "ic_generalization_summary.json").read_text())["rows"][0]
            assert "test_interp_x_rmse_rel_mean" in row and "test_extrap_vpt_mean" in row
            vrow = json.loads((d / "eval_val" / "ic_generalization_summary.json").read_text())["rows"][0]
            assert "val_x_rmse_rel_mean" in vrow
            cfg = json.loads((d / "config.json").read_text())
            assert cfg["system"] == system and cfg["dim"] == (3 if system == "lorenz" else 2)
    finally:
        for did in made:
            shutil.rmtree(DATASETS_DIR / did, ignore_errors=True)
            shutil.rmtree(ROOT / "picture" / "ic_gen" / did, ignore_errors=True)


def test_duffing():
    """ダフィング方程式（外力なし）: エネルギー保存，振幅の範囲，硬化型（振幅が大きいほど周期が短い），減衰."""
    d = make_system("duffing")
    rng = np.random.default_rng(3)
    ic = d.sample_ics(300, 0.05, 1.0, rng)
    A = d.amplitude(ic)
    assert A.min() > 0.05 and A.max() <= 1.0 + 1e-12
    ex = d.amplitude(d.sample_ics(100, 1.0, 1.2, rng))
    assert ex.min() > 1.0 and ex.max() <= 1.2 + 1e-12
    r = d.integrate(ic[:4], 0.025, 801)
    E = d.energy(r)
    assert np.abs(E - E[:, :1]).max() < 1e-10
    assert np.abs(np.abs(r[:, :, 0]).max(axis=1) - A[:4]).max() < 1e-3  # 折り返し点の |x| が振幅

    def period(a0):
        x = d.integrate(np.array([[a0, 0.0]]), 0.002, 10001)[0, :, 0]
        z = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
        return (z[-1] - z[0]) / (len(z) - 1) * 0.002
    assert period(0.1) > period(1.0) > period(1.2)
    dd = make_system("duffing", delta=0.1)
    Ed = dd.energy(dd.integrate(np.array([[1.0, 0.0]]), 0.025, 401))
    assert Ed[0, -1] < 0.5 * Ed[0, 0]


def test_split_state_models():
    """Split_*_state（2026-10-02）: LF と同じパラメータ数，2 つのセルが自分の隠れ状態だけを使い，x のセルは同じ (x_n, v_n) を読む."""
    torch.manual_seed(0)
    for split, lf in (("Split_LTCs_state", "LF_LTCs"), ("Split_RNN_state", "LF_RNN")):
        for hs in (10, 16, 22):
            ms, ml = build_model(split, hs), build_model(lf, hs)
            assert sum(p.numel() for p in ms.parameters()) == sum(p.numel() for p in ml.parameters())
        m = build_model(split, 6)
        from harmonic_models import hidden_size_of
        assert hidden_size_of(m) == 12
        xv, h = torch.randn(5, 2), torch.randn(5, 12)
        out, h2 = m(xv, h)
        assert out.shape == (5, 2) and h2.shape == (5, 12)
        # x の出力は h_v に依存しない（同時更新・セルは独立）
        h_alt = h.clone()
        h_alt[:, 6:] += 1.0
        out2, _ = m(xv, h_alt)
        assert torch.allclose(out[:, 0], out2[:, 0]) and not torch.allclose(out[:, 1], out2[:, 1])
        x, v = torch.randn(3, 6), torch.randn(3, 6)
        loss = direct_state_one_step_loss(m, x, v, torch.ones(3), 1.0, "mse")
        loss.backward()
        assert torch.isfinite(loss)


def test_derived_dataset_and_model_dt():
    """派生データセット（学習数を減らす・刻みを粗くする）と，刻みが違うときモデル内部の刻みをデータに合わせること."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("derive_dataset", ROOT / "scripts" / "derive_dataset.py")
    dd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dd)
    did = "_test_sys_pendulum_base"
    made = [did]
    try:
        _tiny_dataset("pendulum", did)
        base = json.loads((DATASETS_DIR / did / "manifest.json").read_text())
        sub = dd.derive(base, did + "_n3", 3, None)
        assert sub["counts"]["train"] == 3 and len(sub["splits"]["train"]["ics"]) == 3
        assert sub["splits"]["train"]["ics"] == base["splits"]["train"]["ics"][:3]
        assert sub["splits"]["val"] == base["splits"]["val"] and sub["splits"]["test_interp"] == base["splits"]["test_interp"]
        crs = dd.derive(base, did + "_dt4", None, 4)
        assert abs(crs["dt_direct"] - 4 * base["dt_direct"]) < 1e-12 and abs(crs["dt_lf"] - 4 * base["dt_lf"]) < 1e-12
        assert crs["splits"] == base["splits"]
        for m in (sub, crs):
            save_manifest(m, DATASETS_DIR / m["dataset_id"])
            made.append(m["dataset_id"])
        base_cmd = [sys.executable, "-u"]
        for ds, model, hs in ((did + "_dt4", "LF_LTCs", 16), (did + "_dt4", "Split_LTCs_state", 16), (did + "_n3", "RNN_state", 23)):
            r = subprocess.run(base_cmd + ["train_ic_generalization.py", "--model", model, "--dataset-id", ds, "--run-id", "_t",
                                           "--epochs", "2", "--batch-size", "4", "--lr", "1e-2", "--loss-type", "mse",
                                           "--seed", "0", "--no-progress"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-1500:]
            r = subprocess.run(base_cmd + ["ic_generalization_eval.py", "--model", model, "--dataset-id", ds, "--run-id", "_t",
                                           "--t-evals", "2", "--no-progress"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-1500:]
            cfg = json.loads((ROOT / "picture" / "ic_gen" / ds / "_t" / model / f"hs{hs}" / "config.json").read_text())
            if ds.endswith("_dt4"):
                want = crs["dt_lf"] if model.startswith("LF") else crs["dt_direct"]
                assert abs(cfg["model_delta_t"] - want) < 1e-12, cfg
                import ic_generalization_eval as ev
                from harmonic_common import find_model_checkpoint
                d = ROOT / "picture" / "ic_gen" / ds / "_t" / model / f"hs{hs}"
                mm, _ = ev.load_model_for_eval(model, hs, d, find_model_checkpoint(d, model, hs))
                assert abs(mm.delta_t - want) < 1e-12
            else:
                assert "model_delta_t" not in cfg and cfg["n_train"] == 3
    finally:
        for d in made:
            shutil.rmtree(DATASETS_DIR / d, ignore_errors=True)
            shutil.rmtree(ROOT / "picture" / "ic_gen" / d, ignore_errors=True)


def test_split_state_harmonic_smoke():
    """調和振動子のデータセットでも Split_*_state の学習と評価が通る（direct_state 系として扱われる）."""
    ds = "_ampmin_smoke"
    if not (DATASETS_DIR / ds / "manifest.json").exists():
        print("  (skip: datasets/_ampmin_smoke なし)")
        return
    try:
        for model in ("Split_LTCs_state", "Split_RNN_state"):
            r = subprocess.run([sys.executable, "-u", "train_ic_generalization.py", "--model", model, "--dataset-id", ds,
                                "--run-id", "_t_split", "--epochs", "2", "--batch-size", "128", "--lr", "1e-2",
                                "--loss-type", "mse", "--seed", "0", "--no-progress"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-1500:]
            r = subprocess.run([sys.executable, "-u", "ic_generalization_eval.py", "--model", model, "--dataset-id", ds,
                                "--run-id", "_t_split", "--t-evals", "2", "--no-progress"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-1500:]
            row = json.loads((ROOT / "picture" / "ic_gen" / ds / "_t_split" / model / "hs16" / "ic_generalization_summary.json").read_text())["rows"][0]
            assert math.isfinite(row["test_interp_x_rmse_rel_mean"])
    finally:
        shutil.rmtree(ROOT / "picture" / "ic_gen" / ds / "_t_split", ignore_errors=True)


def test_direct_dt_matched_to_lf():
    """--dt-direct（2026-10-05）：状態入力の 1 段系を LF と同じ刻みで学習する。調和振動子でもモデル内部の刻みがデータに合う."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("derive_dataset", ROOT / "scripts" / "derive_dataset.py")
    dd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dd)
    made = []
    try:
        cases = []
        if (DATASETS_DIR / "_ampmin_smoke" / "manifest.json").exists():
            base = json.loads((DATASETS_DIR / "_ampmin_smoke" / "manifest.json").read_text())
            m = dd.derive(base, "_test_harm_dtd2", None, None, 2 * base["dt_direct"])
            assert m["dt_lf"] == base["dt_lf"] and m["splits"] == base["splits"]
            save_manifest(m, DATASETS_DIR / m["dataset_id"])
            made.append(m["dataset_id"])
            cases.append((m["dataset_id"], m["dt_direct"], "128"))
        _tiny_dataset("pendulum", "_test_sys_pend_dtd_base")
        made.append("_test_sys_pend_dtd_base")
        base = json.loads((DATASETS_DIR / "_test_sys_pend_dtd_base" / "manifest.json").read_text())
        m = dd.derive(base, "_test_sys_pend_dtd2", None, None, base["dt_lf"])
        save_manifest(m, DATASETS_DIR / m["dataset_id"])
        made.append(m["dataset_id"])
        cases.append((m["dataset_id"], m["dt_direct"], "4"))
        for ds, want, bs in cases:
            for model, hs in (("LTCs_state", 23), ("RNN_state", 23)):
                r = subprocess.run([sys.executable, "-u", "train_ic_generalization.py", "--model", model, "--dataset-id", ds,
                                    "--run-id", "_t", "--epochs", "2", "--batch-size", bs, "--lr", "1e-2", "--loss-type", "mse",
                                    "--seed", "0", "--no-progress"], cwd=ROOT, capture_output=True, text=True)
                assert r.returncode == 0, r.stderr[-1500:]
                r = subprocess.run([sys.executable, "-u", "ic_generalization_eval.py", "--model", model, "--dataset-id", ds,
                                    "--run-id", "_t", "--t-evals", "2", "--no-progress"], cwd=ROOT, capture_output=True, text=True)
                assert r.returncode == 0, r.stderr[-1500:]
                d = ROOT / "picture" / "ic_gen" / ds / "_t" / model / f"hs{hs}"
                cfg = json.loads((d / "config.json").read_text())
                assert abs(cfg["model_delta_t"] - want) < 1e-12 and abs(cfg["dt"] - want) < 1e-12, cfg
                import ic_generalization_eval as ev
                from harmonic_common import find_model_checkpoint
                mm, _ = ev.load_model_for_eval(model, hs, d, find_model_checkpoint(d, model, hs))
                assert abs(mm.delta_t - want) < 1e-12
                row = json.loads((d / "ic_generalization_summary.json").read_text())["rows"][0]
                assert row["steps_total"] == int(round(2 / want)) + 1, row["steps_total"]
    finally:
        for did in made:
            shutil.rmtree(DATASETS_DIR / did, ignore_errors=True)
            shutil.rmtree(ROOT / "picture" / "ic_gen" / did, ignore_errors=True)


def test_dt_sweep_lf_harmonic():
    """--dt-lf（2026-10-05）：調和振動子でも LF の刻みを変えて学習・評価でき，モデル内部の刻みがデータに合う."""
    import importlib.util
    if not (DATASETS_DIR / "_ampmin_smoke" / "manifest.json").exists():
        print("  (skip: datasets/_ampmin_smoke なし)")
        return
    spec = importlib.util.spec_from_file_location("derive_dataset", ROOT / "scripts" / "derive_dataset.py")
    dd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dd)
    base = json.loads((DATASETS_DIR / "_ampmin_smoke" / "manifest.json").read_text())
    did = "_test_harm_dt0p2"
    m = dd.derive(base, did, None, None, 0.2, 0.2)
    assert m["dt_lf"] == 0.2 and m["dt_direct"] == 0.2 and m["splits"] == base["splits"]
    try:
        save_manifest(m, DATASETS_DIR / did)
        for model, hs in (("LF_LTCs", 16), ("LTCs_state", 23)):
            r = subprocess.run([sys.executable, "-u", "train_ic_generalization.py", "--model", model, "--dataset-id", did,
                                "--run-id", "_t", "--epochs", "2", "--batch-size", "128", "--lr", "1e-2", "--loss-type", "mse",
                                "--seed", "0", "--no-progress"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-1500:]
            r = subprocess.run([sys.executable, "-u", "ic_generalization_eval.py", "--model", model, "--dataset-id", did,
                                "--run-id", "_t", "--t-evals", "2", "--no-progress"], cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-1500:]
            d = ROOT / "picture" / "ic_gen" / did / "_t" / model / f"hs{hs}"
            cfg = json.loads((d / "config.json").read_text())
            assert abs(cfg["model_delta_t"] - 0.2) < 1e-12, cfg
            row = json.loads((d / "ic_generalization_summary.json").read_text())["rows"][0]
            assert row["steps_total"] == 11 and math.isfinite(row["test_interp_x_rmse_rel_mean"]), row
    finally:
        shutil.rmtree(DATASETS_DIR / did, ignore_errors=True)
        shutil.rmtree(ROOT / "picture" / "ic_gen" / did, ignore_errors=True)


def test_lf_rejected_for_lorenz():
    try:
        build_model("LF_LTCs", 8, dim=3)
        raise AssertionError("LF 系に 3 次元は渡せない")
    except ValueError:
        pass
    assert not Lorenz().second_order


def test_nlcap_circuit():
    """電圧に依存する容量の LC 回路（2026-10-18）：エネルギー ln(1+V²)/2 + I²/2 が保存され，dV/dt は x′ = v ではない."""
    from systems import make_system

    s = make_system("nlcap")
    rng = np.random.default_rng(0)
    ics = s.sample_ics(32, 0.00125, 0.5, rng)
    E = s.energy(ics)
    assert E.min() > 0.00125 and E.max() <= 0.5
    tr = s.integrate(ics, 0.05, 201)
    assert np.max(np.abs(s.energy(tr[:, -1]) - E) / E) < 1e-9
    d = s.rhs(ics)
    assert np.allclose(d[:, 0], (1 + ics[:, 0] ** 2) * ics[:, 1]) and np.allclose(d[:, 1], -ics[:, 0])


def test_nllc_circuit():
    """容量もインダクタンスも非線形の LC 回路（2026-10-19）：H = ln(1+V²)/2 + ln(1+I²)/2 が保存され，電流の更新（力）も非線形."""
    from systems import make_system

    s = make_system("nllc")
    rng = np.random.default_rng(1)
    ics = s.sample_ics(32, 0.00125, 0.5, rng)
    E = s.energy(ics)
    assert E.min() > 0.00125 and E.max() <= 0.5
    tr = s.integrate(ics, 0.05, 201)
    assert np.max(np.abs(s.energy(tr[:, -1]) - E) / E) < 1e-9
    d = s.rhs(ics)
    assert np.allclose(d[:, 0], (1 + ics[:, 0] ** 2) * ics[:, 1]) and np.allclose(d[:, 1], -(1 + ics[:, 1] ** 2) * ics[:, 0])


def test_nlind_circuit():
    """線形の容量（値は未知の C0 = 2）と飽和するインダクタの LC 回路（2026-10-19）：H が保存され，dV/dt = I/C0."""
    from systems import make_system

    s = make_system("nlind")
    rng = np.random.default_rng(2)
    ics = s.sample_ics(32, 0.0025, 1.0, rng)
    E = s.energy(ics)
    assert E.min() > 0.0025 and E.max() <= 1.0
    tr = s.integrate(ics, 0.05, 201)
    assert np.max(np.abs(s.energy(tr[:, -1]) - E) / E) < 1e-9
    d = s.rhs(ics)
    assert np.allclose(d[:, 0], ics[:, 1] / 2.0) and np.allclose(d[:, 1], -(1 + ics[:, 1] ** 2) * ics[:, 0])


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
