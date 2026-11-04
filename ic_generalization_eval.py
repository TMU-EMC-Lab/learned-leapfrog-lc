#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習済み IC汎化モデルを未学習初期条件で rollout 評価する。

使い方:
  python ic_generalization_eval.py --model LF_RNN --dataset-id mix_v1 --hidden-size 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from harmonic_common import find_model_checkpoint, ic_gen_model_dir
from harmonic_ic_dataset import load_manifest, split_to_array
from harmonic_models import MODEL_CHOICES, build_model, hidden_size_of, ltc_options_from_config, resolve_model_spec
from harmonic_ref import (
    TEACHER_CHOICES,
    lf_half_kick_velocity,
    lf_initial_velocity,
    primary_ref_for_teacher,
    reference_direct_numerical,
    reference_lf_numerical,
    reference_kdk_analytic,
    reference_leapfrog_kdk,
    reference_lf_analytic,
    sync_half_step_velocity,
    DEFAULT_AMP_FLOOR,
    effective_amplitude,
    energy,
    hamiltonian,
    ic_metrics_filename,
    ic_trajectories_filename,
    phase_angle,
    reference_analytic_v,
    reference_analytic_x,
    reference_leapfrog,
    relative_error_stats,
    steps_from_t_end,
    t_eval_tag,
    trajectory_stats,
    wrap_phase,
)
from harmonic_progress import iter_progress, progress_write
from harmonic_timing import PhaseTimer, timing_filename


DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

# (x, v) を予測する系統（_eval_split_lf で v の誤差・エネルギーも評価する）
XV_FAMILIES = ("lf", "kdk3", "direct_state")


def _model_dir(dataset_id: str, run_id: str, model: str, hidden: int) -> Path:
    return ic_gen_model_dir(dataset_id, run_id, model, hidden)


def _rollout_lf(model, x0: float, v0: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """LF 系の自己回帰 rollout. v0 は速度入力の初期値（teacher=analytic なら v(-Δt/2)）."""
    model.eval()
    x_preds = torch.zeros(steps, device=DEVICE)
    v_preds = torch.zeros(steps, device=DEVICE)
    x_preds[0] = x0
    v_preds[0] = v0
    hidden = hidden_size_of(model)
    h_pp = torch.zeros(1, hidden, device=DEVICE)
    h_p = torch.zeros(1, hidden, device=DEVICE)
    with torch.no_grad():
        for i in range((steps - 1) * 2):
            if i % 2 == 0:
                t = i // 2
                x_in = torch.cat([x_preds[t : t + 1].reshape(1, 1), v_preds[t : t + 1].reshape(1, 1)], dim=1)
                out, h_new, h_p_out = model(x_in, h_pp, h_p, is_velocity_step=True)
                v_preds[t + 1] = out.squeeze()
                h_pp, h_p = h_p_out, h_new
            else:
                t = i // 2
                x_in = torch.cat([x_preds[t : t + 1].reshape(1, 1), v_preds[t + 1 : t + 2].reshape(1, 1)], dim=1)
                out, h_new, h_p_out = model(x_in, h_pp, h_p, is_velocity_step=False)
                x_preds[t + 1] = out.squeeze()
                h_pp, h_p = h_p_out, h_new
    return x_preds.cpu().numpy(), v_preds.cpu().numpy()


def _rollout_kdk3(model, x0: float, v0: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """KDK3 系の自己回帰 rollout: kick1 → drift → kick2 を 1 タイムステップとして反復."""
    model.eval()
    x_preds = torch.zeros(steps, device=DEVICE)
    v_preds = torch.zeros(steps, device=DEVICE)
    x_preds[0] = x0
    v_preds[0] = v0
    hidden = hidden_size_of(model)
    h_pp = torch.zeros(1, hidden, device=DEVICE)
    h_p = torch.zeros(1, hidden, device=DEVICE)
    with torch.no_grad():
        for t in range(steps - 1):
            x_in = torch.cat([x_preds[t : t + 1].reshape(1, 1), v_preds[t : t + 1].reshape(1, 1)], dim=1)
            vh, h_new, h_p_out = model(x_in, h_pp, h_p, step_id=0)
            h_pp, h_p = h_p_out, h_new
            x_in = torch.cat([x_preds[t : t + 1].reshape(1, 1), vh.reshape(1, 1)], dim=1)
            out_x, h_new, h_p_out = model(x_in, h_pp, h_p, step_id=1)
            x_preds[t + 1] = out_x.squeeze()
            h_pp, h_p = h_p_out, h_new
            x_in = torch.cat([x_preds[t + 1 : t + 2].reshape(1, 1), vh.reshape(1, 1)], dim=1)
            out_v, h_new, h_p_out = model(x_in, h_pp, h_p, step_id=2)
            v_preds[t + 1] = out_v.squeeze()
            h_pp, h_p = h_p_out, h_new
    return x_preds.cpu().numpy(), v_preds.cpu().numpy()


def _rollout_lf_batch(model, x0: np.ndarray, v0: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """_rollout_lf を全初期条件まとめて回す版（結果は同じ。GPU との往復を 1 回にする）."""
    model.eval()
    n = len(x0)
    xs = torch.zeros(n, steps, device=DEVICE)
    vs = torch.zeros(n, steps, device=DEVICE)
    xs[:, 0] = torch.as_tensor(np.asarray(x0, dtype=np.float32), device=DEVICE)
    vs[:, 0] = torch.as_tensor(np.asarray(v0, dtype=np.float32), device=DEVICE)
    hidden = hidden_size_of(model)
    h_pp = torch.zeros(n, hidden, device=DEVICE)
    h_p = torch.zeros(n, hidden, device=DEVICE)
    with torch.no_grad():
        for t in range(steps - 1):
            out, h_new, h_p_out = model(torch.stack([xs[:, t], vs[:, t]], dim=1), h_pp, h_p, is_velocity_step=True)
            vs[:, t + 1] = out[:, 0]
            h_pp, h_p = h_p_out, h_new
            out, h_new, h_p_out = model(torch.stack([xs[:, t], vs[:, t + 1]], dim=1), h_pp, h_p, is_velocity_step=False)
            xs[:, t + 1] = out[:, 0]
            h_pp, h_p = h_p_out, h_new
    return xs.cpu().numpy(), vs.cpu().numpy()


def _rollout_kdk3_batch(model, x0: np.ndarray, v0: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """_rollout_kdk3 を全初期条件まとめて回す版."""
    model.eval()
    n = len(x0)
    xs = torch.zeros(n, steps, device=DEVICE)
    vs = torch.zeros(n, steps, device=DEVICE)
    xs[:, 0] = torch.as_tensor(np.asarray(x0, dtype=np.float32), device=DEVICE)
    vs[:, 0] = torch.as_tensor(np.asarray(v0, dtype=np.float32), device=DEVICE)
    hidden = hidden_size_of(model)
    h_pp = torch.zeros(n, hidden, device=DEVICE)
    h_p = torch.zeros(n, hidden, device=DEVICE)
    with torch.no_grad():
        for t in range(steps - 1):
            vh, h_new, h_p_out = model(torch.stack([xs[:, t], vs[:, t]], dim=1), h_pp, h_p, step_id=0)
            h_pp, h_p = h_p_out, h_new
            out_x, h_new, h_p_out = model(torch.stack([xs[:, t], vh[:, 0]], dim=1), h_pp, h_p, step_id=1)
            xs[:, t + 1] = out_x[:, 0]
            h_pp, h_p = h_p_out, h_new
            out_v, h_new, h_p_out = model(torch.stack([xs[:, t + 1], vh[:, 0]], dim=1), h_pp, h_p, step_id=2)
            vs[:, t + 1] = out_v[:, 0]
            h_pp, h_p = h_p_out, h_new
    return xs.cpu().numpy(), vs.cpu().numpy()


def _rollout_direct_batch(model, x0: np.ndarray, steps: int) -> np.ndarray:
    """_rollout_direct を全初期条件まとめて回す版（毎ステップの .item() による同期を無くす）."""
    model.eval()
    n = len(x0)
    xs = torch.zeros(n, steps, device=DEVICE)
    xs[:, 0] = torch.as_tensor(np.asarray(x0, dtype=np.float32), device=DEVICE)
    h = torch.zeros(n, hidden_size_of(model), device=DEVICE)
    with torch.no_grad():
        for t in range(steps - 1):
            out, h = model(xs[:, t : t + 1], h)
            xs[:, t + 1] = out[:, 0]
    return xs.cpu().numpy()


def lf_family_eval_kwargs(
    family: str,
    *,
    ref_mode: str,
    teacher: str,
    omega: float,
    dt: float,
) -> dict[str, Any]:
    """LF / KDK3 系の _eval_split_lf に渡す rollout・参照解・初期速度の設定.

    ref_mode:  analytic → 解析解（LF の v は半整数時刻）/ numerical → Störmer–Verlet 数値解 /
               legacy → 旧数値参照解
    teacher:   学習時の教師。LF 系の速度入力の初期値を決める
               （analytic → v(-Δt/2)、numerical → ハーフキック、legacy → v0）
    """
    if family == "direct_state":
        # v は x と同じ整数時刻。初期速度は v0 のまま。legacy 参照解は無いので数値解で代用しない
        if ref_mode not in ("analytic", "numerical"):
            raise ValueError(f"direct_state 系の参照解は analytic / numerical のみ（指定: {ref_mode}）")
        return {
            "rollout_fn": _rollout_direct_state,
            "ref_fn": _analytic_state_ref if ref_mode == "analytic" else reference_direct_numerical,
            "v_init_fn": None,
            "v_half_step": False,
        }
    if ref_mode == "analytic":
        ref_fn = (lambda x0, v0, om, d, st: reference_kdk_analytic(x0, v0, om, d, st)[:2]) \
            if family == "kdk3" else reference_lf_analytic
    elif ref_mode == "numerical":
        ref_fn = (lambda x0, v0, om, d, st: reference_leapfrog_kdk(x0, v0, om, d, st)[:2]) \
            if family == "kdk3" else reference_lf_numerical
    else:
        ref_fn = (lambda x0, v0, om, d, st: reference_leapfrog_kdk(x0, v0, om, d, st)[:2]) \
            if family == "kdk3" else reference_leapfrog
    v_init_fn = None
    if family == "lf" and teacher == "analytic":
        v_init_fn = lambda x0, v0: lf_initial_velocity(x0, v0, omega, dt)  # noqa: E731
    elif family == "lf" and teacher == "numerical":
        v_init_fn = lambda x0, v0: lf_half_kick_velocity(x0, v0, omega, dt)  # noqa: E731
    return {
        "rollout_fn": _rollout_kdk3 if family == "kdk3" else _rollout_lf,
        "ref_fn": ref_fn,
        "v_init_fn": v_init_fn,
        "v_half_step": family == "lf" and ref_mode in ("analytic", "numerical"),
    }


def load_model_for_eval(model_name: str, hidden: int, model_dir: Path, model_path: Path):
    """config.json の teacher / LTC 設定に従ってモデルを復元する. 戻り値 (model, config)."""
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config.setdefault("teacher", "legacy")  # キーが無い 2026-09 以前の run は legacy
    model = build_model(model_name, hidden, **ltc_options_from_config(config), dim=config.get("dim")).to(DEVICE)
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE, weights_only=True))
    if "model_delta_t" in config:
        model.delta_t = float(config["model_delta_t"])
    return model, config


def _rollout_direct_state_batch(model, x0: np.ndarray, v0: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """1 段系の状態入力版: (x_n, v_n) → (x_{n+1}, v_{n+1}) を全初期条件まとめて回す."""
    model.eval()
    n = len(x0)
    xs = torch.zeros(n, steps, device=DEVICE)
    vs = torch.zeros(n, steps, device=DEVICE)
    xs[:, 0] = torch.as_tensor(np.asarray(x0, dtype=np.float32), device=DEVICE)
    vs[:, 0] = torch.as_tensor(np.asarray(v0, dtype=np.float32), device=DEVICE)
    h = torch.zeros(n, hidden_size_of(model), device=DEVICE)
    with torch.no_grad():
        for t in range(steps - 1):
            out, h = model(torch.stack([xs[:, t], vs[:, t]], dim=1), h)
            xs[:, t + 1] = out[:, 0]
            vs[:, t + 1] = out[:, 1]
    return xs.cpu().numpy(), vs.cpu().numpy()


def _rollout_direct_state(model, x0: float, v0: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    xs, vs = _rollout_direct_state_batch(model, np.array([x0]), np.array([v0]), steps)
    return xs[0], vs[0]


def _analytic_state_ref(x0: float, v0: float, omega: float, dt: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """整数時刻の解析解 (x, v)（direct_state 系の参照解）."""
    return (
        reference_analytic_x(x0, v0, omega, dt, steps),
        reference_analytic_v(x0, v0, omega, dt, steps),
    )


def _rollout_direct(model, x0: float, steps: int) -> np.ndarray:
    model.eval()
    preds = torch.zeros(steps, device=DEVICE)
    preds[0] = x0
    hidden = hidden_size_of(model)
    h = torch.zeros(1, hidden, device=DEVICE)
    current = float(x0)
    with torch.no_grad():
        for t in range(steps - 1):
            x_in = torch.tensor([[current]], device=DEVICE, dtype=torch.float32)
            out, h = model(x_in, h)
            current = float(out.item())
            preds[t + 1] = current
    return preds.cpu().numpy()


def _ham_phase_stats(
    x0: float,
    v0: float,
    ref_x: np.ndarray,
    ref_v: np.ndarray,
    pred_x: np.ndarray,
    pred_v: np.ndarray,
    omega: float,
) -> tuple[float, float, float, float]:
    """1 IC 分のハミルトニアン・位相誤差統計（全件集計用）.

    H の誤差は解析的に厳密保存される値 H(x0,v0) を基準に評価する
    （数値参照解自身の H は積分誤差で微小に振動するため）。
    """
    h_exact = float(hamiltonian(np.array([x0]), np.array([v0]), omega)[0])
    h_pred = hamiltonian(pred_x, pred_v, omega)
    dh = h_pred - h_exact
    h_scale = abs(h_exact) + 1e-12
    th_ref = phase_angle(ref_x, ref_v, omega)
    th_pred = phase_angle(pred_x, pred_v, omega)
    dth = wrap_phase(th_pred - th_ref)
    return (
        float(np.mean(np.abs(dh))),
        float(np.mean(np.abs(dh)) / h_scale),
        float(np.mean(np.abs(dth))),
        float(np.sqrt(np.mean(dth * dth))),
    )


def _eval_split_lf(
    model,
    ics: np.ndarray,
    omega: float,
    dt: float,
    t_end: float,
    *,
    amp_floor: float = DEFAULT_AMP_FLOOR,
    cache_trajectories: bool = False,
    progress_desc: str = "",
    disable_progress: bool = False,
    rollout_fn=None,
    ref_fn=None,
    v_init_fn=None,
    v_half_step: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]] | None]:
    """LF / KDK3 系の評価.

    v_init_fn: (x0, v0) → rollout に渡す速度入力の初期値（None なら v0 のまま）。
    v_half_step: v 列が半整数時刻 v[n] = v_{n-1/2} の場合 True。エネルギー・H・位相は
        sync_half_step_velocity で整数時刻に揃えた v と x[:-1] で計算する。
    """
    rollout_fn = rollout_fn or _rollout_lf
    ref_fn = ref_fn or reference_leapfrog
    steps = steps_from_t_end(t_end, dt)
    rows: list[dict[str, Any]] = []
    cache: list[dict[str, np.ndarray]] | None = [] if cache_trajectories else None
    t = np.arange(steps, dtype=np.float64) * dt
    ic_bar = iter_progress(
        range(len(ics)),
        desc=progress_desc or "eval ICs",
        total=len(ics),
        disable=disable_progress,
        unit="ic",
        leave=False,
    )
    # 予測は全初期条件まとめて回す（1 本ずつの関数を渡された場合はそのまま使う）
    batch_fn = {
        _rollout_lf: _rollout_lf_batch,
        _rollout_kdk3: _rollout_kdk3_batch,
        _rollout_direct_state: _rollout_direct_state_batch,
    }.get(rollout_fn)
    if batch_fn is not None and len(ics):
        v_in_all = [v_init_fn(float(x0), float(v0)) if v_init_fn is not None else float(v0) for x0, v0 in ics]
        pred_x_all, pred_v_all = batch_fn(model, ics[:, 0], np.asarray(v_in_all), steps)
    for i in ic_bar:
        x0, v0 = ics[i]
        amp_eff = effective_amplitude(float(x0), float(v0), omega, amp_floor)
        ref_x, ref_v = ref_fn(float(x0), float(v0), omega, dt, steps)
        if batch_fn is not None:
            pred_x, pred_v = pred_x_all[i].astype(np.float64), pred_v_all[i].astype(np.float64)
        else:
            v_in0 = v_init_fn(float(x0), float(v0)) if v_init_fn is not None else float(v0)
            pred_x, pred_v = rollout_fn(model, float(x0), v_in0, steps)
        x_diff = ref_x - pred_x
        v_diff = ref_v - pred_v
        x_err = trajectory_stats(x_diff, np.abs(x_diff))
        v_err = trajectory_stats(v_diff, np.abs(v_diff))
        if v_half_step:
            ph_ref_x, ph_ref_v = ref_x[:-1], sync_half_step_velocity(ref_v)
            ph_pred_x, ph_pred_v = pred_x[:-1], sync_half_step_velocity(pred_v)
        else:
            ph_ref_x, ph_ref_v, ph_pred_x, ph_pred_v = ref_x, ref_v, pred_x, pred_v
        e_pred = energy(ph_pred_x, ph_pred_v, omega)
        e0 = float(e_pred[0])
        e_drift = np.abs(e_pred - e0)
        h_mae, h_rel_mae, phase_mae, phase_rmse = _ham_phase_stats(
            float(x0), float(v0), ph_ref_x, ph_ref_v, ph_pred_x, ph_pred_v, omega
        )
        rows.append({
            "x0": float(x0),
            "v0": float(v0),
            "amplitude": float(amp_eff),
            "x_err": x_err,
            "x_err_rel": relative_error_stats(x_err, amp_eff),
            "v_err": v_err,
            "v_err_rel": relative_error_stats(v_err, omega * amp_eff),
            "energy_drift_max": float(np.max(e_drift)),
            "energy_drift_mean": float(np.mean(e_drift)),
            "energy_rel_drift_max": float(np.max(e_drift) / (abs(e0) + 1e-12)),
            "h_mae": h_mae,
            "h_rel_mae": h_rel_mae,
            "phase_mae_rad": phase_mae,
            "phase_rmse_rad": phase_rmse,
        })
        if cache is not None:
            cache.append({
                "t": t,
                "x0": np.array([float(x0)]),
                "v0": np.array([float(v0)]),
                "x_ref": ref_x,
                "v_ref": ref_v,
                "x_hat": pred_x,
                "v_hat": pred_v,
                # True: v_ref / v_hat は半整数時刻 v[n] = v_{n-1/2}（ic_plot で整数時刻に揃える）
                "v_half_step": np.array([v_half_step]),
            })
    return rows, cache


def _eval_split_direct(
    model,
    ics: np.ndarray,
    omega: float,
    dt: float,
    t_end: float,
    *,
    amp_floor: float = DEFAULT_AMP_FLOOR,
    cache_trajectories: bool = False,
    progress_desc: str = "",
    disable_progress: bool = False,
    rollout_fn=None,
    ref_fn=None,
    ref_mode: str = "analytic",
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]] | None]:
    """1 段系の評価. ref_mode=numerical のときだけ Störmer–Verlet 数値解、それ以外は解析解."""
    steps = steps_from_t_end(t_end, dt)
    rows: list[dict[str, Any]] = []
    cache: list[dict[str, np.ndarray]] | None = [] if cache_trajectories else None
    t = np.arange(steps, dtype=np.float64) * dt
    ic_bar = iter_progress(
        range(len(ics)),
        desc=progress_desc or "eval ICs",
        total=len(ics),
        disable=disable_progress,
        unit="ic",
        leave=False,
    )
    pred_x_all = _rollout_direct_batch(model, ics[:, 0], steps) if len(ics) else None
    for i in ic_bar:
        x0, v0 = ics[i]
        amp_eff = effective_amplitude(float(x0), float(v0), omega, amp_floor)
        if ref_mode == "numerical":
            ref_x, ref_v = reference_direct_numerical(float(x0), float(v0), omega, dt, steps)
        else:
            ref_x = reference_analytic_x(float(x0), float(v0), omega, dt, steps)
            ref_v = reference_analytic_v(float(x0), float(v0), omega, dt, steps)
        pred_x = pred_x_all[i].astype(np.float64)
        x_diff = ref_x - pred_x
        x_err = trajectory_stats(x_diff, np.abs(x_diff))
        pred_v = np.gradient(pred_x, dt)
        h_mae, h_rel_mae, phase_mae, phase_rmse = _ham_phase_stats(
            float(x0), float(v0), ref_x, ref_v, pred_x, pred_v, omega
        )
        rows.append({
            "x0": float(x0),
            "v0": float(v0),
            "amplitude": float(amp_eff),
            "x_err": x_err,
            "x_err_rel": relative_error_stats(x_err, amp_eff),
            "v_err": None,
            "v_err_rel": None,
            "energy_drift_max": None,
            "energy_drift_mean": None,
            "energy_rel_drift_max": None,
            "h_mae": h_mae,
            "h_rel_mae": h_rel_mae,
            "phase_mae_rad": phase_mae,
            "phase_rmse_rad": phase_rmse,
        })
        if cache is not None:
            cache.append({
                "t": t,
                "x0": np.array([float(x0)]),
                "v0": np.array([float(v0)]),
                "x_ref": ref_x,
                "x_hat": pred_x,
            })
    return rows, cache


def _pack_npz(cache: list[dict[str, np.ndarray]], indices: list[int]) -> dict[str, np.ndarray]:
    out: dict[str, list] = {}
    for i in indices:
        for k, v in cache[i].items():
            out.setdefault(k, []).append(v)
    packed = {k: np.array(v) for k, v in out.items()}
    if "t" in packed:
        packed["t"] = packed["t"][0]
    if "x0" in packed:
        packed["x0"] = packed["x0"].reshape(-1)
    if "v0" in packed:
        packed["v0"] = packed["v0"].reshape(-1)
    return packed


def _aggregate(rows: list[dict[str, Any]], key: str, metric: str = "mae") -> dict[str, float]:
    vals = [
        r[key][metric] for r in rows
        if r.get(key) and metric in r[key] and np.isfinite(r[key][metric])
    ]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan")}
    a = np.array(vals, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "max": float(np.max(a))}


def _aggregate_flat(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    """フラットなスカラー値フィールド（h_rel_mae 等）を全件集計する.

    h_exact≈0 の IC で h_rel_mae が非有限になり得るため，非有限値は除外する
    （このとき集計対象の n は summary["n"] より少なくなり得る）。
    """
    vals = [r[key] for r in rows if key in r and r[key] is not None and np.isfinite(r[key])]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan"), "n": 0}
    a = np.array(vals, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "max": float(np.max(a)), "n": len(vals)}


def _resolve_t_eval_list(
    t_eval: float | None,
    t_evals: list[float] | None,
    manifest_t_end: float,
) -> list[float]:
    if t_evals is not None:
        return [float(t) for t in t_evals]
    if t_eval is not None:
        return [float(t_eval)]
    return [float(manifest_t_end)]


def _run_one_t_eval(
    model,
    *,
    family: str,
    manifest: dict[str, Any],
    model_dir: Path,
    model_path: Path,
    args_model: str,
    dataset_id: str,
    run_id: str,
    hidden: int,
    omega: float,
    dt: float,
    t_eval: float,
    splits: list[str],
    save_npz: bool,
    n_save: int,
    amp_floor: float,
    disable_progress: bool = False,
    ref_mode: str = "legacy",
    teacher: str = "legacy",
) -> dict[str, Any]:
    """model_dir は出力先（ref_mode が学習時の teacher と異なる場合はサブディレクトリ）."""
    tag = t_eval_tag(t_eval)
    results: dict[str, Any] = {
        "model": args_model,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "hidden_size": hidden,
        "omega": omega,
        "t_eval": t_eval,
        "t_eval_tag": tag,
        "dt": dt,
        "steps_total": steps_from_t_end(t_eval, dt),
        "model_path": str(model_path),
        "ref_mode": ref_mode,
        "teacher": teacher,
        "splits": {},
    }

    infer_timer = PhaseTimer(f"infer_{tag}")
    infer_timer.start()
    progress_write(f"\n=== t_eval = {t_eval}s ({tag}) ===", disable=disable_progress)
    split_bar = iter_progress(
        splits,
        desc=f"eval {args_model} t={t_eval}s",
        disable=disable_progress,
        unit="split",
    )
    for split in split_bar:
        ics = split_to_array(manifest, split)
        if family in XV_FAMILIES:
            per_ic, cache = _eval_split_lf(
                model,
                ics,
                omega,
                dt,
                t_eval,
                amp_floor=amp_floor,
                cache_trajectories=save_npz,
                progress_desc=f"{split} ICs",
                disable_progress=disable_progress,
                **lf_family_eval_kwargs(family, ref_mode=ref_mode, teacher=teacher, omega=omega, dt=dt),
            )
        else:
            per_ic, cache = _eval_split_direct(
                model,
                ics,
                omega,
                dt,
                t_eval,
                amp_floor=amp_floor,
                cache_trajectories=save_npz,
                progress_desc=f"{split} ICs",
                disable_progress=disable_progress,
                ref_mode=ref_mode,
            )

        if save_npz and cache is not None and len(cache) > 0:
            n_save_eff = min(n_save, len(cache))
            rank_specs = (
                ("worst", "abs", [r["x_err"]["mae"] for r in per_ic]),
                ("best", "abs", [r["x_err"]["mae"] for r in per_ic]),
                ("worst", "rel", [r["x_err_rel"]["mae"] for r in per_ic]),
                ("best", "rel", [r["x_err_rel"]["mae"] for r in per_ic]),
            )
            for selection, metric, scores in rank_specs:
                if selection == "worst":
                    indices = np.argsort(scores)[-n_save_eff:][::-1].tolist()
                else:
                    indices = np.argsort(scores)[:n_save_eff].tolist()
                npz_data = _pack_npz(cache, indices)
                npz_path = model_dir / ic_trajectories_filename(
                    split, t_eval, selection=selection, metric=metric
                )
                np.savez(
                    npz_path,
                    split=np.array([split]),
                    t_eval=np.array([t_eval]),
                    metric=np.array([metric]),
                    selection=np.array([selection]),
                    **npz_data,
                )
                progress_write(f"  trajectories ({selection} {metric}) saved → {npz_path}", disable=disable_progress)

        summary = {
            "n": len(per_ic),
            "x_mae": _aggregate(per_ic, "x_err", "mae"),
            "x_rmse": _aggregate(per_ic, "x_err", "rmse"),
            "x_mae_rel": _aggregate(per_ic, "x_err_rel", "mae"),
            "x_rmse_rel": _aggregate(per_ic, "x_err_rel", "rmse"),
            "h_mae": _aggregate_flat(per_ic, "h_mae"),
            "h_rel_mae": _aggregate_flat(per_ic, "h_rel_mae"),
            "phase_mae_rad": _aggregate_flat(per_ic, "phase_mae_rad"),
            "phase_rmse_rad": _aggregate_flat(per_ic, "phase_rmse_rad"),
        }
        if family in XV_FAMILIES:
            summary["v_mae"] = _aggregate(per_ic, "v_err", "mae")
            summary["v_rmse"] = _aggregate(per_ic, "v_err", "rmse")
            summary["v_mae_rel"] = _aggregate(per_ic, "v_err_rel", "mae")
            summary["v_rmse_rel"] = _aggregate(per_ic, "v_err_rel", "rmse")
            ed = [r["energy_rel_drift_max"] for r in per_ic if r["energy_rel_drift_max"] is not None]
            summary["energy_rel_drift_max"] = {
                "mean": float(np.mean(ed)),
                "std": float(np.std(ed)),
                "max": float(np.max(ed)),
            } if ed else {"mean": float("nan"), "std": float("nan"), "max": float("nan")}

        results["splits"][split] = {"summary": summary, "per_ic": per_ic}
        progress_write(f"[{split}] n={summary['n']}", disable=disable_progress)
        progress_write(
            f"  x_MAE  mean/std/max: {summary['x_mae']['mean']:.6f} / "
            f"{summary['x_mae']['std']:.6f} / {summary['x_mae']['max']:.6f}",
            disable=disable_progress,
        )
        progress_write(
            f"  x_RMSE mean/std/max: {summary['x_rmse']['mean']:.6f} / "
            f"{summary['x_rmse']['std']:.6f} / {summary['x_rmse']['max']:.6f}",
            disable=disable_progress,
        )
        progress_write(
            f"  x_MAE_rel  mean/std/max: {summary['x_mae_rel']['mean']:.6f} / "
            f"{summary['x_mae_rel']['std']:.6f} / {summary['x_mae_rel']['max']:.6f}",
            disable=disable_progress,
        )
        progress_write(
            f"  x_RMSE_rel mean/std/max: {summary['x_rmse_rel']['mean']:.6f} / "
            f"{summary['x_rmse_rel']['std']:.6f} / {summary['x_rmse_rel']['max']:.6f}",
            disable=disable_progress,
        )
        progress_write(
            f"  H_rel_MAE  mean/std/max: {summary['h_rel_mae']['mean']:.6f} / "
            f"{summary['h_rel_mae']['std']:.6f} / {summary['h_rel_mae']['max']:.6f}  "
            f"(n={summary['h_rel_mae']['n']}/{summary['n']}, 非有限は除外)",
            disable=disable_progress,
        )
        progress_write(
            f"  phase_RMSE[rad] mean/std/max: {summary['phase_rmse_rad']['mean']:.6f} / "
            f"{summary['phase_rmse_rad']['std']:.6f} / {summary['phase_rmse_rad']['max']:.6f}",
            disable=disable_progress,
        )
        if family in XV_FAMILIES:
            progress_write(
                f"  v_MAE  mean/std/max: {summary['v_mae']['mean']:.6f} / "
                f"{summary['v_mae']['std']:.6f} / {summary['v_mae']['max']:.6f}",
                disable=disable_progress,
            )
            progress_write(
                f"  v_RMSE mean/std/max: {summary['v_rmse']['mean']:.6f} / "
                f"{summary['v_rmse']['std']:.6f} / {summary['v_rmse']['max']:.6f}",
                disable=disable_progress,
            )
            progress_write(
                f"  v_MAE_rel  mean/std/max: {summary['v_mae_rel']['mean']:.6f} / "
                f"{summary['v_mae_rel']['std']:.6f} / {summary['v_mae_rel']['max']:.6f}",
                disable=disable_progress,
            )
            progress_write(
                f"  energy_rel_drift_max mean/max: {summary['energy_rel_drift_max']['mean']:.6f} / "
                f"{summary['energy_rel_drift_max']['max']:.6f}",
                disable=disable_progress,
            )

    results["infer_s"] = infer_timer.stop()
    results["timing"] = infer_timer.to_dict()
    infer_timer.print_line()
    out_path = model_dir / ic_metrics_filename(t_eval)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics saved → {out_path}  ({results['infer_s']:.2f}s)")
    return results


def _compact_summary(results: dict[str, Any]) -> dict[str, Any]:
    """比較用の数値サマリ."""
    row: dict[str, Any] = {
        "t_eval": results["t_eval"],
        "t_eval_tag": results["t_eval_tag"],
        "steps_total": results["steps_total"],
        "infer_s": results["infer_s"],
        "metrics_path": ic_metrics_filename(results["t_eval"]),
    }
    timing = results.get("timing")
    if timing:
        row["infer_started_at"] = timing.get("started_at")
        row["infer_ended_at"] = timing.get("ended_at")
    for split, data in results.get("splits", {}).items():
        s = data["summary"]
        row[f"{split}_x_mae_mean"] = s["x_mae"]["mean"]
        row[f"{split}_x_mae_max"] = s["x_mae"]["max"]
        row[f"{split}_x_rmse_mean"] = s["x_rmse"]["mean"]
        row[f"{split}_x_rmse_max"] = s["x_rmse"]["max"]
        row[f"{split}_x_mae_rel_mean"] = s["x_mae_rel"]["mean"]
        row[f"{split}_x_mae_rel_max"] = s["x_mae_rel"]["max"]
        row[f"{split}_x_rmse_rel_mean"] = s["x_rmse_rel"]["mean"]
        row[f"{split}_x_rmse_rel_max"] = s["x_rmse_rel"]["max"]
        row[f"{split}_h_rel_mae_mean"] = s["h_rel_mae"]["mean"]
        row[f"{split}_h_rel_mae_max"] = s["h_rel_mae"]["max"]
        row[f"{split}_phase_mae_rad_mean"] = s["phase_mae_rad"]["mean"]
        row[f"{split}_phase_rmse_rad_mean"] = s["phase_rmse_rad"]["mean"]
        if "v_mae" in s:
            row[f"{split}_v_mae_mean"] = s["v_mae"]["mean"]
            row[f"{split}_v_rmse_mean"] = s["v_rmse"]["mean"]
            row[f"{split}_v_mae_rel_mean"] = s["v_mae_rel"]["mean"]
            row[f"{split}_v_rmse_rel_mean"] = s["v_rmse_rel"]["mean"]
            row[f"{split}_energy_rel_drift_max"] = s["energy_rel_drift_max"]["max"]
    return row


def _print_summary_table(rows: list[dict[str, Any]], family: str) -> None:
    print("\n=== IC generalization summary (by t_eval) ===")
    header = ["t_eval", "steps", "interp_x_mae", "extrap_x_mae", "interp_x_rmse", "extrap_x_rmse"]
    if family in XV_FAMILIES:
        header += ["interp_v_mae", "extrap_v_mae", "interp_v_rmse", "extrap_v_rmse", "extrap_E_drift"]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header)) + 10))
    for r in rows:
        cols = [
            f"{r['t_eval']:>5.0f}s",
            f"{r['steps_total']:>5d}",
            f"{r.get('test_interp_x_mae_mean', float('nan')):>12.6f}",
            f"{r.get('test_extrap_x_mae_mean', float('nan')):>12.6f}",
            f"{r.get('test_interp_x_rmse_mean', float('nan')):>13.6f}",
            f"{r.get('test_extrap_x_rmse_mean', float('nan')):>13.6f}",
        ]
        if family in XV_FAMILIES:
            cols += [
                f"{r.get('test_interp_v_mae_mean', float('nan')):>10.6f}",
                f"{r.get('test_extrap_v_mae_mean', float('nan')):>10.6f}",
                f"{r.get('test_interp_v_rmse_mean', float('nan')):>11.6f}",
                f"{r.get('test_extrap_v_rmse_mean', float('nan')):>11.6f}",
                f"{r.get('test_extrap_energy_rel_drift_max', float('nan')):>10.6f}",
            ]
        print(" | ".join(cols))


def main() -> int:
    p = argparse.ArgumentParser(description="IC汎化 rollout 評価")
    p.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    p.add_argument("--dataset-id", type=str, default="mix_amp1_v1")
    p.add_argument("--run-id", type=str, required=True, help="実験 run ID")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--splits", type=str, nargs="+", default=["test_interp", "test_extrap"])
    p.add_argument("--t-eval", type=float, default=None, help="単一評価時間 [s]")
    p.add_argument(
        "--t-evals",
        type=float,
        nargs="+",
        default=None,
        help="複数評価時間 [s]（例: 10 20 50）。指定時は --t-eval より優先",
    )
    p.add_argument("--save-npz", action="store_true", help="代表軌道を npz に保存（ic_plot 用）")
    p.add_argument("--n-save", type=int, default=3, help="各 split・指標ごとに保存する軌道数")
    p.add_argument(
        "--amp-floor",
        type=float,
        default=DEFAULT_AMP_FLOOR,
        help="相対誤差の規格化に用いる振幅下限",
    )
    p.add_argument(
        "--ref",
        type=str,
        default="auto",
        choices=["auto", *TEACHER_CHOICES],
        help="評価の参照解。auto: 主指標（analytic / numerical 教師は analytic、legacy は legacy）/ analytic: 解析解 / "
             "numerical: Störmer–Verlet 数値解 / legacy: 旧数値参照解。"
             "auto 以外の参照解で評価した結果は eval_ref_<ref>/ に保存（既存結果を上書きしない）",
    )
    p.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help="結果をモデルのフォルダの下のこのサブフォルダに出す（例: eval_val。検証データでの評価など、"
             "主評価の ic_generalization_summary.json を上書きしないため）",
    )
    p.add_argument("--no-progress", action="store_true", help="tqdm プログレスバーを無効化")
    p.add_argument("--data-dataset-id", type=str, default=None,
                   help="評価に使うデータのデータセット（既定は --dataset-id）。段ごとの網のモデル（length_agnostic）を，"
                        "学習と違う段数の線路で評価するとき（2026-10-21）。--out-subdir と一緒に使う")
    args = p.parse_args()

    args.model, spec = resolve_model_spec(args.model)
    hidden = int(args.hidden_size) if args.hidden_size is not None else int(spec["default_hs"])
    family = spec["family"]

    manifest = load_manifest(args.data_dataset_id or args.dataset_id)
    if args.data_dataset_id and not args.out_subdir:
        raise ValueError("--data-dataset-id を使うときは --out-subdir で出力先を分ける")
    if "system" in manifest:
        # 調和振動子以外の系（system_dataset.py のデータセット）は system_eval で評価する
        import system_eval

        return system_eval.main_generic(args, args.model, hidden, family, manifest)
    omega = float(manifest["omega"])
    manifest_t_end = float(manifest["t_end"])
    dt = float(manifest["dt_lf"] if family in ("lf", "kdk3") else manifest["dt_direct"])
    t_eval_list = _resolve_t_eval_list(args.t_eval, args.t_evals, manifest_t_end)

    model_dir = _model_dir(args.dataset_id, args.run_id, args.model, hidden)
    model_path = find_model_checkpoint(model_dir, args.model, hidden)
    if not model_path.exists():
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")

    model, config = load_model_for_eval(args.model, hidden, model_dir, model_path)
    teacher = config["teacher"]
    primary_ref = primary_ref_for_teacher(teacher)
    ref_mode = primary_ref if args.ref == "auto" else args.ref
    out_dir = model_dir if ref_mode == primary_ref else model_dir / f"eval_ref_{ref_mode}"
    if args.out_subdir:
        out_dir = out_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"teacher={teacher} ref={ref_mode} → {out_dir}")

    total_timer = PhaseTimer("infer_total")
    total_timer.start()

    all_results: list[dict[str, Any]] = []
    t_eval_bar = iter_progress(
        t_eval_list,
        desc=f"eval {args.model} hs={hidden}",
        disable=args.no_progress,
        unit="t_eval",
    )
    for t_eval in t_eval_bar:
        res = _run_one_t_eval(
            model,
            family=family,
            manifest=manifest,
            model_dir=out_dir,
            model_path=model_path,
            args_model=args.model,
            dataset_id=args.dataset_id,
            run_id=args.run_id,
            hidden=hidden,
            omega=omega,
            dt=dt,
            t_eval=t_eval,
            splits=args.splits,
            save_npz=args.save_npz,
            n_save=args.n_save,
            amp_floor=args.amp_floor,
            disable_progress=args.no_progress,
            ref_mode=ref_mode,
            teacher=teacher,
        )
        all_results.append(res)

    total_infer_s = total_timer.stop()
    total_timer.print_line()

    compact_rows = [_compact_summary(r) for r in all_results]
    _print_summary_table(compact_rows, family)

    per_t_eval_timing = [r["timing"] for r in all_results if "timing" in r]
    timing_doc = {
        "infer_total": total_timer.to_dict(),
        "per_t_eval": per_t_eval_timing,
        "infer_total_s": total_infer_s,
    }
    timing_path = out_dir / timing_filename("infer")
    timing_path.write_text(json.dumps(timing_doc, indent=2) + "\n", encoding="utf-8")
    print(f"Timing saved → {timing_path}")

    summary_doc = {
        "model": args.model,
        "dataset_id": args.dataset_id,
        "run_id": args.run_id,
        "hidden_size": hidden,
        "omega": omega,
        "dt": dt,
        "amp_floor": args.amp_floor,
        "ref_mode": ref_mode,
        "teacher": teacher,
        "t_evals": t_eval_list,
        "infer_total_s": total_infer_s,
        "timing": timing_doc,
        "rows": compact_rows,
        "details": {r["t_eval_tag"]: ic_metrics_filename(r["t_eval"]) for r in all_results},
    }
    summary_path = out_dir / "ic_generalization_summary.json"
    summary_path.write_text(json.dumps(summary_doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary saved → {summary_path}")
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "model": args.model,
                "dataset_id": args.dataset_id,
                "run_id": args.run_id,
                "hidden_size": hidden,
                "t_evals": t_eval_list,
                "infer_total_s": total_infer_s,
                "timing": timing_doc,
                "summary_path": str(summary_path),
                "rows": compact_rows,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
