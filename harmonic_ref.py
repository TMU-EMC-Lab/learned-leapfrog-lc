#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""調和振動子の参照解・物理量計算（IC汎化パイプライン共通）."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def reference_leapfrog(
    x0: float,
    v0: float,
    omega: float,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Leapfrog 数値積分（旧 LF 系の教師データ。teacher=legacy 用に残す）.

    注意: 漸化式は半ステップ leapfrog と同型（v[i] が v_{i-1/2} の役）だが、
    v[0] に整数時刻の初速 v0 を入れているため 1 ステップ目がフルキックになり、
    x の精度が 1 次に落ちる（2026-08-30 実測）。新しい学習・評価は
    reference_lf_analytic（半整数時刻の解析解）を使う。
    """
    x = np.empty(steps, dtype=np.float64)
    v = np.empty(steps, dtype=np.float64)
    x[0] = x0
    v[0] = v0
    o2 = omega * omega
    for i in range(steps - 1):
        v[i + 1] = v[i] - o2 * dt * x[i]
        x[i + 1] = x[i] + dt * v[i + 1]
    return x, v


def reference_leapfrog_kdk(
    x0: float,
    v0: float,
    omega: float,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KDK (velocity Verlet) 数値積分（KDK3 系モデルの教師データと一致）.

    半ステップ速度 v_half[i] = v[i] - (ω²dt/2)x[i] も返す（kick1 の教師）。
    reference_leapfrog（KD 型）と異なり v は x と同時刻に同期する。
    """
    x = np.empty(steps, dtype=np.float64)
    v = np.empty(steps, dtype=np.float64)
    v_half = np.empty(steps, dtype=np.float64)
    x[0] = x0
    v[0] = v0
    o2 = omega * omega
    for i in range(steps - 1):
        v_half[i] = v[i] - 0.5 * dt * o2 * x[i]
        x[i + 1] = x[i] + dt * v_half[i]
        v[i + 1] = v_half[i] - 0.5 * dt * o2 * x[i + 1]
    v_half[steps - 1] = v[steps - 1] - 0.5 * dt * o2 * x[steps - 1]
    return x, v, v_half


def reference_analytic_x(
    x0: float,
    v0: float,
    omega: float,
    dt: float,
    steps: int,
) -> np.ndarray:
    """位置の解析解 x(t) = x0 cos(ωt) + (v0/ω) sin(ωt).

    Direct 系は v をモデル出力に含めないため、教師は x のみ。
    v0 は解析解の振幅・位相に反映される。
    """
    t = np.arange(steps, dtype=np.float64) * dt
    if omega == 0.0:
        return np.full(steps, x0, dtype=np.float64)
    return x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)


def reference_analytic_v(
    x0: float,
    v0: float,
    omega: float,
    dt: float,
    steps: int,
) -> np.ndarray:
    """速度の解析解 v(t) = -x0 ω sin(ωt) + v0 cos(ωt)."""
    t = np.arange(steps, dtype=np.float64) * dt
    if omega == 0.0:
        return np.full(steps, v0, dtype=np.float64)
    return -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)


# 教師・評価の参照解の種類
#   analytic:  解析解。LF 系の v は半整数時刻 (n-1/2)Δt、KDK3 系の v_half は (n+1/2)Δt で与える
#   numerical: 全系統で Störmer–Verlet（Leapfrog）法の数値解（2 次精度・シンプレクティック）。
#              LF 系は半ステップずらし（初期速度はハーフキック）、KDK3 系は velocity Verlet、
#              1 段系は同じ解の x 列（x だけで書ける 2 段漸化式 x_{n+1} = 2x_n - x_{n-1} - Δt²ω²x_n）
#   legacy:    2026-09 以前の数値参照解（reference_leapfrog / reference_leapfrog_kdk）
#   fdpos:     位置だけの計測（2026-10-17，調和振動子以外の系のみ）。参照解の位置を整数時刻 t = −Δt, 0, Δt, … で
#              記録し，速度は差分で作る。LF 系は v_{n−1/2} = (x_n − x_{n−1})/Δt，1 段系は v_n = (x_{n+1} − x_{n−1})/(2Δt)。
#              どのモデルも同じ位置の列だけから学ぶ（半ステップの速度の「良い情報」を与えない対照）
TEACHER_CHOICES = ("analytic", "numerical", "legacy", "fdpos")


def primary_ref_for_teacher(teacher: str) -> str:
    """評価の主参照解（--ref auto の解決先）. 主指標は解析解（2026-09-29 に決めた）.

    numerical 教師の run も主指標は解析解で測り、教師（数値解）に対する誤差は
    --ref numerical で eval_ref_numerical/ に補助として出す。legacy は旧来どおり。
    """
    return "analytic" if teacher == "numerical" else teacher


def lf_half_kick_velocity(x0: float, v0: float, omega: float, dt: float) -> float:
    """数値解（Leapfrog）の速度初期値 v_{-1/2} = v0 + (Δt/2)ω²x0（ハーフキック）."""
    return v0 + 0.5 * dt * omega * omega * x0


def reference_lf_numerical(
    x0: float,
    v0: float,
    omega: float,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """LF 系の数値解の教師: 半ステップずらした Leapfrog（v[n] = v_{n-1/2}、2 次精度）.

    漸化式は reference_leapfrog と同じで、v[0] だけをハーフキック v_{-1/2} にする。
    x 列は reference_leapfrog_kdk（velocity Verlet）と一致する。
    """
    x = np.empty(steps, dtype=np.float64)
    v = np.empty(steps, dtype=np.float64)
    x[0] = x0
    v[0] = lf_half_kick_velocity(x0, v0, omega, dt)
    o2 = omega * omega
    for i in range(steps - 1):
        v[i + 1] = v[i] - o2 * dt * x[i]
        x[i + 1] = x[i] + dt * v[i + 1]
    return x, v


def reference_direct_numerical(
    x0: float,
    v0: float,
    omega: float,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """1 段系の数値解の教師: Störmer–Verlet の (x, v)（教師に使うのは x のみ、v は評価用）."""
    x, v, _ = reference_leapfrog_kdk(x0, v0, omega, dt, steps)
    return x, v


def analytic_state(
    x0: float,
    v0: float,
    omega: float,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """任意時刻 t（負や半整数も可）での解析解 (x(t), v(t))."""
    t = np.asarray(t, dtype=np.float64)
    if omega == 0.0:
        return x0 + v0 * t, np.full_like(t, v0)
    c = np.cos(omega * t)
    s = np.sin(omega * t)
    return x0 * c + (v0 / omega) * s, -x0 * omega * s + v0 * c


def lf_initial_velocity(x0: float, v0: float, omega: float, dt: float) -> float:
    """LF 系の速度入力の初期値 v(-Δt/2)（解析解）."""
    _, v = analytic_state(x0, v0, omega, np.array([-0.5 * dt]))
    return float(v[0])


def reference_lf_analytic(
    x0: float,
    v0: float,
    omega: float,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """LF 系の教師・評価用の解析解.

    x[n] = x(nΔt)、v[n] = v((n-1/2)Δt)。v[0] = v(-Δt/2) がモデルの初期速度入力、
    v[n+1] = v((n+1/2)Δt) が n ステップ目の kick の教師になる。
    """
    n = np.arange(steps, dtype=np.float64)
    x, _ = analytic_state(x0, v0, omega, n * dt)
    _, v = analytic_state(x0, v0, omega, (n - 0.5) * dt)
    return x, v


def reference_kdk_analytic(
    x0: float,
    v0: float,
    omega: float,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KDK3 系の教師・評価用の解析解（reference_leapfrog_kdk と同じ並び）.

    x[n] = x(nΔt)、v[n] = v(nΔt)、v_half[n] = v((n+1/2)Δt)。
    """
    n = np.arange(steps, dtype=np.float64)
    x, v = analytic_state(x0, v0, omega, n * dt)
    _, v_half = analytic_state(x0, v0, omega, (n + 0.5) * dt)
    return x, v, v_half


def sync_half_step_velocity(v_half: np.ndarray) -> np.ndarray:
    """半整数時刻の速度列 v[n] = v_{n-1/2} から整数時刻の速度を近似する.

    v_n ≈ (v_{n-1/2} + v_{n+1/2}) / 2 = (v[n] + v[n+1]) / 2（長さ steps-1）。
    LF 系の H・位相を x と同時刻の v で評価するために使う（誤差 O(Δt²)）。
    """
    v_half = np.asarray(v_half, dtype=np.float64)
    return 0.5 * (v_half[:-1] + v_half[1:])


def energy(x: np.ndarray, v: np.ndarray, omega: float) -> np.ndarray:
    """E = 0.5 v^2 + 0.5 ω^2 x^2"""
    return 0.5 * v * v + 0.5 * omega * omega * x * x


def hamiltonian(x: np.ndarray, v: np.ndarray, omega: float) -> np.ndarray:
    """調和振動子のハミルトニアン H = 0.5 v^2 + 0.5 ω^2 x^2（energy と同一）."""
    return energy(x, v, omega)


def phase_angle(x: np.ndarray, v: np.ndarray, omega: float) -> np.ndarray:
    """位相 θ(t) [rad]。x = A cos(θ), v = -Aω sin(θ) より θ = atan2(-v/ω, x).

    値域は atan2 の [-π, π]。連続位相が必要な場合は np.unwrap を併用する。
    """
    if omega == 0.0:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    return np.arctan2(-np.asarray(v, dtype=np.float64) / omega, np.asarray(x, dtype=np.float64))


def wrap_phase(dphi: np.ndarray) -> np.ndarray:
    """位相差を [-π, π) に折り返す."""
    return (np.asarray(dphi, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def trajectory_stats(
    diff: np.ndarray,
    abs_err: np.ndarray,
) -> dict[str, float]:
    """1軌道分の誤差統計."""
    finite = np.isfinite(diff) & np.isfinite(abs_err)
    if not np.any(finite):
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "finite_ratio": 0.0,
        }
    d = diff[finite]
    a = abs_err[finite]
    return {
        "mae": float(np.mean(a)),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "max": float(np.max(a)),
        "std": float(np.std(d)),
        "finite_ratio": float(np.mean(finite)),
    }


def steps_from_t_end(t_end: float, dt: float) -> int:
    return int(round(t_end / dt)) + 1


def t_eval_tag(t_eval: float) -> str:
    """評価時間 [s] をファイル名用タグに変換（例: 10 -> t10s）."""
    if abs(t_eval - round(t_eval)) < 1e-9:
        return f"t{int(round(t_eval))}s"
    s = f"{t_eval:.3f}".rstrip("0").rstrip(".")
    return "t" + s.replace(".", "p") + "s"


def ic_metrics_filename(t_eval: float) -> str:
    return f"ic_generalization_metrics_{t_eval_tag(t_eval)}.json"


def ic_trajectories_filename(
    split: str,
    t_eval: float,
    *,
    selection: str = "worst",
    metric: str = "abs",
) -> str:
    """selection: worst | best,  metric: abs | rel（相対 x_MAE 基準）."""
    tag = t_eval_tag(t_eval)
    if metric not in ("abs", "rel"):
        raise ValueError(f"unknown metric: {metric}")
    if selection not in ("worst", "best"):
        raise ValueError(f"unknown selection: {selection}")
    rel_part = "_rel" if metric == "rel" else ""
    best_part = "_best" if selection == "best" else ""
    return f"eval_{split}_{tag}_trajectories{rel_part}{best_part}.npz"


def ic_trajectory_plot_filename(
    split: str,
    t_eval: float,
    hidden: int,
    *,
    selection: str = "worst",
    metric: str = "abs",
) -> str:
    tag = t_eval_tag(t_eval)
    rel_part = "_rel" if metric == "rel" else ""
    best_part = "_best" if selection == "best" else ""
    return f"plot_traj_{split}_{tag}{rel_part}{best_part}_hs{hidden}.png"


def ic_ham_phase_metrics_filename(
    split: str,
    t_eval: float,
    hidden: int,
    *,
    selection: str = "worst",
    metric: str = "abs",
) -> str:
    """ハミルトニアン・位相比較の数値サマリ JSON 名."""
    tag = t_eval_tag(t_eval)
    rel_part = "_rel" if metric == "rel" else ""
    best_part = "_best" if selection == "best" else ""
    return f"ham_phase_{split}_{tag}{rel_part}{best_part}_hs{hidden}.json"


def ic_ham_phase_plot_filename(
    split: str,
    t_eval: float,
    hidden: int,
    *,
    selection: str = "worst",
    metric: str = "abs",
) -> str:
    """ハミルトニアン・位相比較プロットのファイル名."""
    tag = t_eval_tag(t_eval)
    rel_part = "_rel" if metric == "rel" else ""
    best_part = "_best" if selection == "best" else ""
    return f"plot_ham_phase_{split}_{tag}{rel_part}{best_part}_hs{hidden}.png"


def ic_amplitude(x0: float, v0: float, omega: float) -> float:
    """x(t) の振幅 A = sqrt(x0^2 + (v0/ω)^2). ω=1 なら sqrt(x0^2 + v0^2)."""
    if omega == 0.0:
        return float(abs(x0))
    return float(np.sqrt(x0 * x0 + (v0 / omega) ** 2))


DEFAULT_AMP_FLOOR = 0.05


def effective_amplitude(
    x0: float,
    v0: float,
    omega: float,
    floor: float = DEFAULT_AMP_FLOOR,
) -> float:
    """損失・相対誤差の規格化に用いる振幅（下限付き）."""
    return max(ic_amplitude(x0, v0, omega), floor)


def amplitudes_from_ics(ics: np.ndarray, omega: float, floor: float = DEFAULT_AMP_FLOOR) -> np.ndarray:
    """(N, 2) の IC 配列から A_eff を (N,) で返す."""
    out = np.empty(len(ics), dtype=np.float64)
    for i, (x0, v0) in enumerate(ics):
        out[i] = effective_amplitude(float(x0), float(v0), omega, floor)
    return out


def relative_error_stats(err_stats: dict[str, float], scale: float) -> dict[str, float]:
    """絶対誤差統計を振幅スケールで割った相対誤差統計."""
    if scale <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    rel: dict[str, float] = {}
    for k, v in err_stats.items():
        if k == "finite_ratio":
            rel[k] = v
        elif isinstance(v, (int, float)) and np.isfinite(v):
            rel[k] = float(v / scale)
        else:
            rel[k] = float("nan")
    return rel
