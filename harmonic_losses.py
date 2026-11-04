#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IC汎化学習用の損失関数プラグイン."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from harmonic_models import hidden_size_of
from harmonic_ref import DEFAULT_AMP_FLOOR


LOSS_REGISTRY: tuple[str, ...] = ("mse", "amp_norm_mse")


def validate_loss_type(loss_type: str) -> str:
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"unknown loss_type={loss_type!r}; choose from {LOSS_REGISTRY}")
    return loss_type


def _lr_tag(lr: float) -> str:
    """1e-2 -> lr1e2, 3e-3 -> lr3e3"""
    s = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    if "e-" in s:
        mant, exp = s.split("e-")
        mant = mant.replace(".0", "").replace(".", "")
        exp = exp.lstrip("0") or "0"
        return f"lr{mant}e{exp}"
    mant, exp = s.split("e+")
    mant = mant.replace(".0", "").replace(".", "")
    return f"lr{mant}e{exp}"


def _grad_clip_tag(grad_clip_max_norm: float) -> str:
    """1.0 -> gc1, 0.5 -> gc0p5"""
    if grad_clip_max_norm == int(grad_clip_max_norm):
        return f"gc{int(grad_clip_max_norm)}"
    s = f"{grad_clip_max_norm:g}".replace(".", "p")
    return f"gc{s}"


def _wd_tag(weight_decay: float) -> str:
    """0.01 -> wd1e2, 0.1 -> wd1e1（_lr_tag と同じ表記）"""
    return "wd" + _lr_tag(weight_decay)[2:]


def default_run_id(
    loss_type: str,
    lr: float,
    batch_size: int,
    grad_clip_max_norm: float | None = None,
    optimizer: str | None = None,
    weight_decay: float | None = None,
    teacher: str | None = None,
    ltc_variant: str | None = None,
    ltc_activation: str | None = None,
    seed: int | None = None,
    lr_schedule: str | None = None,
    ltc_unfolds: int | None = None,
    ltc_int_time: float | None = None,
) -> str:
    """sweep 用: loss/lr/batch から run_id を自動生成.

    2026-09-28 以降の設定はタグで区別する（旧 run と同じ run_id にならないように）:
      teacher=analytic → _ana、numerical → _num、ltc_variant=paper_fixed_tau → _ltcden、paper → _ltctau、
      ltc_activation=sigmoid → _sig、lr_schedule=cosine → _cos、seed=N → _sN。
      legacy / relu / constant / None はタグなし（旧 run_id と一致）。
    """
    if loss_type == "mse":
        prefix = "abs_mse"
    elif loss_type == "amp_norm_mse":
        prefix = "amp_norm"
    else:
        prefix = loss_type
    run_id = f"{prefix}_{_lr_tag(lr)}_bs{batch_size}"
    if grad_clip_max_norm is not None:
        run_id += f"_{_grad_clip_tag(grad_clip_max_norm)}"
    if optimizer is not None and optimizer.lower() not in ("", "adam"):
        run_id += f"_{optimizer.lower()}"
    if weight_decay:
        run_id += f"_{_wd_tag(weight_decay)}"
    if teacher == "analytic":
        run_id += "_ana"
    elif teacher == "numerical":
        run_id += "_num"
    elif teacher == "fdpos":
        run_id += "_fdpos"
    if ltc_variant == "paper_fixed_tau":
        run_id += "_ltcden"
    elif ltc_variant == "paper":
        run_id += "_ltctau"
    if ltc_activation == "sigmoid":
        run_id += "_sig"
    if (ltc_unfolds and int(ltc_unfolds) != 1) or ltc_int_time is not None:
        # 例: L = 6，T = 1 → _u6T1。既定（L = 1，T = 物理の刻み）はタグなし
        run_id += f"_u{int(ltc_unfolds or 1)}"
        if ltc_int_time is not None:
            run_id += "T" + f"{float(ltc_int_time):g}".replace(".", "p")
    if lr_schedule == "cosine":
        run_id += "_cos"
    if seed is not None:
        run_id += f"_s{seed}"
    return run_id


def loss_config_for_run(
    loss_type: str,
    *,
    amp_floor: float = DEFAULT_AMP_FLOOR,
    **extra: Any,
) -> dict[str, Any]:
    validate_loss_type(loss_type)
    cfg: dict[str, Any] = {"loss_type": loss_type, "amp_floor": amp_floor}
    cfg.update(extra)
    return cfg


def _mse_step(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def _norm_mse_step(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    err2 = (pred - target) ** 2
    return (err2 / (scale * scale)).mean()


StepLossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]


def _make_step_loss(loss_type: str) -> StepLossFn:
    validate_loss_type(loss_type)

    def step_abs(pred: torch.Tensor, target: torch.Tensor, amp_eff: torch.Tensor, omega: float) -> torch.Tensor:
        del amp_eff, omega
        return _mse_step(pred, target)

    def step_amp_norm(
        pred: torch.Tensor, target: torch.Tensor, amp_eff: torch.Tensor, omega: float, *, velocity: bool
    ) -> torch.Tensor:
        scale = (omega * amp_eff) if velocity else amp_eff
        return _norm_mse_step(pred, target, scale.view(-1, 1))

    if loss_type == "mse":
        return step_abs
    return lambda pred, target, amp_eff, omega: step_amp_norm(
        pred, target, amp_eff, omega, velocity=False
    )


def _make_step_loss_lf_v(loss_type: str) -> StepLossFn:
    validate_loss_type(loss_type)
    if loss_type == "mse":
        return lambda pred, target, amp_eff, omega: _mse_step(pred, target)

    return lambda pred, target, amp_eff, omega: _norm_mse_step(
        pred, target, (omega * amp_eff).view(-1, 1)
    )


def lf_one_step_loss(
    model: nn.Module,
    x_true: torch.Tensor,
    v_true: torch.Tensor,
    amp_eff: torch.Tensor,
    omega: float,
    loss_type: str,
) -> torch.Tensor:
    step_x = _make_step_loss(loss_type)
    step_v = _make_step_loss_lf_v(loss_type)
    if x_true.dim() == 2:  # 1 自由度 (B, T) → (B, T, 1)。多自由度の系（2026-10-15）は (B, T, k)
        x_true, v_true = x_true.unsqueeze(-1), v_true.unsqueeze(-1)
    bsz, tmax = x_true.shape[:2]
    hidden = hidden_size_of(model)
    h_prev_prev = torch.zeros(bsz, hidden, device=x_true.device)
    h_prev = torch.zeros(bsz, hidden, device=x_true.device)
    substeps = (tmax - 1) * 2
    total = torch.zeros((), device=x_true.device)
    for i in range(substeps):
        if i % 2 == 0:
            t = i // 2
            x_in = torch.cat([x_true[:, t], v_true[:, t]], dim=1)
            out, h_new, h_old = model(x_in, h_prev_prev, h_prev, is_velocity_step=True)
            total = total + step_v(out, v_true[:, t + 1], amp_eff, omega)
            h_prev_prev = h_old
            h_prev = h_new
        else:
            t = i // 2
            x_in = torch.cat([x_true[:, t], v_true[:, t + 1]], dim=1)
            out, h_new, h_old = model(x_in, h_prev_prev, h_prev, is_velocity_step=False)
            total = total + step_x(out, x_true[:, t + 1], amp_eff, omega)
            h_prev_prev = h_old
            h_prev = h_new
    return total / substeps


def kdk3_one_step_loss(
    model: nn.Module,
    x_true: torch.Tensor,
    v_true: torch.Tensor,
    vh_true: torch.Tensor,
    amp_eff: torch.Tensor,
    omega: float,
    loss_type: str,
) -> torch.Tensor:
    """KDK3 系: 1 タイムステップ = kick1 / drift / kick2 の 3 substep を教師強制で監督.

    kick1: (x_n, v_n) → v_half_n、drift: (x_n, v_half_n) → x_{n+1}、
    kick2: (x_{n+1}, v_half_n) → v_{n+1}。教師は reference_leapfrog_kdk。
    速度系（kick1/kick2）の規格化スケールは ωA、位置系（drift）は A。
    """
    step_x = _make_step_loss(loss_type)
    step_v = _make_step_loss_lf_v(loss_type)
    bsz, tmax = x_true.shape
    hidden = hidden_size_of(model)
    h_prev_prev = torch.zeros(bsz, hidden, device=x_true.device)
    h_prev = torch.zeros(bsz, hidden, device=x_true.device)
    substeps = (tmax - 1) * 3
    total = torch.zeros((), device=x_true.device)
    for i in range(substeps):
        t = i // 3
        sub = i % 3
        if sub == 0:
            x_in = torch.cat([x_true[:, t : t + 1], v_true[:, t : t + 1]], dim=1)
            out, h_new, h_old = model(x_in, h_prev_prev, h_prev, step_id=0)
            total = total + step_v(out, vh_true[:, t : t + 1], amp_eff, omega)
        elif sub == 1:
            x_in = torch.cat([x_true[:, t : t + 1], vh_true[:, t : t + 1]], dim=1)
            out, h_new, h_old = model(x_in, h_prev_prev, h_prev, step_id=1)
            total = total + step_x(out, x_true[:, t + 1 : t + 2], amp_eff, omega)
        else:
            x_in = torch.cat([x_true[:, t + 1 : t + 2], vh_true[:, t : t + 1]], dim=1)
            out, h_new, h_old = model(x_in, h_prev_prev, h_prev, step_id=2)
            total = total + step_v(out, v_true[:, t + 1 : t + 2], amp_eff, omega)
        h_prev_prev = h_old
        h_prev = h_new
    return total / substeps


# 1 段系（状態入力）の損失で位置の誤差に掛ける重み（2026-10-18，ELEX：Res の損失の釣り合いを確かめる）。
# train_ic_generalization.py の --pos-loss-weight が設定する。既定の 1 は従来どおり（位置と速度を同じ重み）。
POS_LOSS_WEIGHT = 1.0


def direct_state_one_step_loss(
    model: nn.Module,
    x_true: torch.Tensor,
    v_true: torch.Tensor,
    amp_eff: torch.Tensor,
    omega: float,
    loss_type: str,
) -> torch.Tensor:
    """1 段系の状態入力版: (x_t, v_t) → (x_{t+1}, v_{t+1}) を教師強制で監督.

    x と v の誤差は LF 系と同じ関数で測り（amp_norm では v を ωA で規格化）、
    LF 系が 1 サブステップに 1 変数を監督するのに合わせて 2 × (T−1) で平均する。
    """
    step_x = _make_step_loss(loss_type)
    step_v = _make_step_loss_lf_v(loss_type)
    bsz, tmax = x_true.shape
    hidden = hidden_size_of(model)
    h = torch.zeros(bsz, hidden, device=x_true.device)
    total = torch.zeros((), device=x_true.device)
    for t in range(tmax - 1):
        xv_in = torch.cat([x_true[:, t : t + 1], v_true[:, t : t + 1]], dim=1)
        if hasattr(model, "teacher_step"):  # 2 つの網の対照（Res2・SIE）：位置の段も教師強制（LF 系と同じ）
            out = model.teacher_step(xv_in, torch.cat([x_true[:, t + 1 : t + 2], v_true[:, t + 1 : t + 2]], dim=1))
        else:
            out, h = model(xv_in, h)
        total = total + POS_LOSS_WEIGHT * step_x(out[:, 0:1], x_true[:, t + 1 : t + 2], amp_eff, omega)
        total = total + step_v(out[:, 1:2], v_true[:, t + 1 : t + 2], amp_eff, omega)
    return total / (2 * (tmax - 1))


def state_one_step_loss(
    model: nn.Module,
    s_true: torch.Tensor,
    scale: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """d 次元の状態入力の 1 段系（調和振動子以外の系，2026-09-30）: s_t → s_{t+1} を教師強制で監督.

    s_true (B, T, d)，scale (B, d)。mse は全変数の二乗誤差の平均，amp_norm_mse は変数ごとに scale で割る。
    変数とステップで平均するので，d = 2 のとき direct_state_one_step_loss と同じ規格化になる。
    """
    validate_loss_type(loss_type)
    bsz, tmax, _ = s_true.shape
    h = torch.zeros(bsz, hidden_size_of(model), device=s_true.device)
    total = torch.zeros((), device=s_true.device)
    for t in range(tmax - 1):
        if hasattr(model, "teacher_step"):  # 2 つの網の対照（Res2・SIE）：位置の段も教師強制（LF 系と同じ）
            out = model.teacher_step(s_true[:, t], s_true[:, t + 1])
        else:
            out, h = model(s_true[:, t], h)
        err = out - s_true[:, t + 1]
        if loss_type == "amp_norm_mse":
            err = err / scale
        if POS_LOSS_WEIGHT != 1.0:  # 前半の d/2 個が位置
            w = torch.ones(err.shape[1], device=err.device)
            w[: err.shape[1] // 2] = POS_LOSS_WEIGHT
            total = total + (w * err * err).mean()
            continue
        total = total + (err * err).mean()
    return total / (tmax - 1)


# 2026-10-22（ELEX：学習の手順が差を作っているかの検証）。train_ic_generalization.py が設定する。
INPUT_NOISE = None  # None，または入力に足す雑音の標準偏差（LF 系は (x 用, v 用) の組，状態入力は状態のテンソル。変数ごと）
ROLLOUT_STEPS = 0  # 0 なら 1 ステップの損失。K > 0 なら長さ K の自由なロールアウトの損失（lf_rollout_loss，state_rollout_loss）


def _window_starts(tmax: int, k_steps: int, device) -> torch.Tensor:
    """長さ K の区間の始まり 0, K, 2K, ...（s + K ≤ tmax − 1）。軌道のどの時刻も 1 つの区間に 1 回だけ入る."""
    return torch.arange(0, tmax - k_steps, k_steps, device=device)


def lf_rollout_loss(
    model: nn.Module,
    x_true: torch.Tensor,
    v_true: torch.Tensor,
    amp_eff: torch.Tensor,
    omega: float,
    loss_type: str,
) -> torch.Tensor:
    """LF 系の K ステップの自由なロールアウトの損失（2026-10-22）.

    各軌道を長さ K の区間に分け，区間の始めの厳密な (x_s, v_{s−1/2}) から，網の出力だけで K ステップ進める
    （位置の段には予測した v を渡す）。各ステップの速度と位置の誤差を 1 ステップの損失と同じ重み（半々）で平均する。
    誤差は区間の中で積み重なるので，1 ステップの損失では罰しにくい符号のそろったずれも罰する。
    """
    k_steps = ROLLOUT_STEPS
    step_x = _make_step_loss(loss_type)
    step_v = _make_step_loss_lf_v(loss_type)
    if x_true.dim() == 2:
        x_true, v_true = x_true.unsqueeze(-1), v_true.unsqueeze(-1)
    bsz, tmax, k = x_true.shape
    st = _window_starts(tmax, k_steps, x_true.device)
    w = len(st)
    x, v = x_true[:, st].reshape(bsz * w, k), v_true[:, st].reshape(bsz * w, k)
    amp = amp_eff.repeat_interleave(w, dim=0)
    total = torch.zeros((), device=x_true.device)
    for j in range(1, k_steps + 1):
        # 隠れ状態を 1 ステップの損失・ロールアウトと同じく渡す（2026-10-24：LF-U は位置の段で更新前の速度を使う。
        # 以前は None を渡していて LF-U が LF と同じ形になっていた。LF・LF-KCL などは隠れ状態を使わないので値は変わらない）
        v, h_new, h_old = model(torch.cat([x, v], dim=1), None, None, is_velocity_step=True)
        x, _, _ = model(torch.cat([x, v], dim=1), h_old, h_new, is_velocity_step=False)
        tv, tx = v_true[:, st + j].reshape(bsz * w, k), x_true[:, st + j].reshape(bsz * w, k)
        total = total + 0.5 * (step_v(v, tv, amp, omega) + step_x(x, tx, amp, omega))
    return total / k_steps


def state_rollout_loss(
    model: nn.Module,
    s_true: torch.Tensor,
    scale: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """状態入力の 1 段系の K ステップの自由なロールアウトの損失（2026-10-22）. 2 つの網の型も forward（予測した v_{n+1}）で進める."""
    validate_loss_type(loss_type)
    k_steps = ROLLOUT_STEPS
    bsz, tmax, d = s_true.shape
    st = _window_starts(tmax, k_steps, s_true.device)
    w = len(st)
    s = s_true[:, st].reshape(bsz * w, d)
    sc = scale.repeat_interleave(w, dim=0)
    h = torch.zeros(bsz * w, hidden_size_of(model), device=s_true.device)
    total = torch.zeros((), device=s_true.device)
    for j in range(1, k_steps + 1):
        s, h = model(s, h)
        err = s - s_true[:, st + j].reshape(bsz * w, d)
        if loss_type == "amp_norm_mse":
            err = err / sc
        total = total + (err * err).mean()
    return total / k_steps


# 2026-10-24（ソリトンの試験の学習し直し）：芯の段の重み。0 なら従来どおり。W > 0 なら，入力の |x|（|v|）が軌道の尺度の半分を超える段と
# その両隣（環）の誤差に 1 + W を掛ける（静かな段は 1）。lf_one_step_loss_flat と state_one_step_loss_flat だけが使う。
CORE_WEIGHT = 0.0


def _core_w(z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """z (M, n)，scale (M, 1) → 重み (M, n) = 1 + W·[|z| > scale/2 とその両隣]."""
    m = (z.abs() > 0.5 * scale).float()
    m = torch.clamp(m + torch.roll(m, 1, dims=1) + torch.roll(m, -1, dims=1), max=1.0)
    return 1.0 + CORE_WEIGHT * m


def supports_flat_onestep(model: nn.Module) -> bool:
    """時刻をまとめて計算してよいモデルか（各時刻の 1 ステップが前の時刻の状態に依らない，記憶なしの MLP 型）."""
    from harmonic_models import _MemorylessLF, _ResidualState, _TwoNetState, RK4KinMLPState

    if isinstance(model, _MemorylessLF):
        return getattr(model, "kind", "") == "mlp"
    if isinstance(model, _ResidualState):
        return getattr(model, "kind", "") == "mlp"
    return isinstance(model, (_TwoNetState, RK4KinMLPState))


def lf_one_step_loss_flat(
    model: nn.Module,
    x_true: torch.Tensor,
    v_true: torch.Tensor,
    amp_eff: torch.Tensor,
    omega: float,
    loss_type: str,
) -> torch.Tensor:
    """lf_one_step_loss と同じ値を，時刻のループを使わずに 2 回の呼び出しで計算する（2026-10-21，GPU で速くするため）.

    各時刻の速度の段・位置の段は教師強制で前の時刻の出力に依らないので，(B, T − 1) を 1 つのバッチにまとめてよい。
    元の損失は時刻ごとの平均の平均（どの時刻もバッチの大きさは同じ）なので，まとめた平均と数学的に等しい。
    """
    step_x = _make_step_loss(loss_type)
    step_v = _make_step_loss_lf_v(loss_type)
    if x_true.dim() == 2:
        x_true, v_true = x_true.unsqueeze(-1), v_true.unsqueeze(-1)
    bsz, tmax, k = x_true.shape
    n = bsz * (tmax - 1)
    x_t, x_n = x_true[:, :-1].reshape(n, k), x_true[:, 1:].reshape(n, k)
    v_t, v_n = v_true[:, :-1].reshape(n, k), v_true[:, 1:].reshape(n, k)
    if INPUT_NOISE is not None and model.training:  # 入力の状態だけに雑音（目標は厳密なまま）。位置の段の速度は参照の v_{n+1/2}
        sx, sv = INPUT_NOISE
        x_t = x_t + sx.to(x_t.device) * torch.randn_like(x_t)
        v_t = v_t + sv.to(v_t.device) * torch.randn_like(v_t)
    amp = amp_eff.repeat_interleave(tmax - 1, dim=0)
    out_v, h_new, h_old = model(torch.cat([x_t, v_t], dim=1), None, None, is_velocity_step=True)
    out_x, _, _ = model(torch.cat([x_t, v_n], dim=1), h_old, h_new, is_velocity_step=False)
    if CORE_WEIGHT > 0:  # 芯の段の重み（2026-10-24）。尺度は軌道の位置の尺度（磁束にも同じものを使う）
        if loss_type != "mse":
            raise ValueError("芯の重みは loss_type=mse だけ")
        sc_ = amp.view(-1, 1)
        wx, wv = _core_w(x_t, sc_), _core_w(v_t, sc_)
        return 0.5 * ((wv * (out_v - v_n) ** 2).mean() + (wx * (out_x - x_n) ** 2).mean())
    return 0.5 * (step_v(out_v, v_n, amp, omega) + step_x(out_x, x_n, amp, omega))


def state_one_step_loss_flat(
    model: nn.Module,
    s_true: torch.Tensor,
    scale: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """state_one_step_loss と同じ値を，時刻のループを使わずに 1 回の呼び出しで計算する（2026-10-21）."""
    validate_loss_type(loss_type)
    bsz, tmax, d = s_true.shape
    n = bsz * (tmax - 1)
    s_t, s_n = s_true[:, :-1].reshape(n, d), s_true[:, 1:].reshape(n, d)
    if INPUT_NOISE is not None and model.training:  # 入力の状態だけに雑音（目標は厳密なまま）
        s_t = s_t + INPUT_NOISE.to(s_t.device) * torch.randn_like(s_t)
    if hasattr(model, "teacher_step"):
        out = model.teacher_step(s_t, s_n)
    else:
        out, _ = model(s_t, torch.zeros(n, hidden_size_of(model), device=s_true.device))
    err = out - s_n
    if CORE_WEIGHT > 0:  # 芯の段の重み（2026-10-24）
        if loss_type != "mse":
            raise ValueError("芯の重みは loss_type=mse だけ")
        k = d // 2
        scr = scale.repeat_interleave(tmax - 1, dim=0)
        w = torch.cat([_core_w(s_t[:, :k], scr[:, :1]), _core_w(s_t[:, k:], scr[:, :1])], dim=1)
        return (w * err * err).mean()
    if loss_type == "amp_norm_mse":
        err = err / scale.repeat_interleave(tmax - 1, dim=0)
    if POS_LOSS_WEIGHT != 1.0:
        w = torch.ones(err.shape[1], device=err.device)
        w[: err.shape[1] // 2] = POS_LOSS_WEIGHT
        return (w * err * err).mean()
    return (err * err).mean()


def direct_one_step_loss(
    model: nn.Module,
    x_true: torch.Tensor,
    amp_eff: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    step_x = _make_step_loss(loss_type)
    bsz, tmax = x_true.shape
    hidden = hidden_size_of(model)
    h = torch.zeros(bsz, hidden, device=x_true.device)
    total = torch.zeros((), device=x_true.device)
    for t in range(tmax - 1):
        x_in = x_true[:, t : t + 1]
        out, h = model(x_in, h)
        total = total + step_x(out, x_true[:, t + 1 : t + 2], amp_eff, 1.0)
    return total / (tmax - 1)
