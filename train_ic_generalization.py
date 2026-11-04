#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数初期条件の軌道を混合ミニバッチで学習する（IC汎化用）。

Phase 1: 1-step teacher forcing
損失: harmonic_losses.py の --loss-type で選択（mse / amp_norm_mse 等）

使い方:
  python harmonic_ic_dataset.py --dataset-id mix_v1
  python train_ic_generalization.py --model LF_RNN --dataset-id mix_v1 \\
      --run-id smoke_amp_norm --loss-type amp_norm_mse --hidden-size 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from harmonic_common import ic_gen_model_dir
from harmonic_ic_dataset import is_generic_system, load_manifest, split_scales, split_to_array
from harmonic_losses import (
    default_run_id,
    direct_one_step_loss,
    direct_state_one_step_loss,
    state_one_step_loss,
    kdk3_one_step_loss,
    lf_one_step_loss,
    loss_config_for_run,
    LOSS_REGISTRY,
    validate_loss_type,
)
from harmonic_models import (
    DEFAULT_LTC_ACTIVATION,
    DEFAULT_LTC_VARIANT,
    LTC_ACTIVATIONS,
    LTC_VARIANTS,
    MODEL_CHOICES,
    build_model,
    resolve_model_spec,
)
from harmonic_optim import build_optimizer
from harmonic_ref import (
    DEFAULT_AMP_FLOOR,
    TEACHER_CHOICES,
    amplitudes_from_ics,
    reference_analytic_v,
    reference_analytic_x,
    reference_direct_numerical,
    reference_kdk_analytic,
    reference_leapfrog,
    reference_leapfrog_kdk,
    reference_lf_analytic,
    reference_lf_numerical,
    steps_from_t_end,
)
from harmonic_progress import iter_progress, progress_write, set_postfix
from harmonic_timing import PhaseTimer, timing_filename


DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)


# ミニバッチのシャッフル用の乱数（--seed 指定時は main で seed から作り直す）
_SHUFFLE_RNG = np.random.default_rng()

CHECKPOINT_NAME = "train_checkpoint.pt"
PROGRESS_NAME = "train_progress.json"
# 途中保存から再開してよいかの判定に使わない config のキー（実行環境で変わるもの）
_RESUME_IGNORED_KEYS = {"device"}


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_progress(out_dir: Path, **fields: Any) -> None:
    """見張り役（オーケストレータ）が読む進捗ファイル."""
    doc = {"updated_at": datetime.now().isoformat(timespec="seconds"), "pid": os.getpid(), **fields}
    _atomic_write_text(out_dir / PROGRESS_NAME, json.dumps(doc) + "\n")


def _apply_constraints(model: torch.nn.Module) -> None:
    """原著型 LTC のパラメータの切り詰め（公式実装と同じく学習の 1 ステップごと）. 他のモデルでは何もしない."""
    fn = getattr(model, "apply_constraints", None)
    if fn is not None:
        fn()


def _windows(arrays, starts: np.ndarray, length: int):
    """各軌道から [start, start + length] の区間（length + 1 点）を切り出す. arrays の None はそのまま返す."""
    idx = torch.as_tensor(starts[:, None] + np.arange(length + 1)[None, :])
    rows = torch.arange(len(starts))[:, None]
    return [None if a is None else a[rows, idx] for a in arrays]


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_done: int,
    train_losses: list[float],
    val_losses: list[tuple[int, float]],
    elapsed_s: float,
    config: dict[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch_done": epoch_done,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "elapsed_s": elapsed_s,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "shuffle_rng": _SHUFFLE_RNG.bit_generator.state,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state.update(extra)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    """条件が一致する途中保存だけを返す（不一致なら None で最初から学習）."""
    if not path.exists():
        return None
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    old = {k: v for k, v in state.get("config", {}).items() if k not in _RESUME_IGNORED_KEYS}
    new = {k: v for k, v in config.items() if k not in _RESUME_IGNORED_KEYS}
    if old != new:
        diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        print(f"[resume] checkpoint config mismatch {diff} → 最初から学習します")
        return None
    return state


def _read_trained_log(out_dir: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    """同じ条件で学習を終えた train_log.json があれば返す.

    記録の model_path は学習したときの絶対パスなので，run のフォルダを別の置き場へ移すと（ほかのディスクへ移したときや，
    run の書庫を展開したときなど）指す先が無くなる。コンテナの中で学習した run も，コンテナの中のパスを指す。
    そのときは out_dir の中に同じ名前のモデルのファイルがあれば学習済みと認め，返す model_path を今の置き場に直す（2026-10-24）。
    """
    log = out_dir / "train_log.json"
    if not log.exists():
        return None
    prev = json.loads(log.read_text(encoding="utf-8"))
    same = all(prev.get(k) == v for k, v in config.items() if k not in _RESUME_IGNORED_KEYS)
    if not same:
        return None
    if Path(prev.get("model_path", "")).exists():
        return prev
    name = Path(prev.get("model_path") or "").name
    if name.endswith(".pth") and (out_dir / name).is_file():
        return {**prev, "model_path": str(out_dir / name)}
    return None


def _maybe_clip_grad(model: torch.nn.Module, max_norm: float | None) -> None:
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _build_lf_trajectories(
    ics: np.ndarray,
    omega: float,
    dt: float,
    t_end: float,
    *,
    teacher: str = "analytic",
    progress_desc: str = "",
    disable_progress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LF 系の教師軌道。analytic では v[n] = v((n-1/2)Δt)（v[0] が初期速度入力）、
    numerical では半ステップずらした Leapfrog（v[0] はハーフキック）."""
    ref_fn = {
        "analytic": reference_lf_analytic,
        "numerical": reference_lf_numerical,
    }.get(teacher, reference_leapfrog)
    steps = steps_from_t_end(t_end, dt)
    x = np.zeros((len(ics), steps), dtype=np.float64)
    v = np.zeros((len(ics), steps), dtype=np.float64)
    for i in iter_progress(
        range(len(ics)),
        desc=progress_desc or "teacher trajectories",
        disable=disable_progress,
        unit="ic",
    ):
        x0, v0 = ics[i]
        x[i], v[i] = ref_fn(float(x0), float(v0), omega, dt, steps)
    return torch.from_numpy(x).float(), torch.from_numpy(v).float()


def _build_kdk3_trajectories(
    ics: np.ndarray,
    omega: float,
    dt: float,
    t_end: float,
    *,
    teacher: str = "analytic",
    progress_desc: str = "",
    disable_progress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KDK3 系の教師軌道。analytic では v_half[n] = v((n+1/2)Δt)、numerical / legacy は velocity Verlet."""
    ref_fn = reference_kdk_analytic if teacher == "analytic" else reference_leapfrog_kdk
    steps = steps_from_t_end(t_end, dt)
    x = np.zeros((len(ics), steps), dtype=np.float64)
    v = np.zeros((len(ics), steps), dtype=np.float64)
    vh = np.zeros((len(ics), steps), dtype=np.float64)
    for i in iter_progress(
        range(len(ics)),
        desc=progress_desc or "teacher trajectories (kdk)",
        disable=disable_progress,
        unit="ic",
    ):
        x0, v0 = ics[i]
        x[i], v[i], vh[i] = ref_fn(float(x0), float(v0), omega, dt, steps)
    return (
        torch.from_numpy(x).float(),
        torch.from_numpy(v).float(),
        torch.from_numpy(vh).float(),
    )


def _build_direct_trajectories(
    ics: np.ndarray,
    omega: float,
    dt: float,
    t_end: float,
    *,
    teacher: str = "analytic",
    progress_desc: str = "",
    disable_progress: bool = False,
) -> torch.Tensor:
    steps = steps_from_t_end(t_end, dt)
    x = np.zeros((len(ics), steps), dtype=np.float64)
    for i in iter_progress(
        range(len(ics)),
        desc=progress_desc or "teacher trajectories",
        disable=disable_progress,
        unit="ic",
    ):
        x0, v0 = ics[i]
        if teacher == "numerical":
            # Störmer–Verlet の x 列（x だけの 2 段漸化式と同じ）
            x[i] = reference_direct_numerical(float(x0), float(v0), omega, dt, steps)[0]
        else:
            x[i] = reference_analytic_x(float(x0), float(v0), omega, dt, steps)
    return torch.from_numpy(x).float()


def _build_direct_state_trajectories(
    ics: np.ndarray,
    omega: float,
    dt: float,
    t_end: float,
    *,
    teacher: str = "numerical",
    progress_desc: str = "",
    disable_progress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """1 段系の状態入力版の教師軌道 (x, v)（v は x と同じ整数時刻）.

    numerical: Störmer–Verlet（velocity Verlet）の数値解。x 列は 1 段系・LF 系の数値解と同じ。
    analytic:  解析解。legacy（旧数値参照解）は 2026-09-29 以降に追加した本系統では扱わない。
    """
    if teacher not in ("numerical", "analytic"):
        raise ValueError(f"direct_state 系は teacher=numerical / analytic のみ対応（指定: {teacher}）")
    steps = steps_from_t_end(t_end, dt)
    x = np.zeros((len(ics), steps), dtype=np.float64)
    v = np.zeros((len(ics), steps), dtype=np.float64)
    for i in iter_progress(
        range(len(ics)),
        desc=progress_desc or "teacher trajectories",
        disable=disable_progress,
        unit="ic",
    ):
        x0, v0 = ics[i]
        if teacher == "numerical":
            x[i], v[i] = reference_direct_numerical(float(x0), float(v0), omega, dt, steps)
        else:
            x[i] = reference_analytic_x(float(x0), float(v0), omega, dt, steps)
            v[i] = reference_analytic_v(float(x0), float(v0), omega, dt, steps)
    return torch.from_numpy(x).float(), torch.from_numpy(v).float()


def _build_generic_trajectories(sysm, ics: np.ndarray, family: str, dt: float, t_end: float, teacher: str = "numerical"):
    """調和振動子以外の系の教師（参照解＝刻みの細かい RK4）.

    teacher=fdpos（2026-10-17）：位置だけの計測。速度は位置の差分で作る（systems.System.fd_teacher）。

    LF 系：x は整数時刻，v は半整数時刻 v((n − 1/2)Δt)（旧「解析解の教師」と同じ置き方）。
    状態入力の 1 段系：状態 (N, T, d) をそのまま返す（v 側は None）。
    """
    steps = steps_from_t_end(t_end, dt)
    if teacher == "fdpos":
        if family not in ("lf", "direct_state") or not sysm.second_order:
            raise ValueError(f"teacher=fdpos は 2 階の系の LF 系・1 段系だけに対応（{sysm.name}，{family}）")
        x, v = sysm.fd_teacher(ics, dt, steps, family)
        if family == "direct_state":
            return torch.from_numpy(np.concatenate([x, v], axis=2)).float(), None
        if sysm.n_dof == 1:
            x, v = x[:, :, 0], v[:, :, 0]
        return torch.from_numpy(x).float(), torch.from_numpy(v).float()
    if family == "lf":
        if not sysm.second_order:
            raise ValueError(f"{sysm.name} は 2 階の系ではないので LF 系は使えない")
        k = getattr(sysm, "n_dof", 1)
        x = sysm.integrate(ics, dt, steps)[:, :, :k]
        x = x[:, :, 0] if k == 1 else x  # 多自由度は (N, T, k)
        v = sysm.half_step_velocity(ics, dt, steps)
        return torch.from_numpy(x).float(), torch.from_numpy(v).float()
    if family == "direct_state":
        return torch.from_numpy(sysm.integrate(ics, dt, steps)).float(), None
    raise ValueError(f"{family} 系は調和振動子以外の系に未対応（LF 系か状態入力の 1 段系を使う）")


def _generic_trajectories_cached(manifest, split, sysm, ics, family, dt, t_end, teacher):
    """教師の軌道を datasets/<id>/teacher_cache/ から読む（段数の多い環で毎回の計算が重いとき）.

    ファイルがあるときだけ使う（中身は _build_generic_trajectories と同じ計算の float32）。
    無ければ従来どおり計算する（既存のデータセットの挙動は変わらない）。
    """
    from harmonic_ic_dataset import DATASETS_DIR

    steps = steps_from_t_end(t_end, dt)
    d = Path(DATASETS_DIR) / manifest.get("dataset_id", "") / "teacher_cache"
    fs, fv = d / f"{split}_dt{dt:g}_n{steps}_state.npy", d / f"{split}_dt{dt:g}_n{steps}_vhalf.npy"
    if teacher == "numerical" and family in ("lf", "direct_state") and fs.exists() and (family != "lf" or fv.exists()):
        s_int = np.load(fs)
        if s_int.shape[:2] == (len(ics), steps):
            print(f"  教師の軌道をキャッシュから読む：{fs.name}")
            if family == "direct_state":
                return torch.from_numpy(np.ascontiguousarray(s_int)).float(), None
            k = getattr(sysm, "n_dof", 1)
            x = s_int[:, :, :k]
            v = np.load(fv)
            if k == 1:
                x, v = x[:, :, 0], v[:, :, 0]
            return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(np.ascontiguousarray(v)).float()
    return _build_generic_trajectories(sysm, ics, family, dt, t_end, teacher=teacher)


def _epoch_state_generic(
    model: torch.nn.Module,
    s_all: torch.Tensor,
    scale_all: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    loss_type: str,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_max_norm: float | None = None,
) -> float:
    model.train(optimizer is not None)
    order = indices.copy()
    _SHUFFLE_RNG.shuffle(order)
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        loss = state_one_step_loss(model, s_all[batch_idx].to(DEVICE), scale_all[batch_idx].to(DEVICE), loss_type)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            _maybe_clip_grad(model, grad_clip_max_norm)
            optimizer.step()
            _apply_constraints(model)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def _epoch_lf(
    model: torch.nn.Module,
    x_all: torch.Tensor,
    v_all: torch.Tensor,
    amp_all: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    omega: float,
    loss_type: str,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_max_norm: float | None = None,
) -> float:
    model.train(optimizer is not None)
    rng = _SHUFFLE_RNG
    order = indices.copy()
    rng.shuffle(order)
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        xb = x_all[batch_idx].to(DEVICE)
        vb = v_all[batch_idx].to(DEVICE)
        ab = amp_all[batch_idx].to(DEVICE)
        loss = lf_one_step_loss(model, xb, vb, ab, omega, loss_type)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            _maybe_clip_grad(model, grad_clip_max_norm)
            optimizer.step()
            _apply_constraints(model)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def _epoch_kdk3(
    model: torch.nn.Module,
    x_all: torch.Tensor,
    v_all: torch.Tensor,
    vh_all: torch.Tensor,
    amp_all: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    omega: float,
    loss_type: str,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_max_norm: float | None = None,
) -> float:
    model.train(optimizer is not None)
    rng = _SHUFFLE_RNG
    order = indices.copy()
    rng.shuffle(order)
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        xb = x_all[batch_idx].to(DEVICE)
        vb = v_all[batch_idx].to(DEVICE)
        vhb = vh_all[batch_idx].to(DEVICE)
        ab = amp_all[batch_idx].to(DEVICE)
        loss = kdk3_one_step_loss(model, xb, vb, vhb, ab, omega, loss_type)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            _maybe_clip_grad(model, grad_clip_max_norm)
            optimizer.step()
            _apply_constraints(model)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def _epoch_direct(
    model: torch.nn.Module,
    x_all: torch.Tensor,
    amp_all: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    loss_type: str,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_max_norm: float | None = None,
) -> float:
    model.train(optimizer is not None)
    rng = _SHUFFLE_RNG
    order = indices.copy()
    rng.shuffle(order)
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        xb = x_all[batch_idx].to(DEVICE)
        ab = amp_all[batch_idx].to(DEVICE)
        loss = direct_one_step_loss(model, xb, ab, loss_type)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            _maybe_clip_grad(model, grad_clip_max_norm)
            optimizer.step()
            _apply_constraints(model)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def _epoch_direct_state(
    model: torch.nn.Module,
    x_all: torch.Tensor,
    v_all: torch.Tensor,
    amp_all: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    omega: float,
    loss_type: str,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_max_norm: float | None = None,
) -> float:
    model.train(optimizer is not None)
    rng = _SHUFFLE_RNG
    order = indices.copy()
    rng.shuffle(order)
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        xb = x_all[batch_idx].to(DEVICE)
        vb = v_all[batch_idx].to(DEVICE)
        ab = amp_all[batch_idx].to(DEVICE)
        loss = direct_state_one_step_loss(model, xb, vb, ab, omega, loss_type)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            _maybe_clip_grad(model, grad_clip_max_norm)
            optimizer.step()
            _apply_constraints(model)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def main() -> int:
    p = argparse.ArgumentParser(description="IC汎化: 複数軌道混合学習")
    p.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    p.add_argument("--dataset-id", type=str, default="mix_amp1_v1")
    p.add_argument("--run-id", type=str, default=None, help="実験 run ID（出力サブディレクトリ）")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8, help="軌道数のミニバッチサイズ")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument(
        "--loss-type",
        type=str,
        default="amp_norm_mse",
        choices=list(LOSS_REGISTRY),
    )
    p.add_argument(
        "--amp-floor",
        type=float,
        default=DEFAULT_AMP_FLOOR,
        help="振幅規格化の下限 A_eff = max(A, amp_floor)",
    )
    p.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=None,
        help="勾配クリッピングの max_norm（省略時は無効）。例: 1.0",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam"],
        help="最適化手法（adam）",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="weight decay（Adam の L2）。0 で無効。指定時 run_id に _wd タグ",
    )
    p.add_argument(
        "--teacher",
        type=str,
        default="numerical",  # 2026-09-29: 基本は数値解（モデルの構造に合わせる）。主な評価指標は解析解
        choices=list(TEACHER_CHOICES),
        help="教師軌道。analytic: 解析解（LF の v は半整数時刻、run_id に _ana）/ "
             "numerical: 全系統 Störmer–Verlet 数値解（_num）/ legacy: 旧数値参照解",
    )
    p.add_argument(
        "--ltc-variant",
        type=str,
        default=DEFAULT_LTC_VARIANT,
        choices=list(LTC_VARIANTS),
        help="LTC 更新則（legacy: 旧実装 / paper_fixed_tau: 分母修正のみ / paper: 分母修正＋τ 学習）",
    )
    p.add_argument(
        "--ltc-activation",
        type=str,
        default=DEFAULT_LTC_ACTIVATION,
        choices=list(LTC_ACTIVATIONS),
        help="LTC の f の活性化関数（sigmoid 指定時 run_id に _sig）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="乱数 seed（初期化・ミニバッチのシャッフル）。指定時 run_id に _s<seed>。省略時は固定しない（従来どおり）",
    )
    p.add_argument("--window-len", type=int, default=0,
                   help="0 以外なら，毎エポック各軌道から長さ L のランダムな区間を 1 つ切り出して学習する（原著の BPTT 長 32 に相当）")
    p.add_argument("--fast-onestep", action="store_true",
                   help="教師強制の 1 ステップ損失を時刻のループなしでまとめて計算する（記憶なしの MLP 型だけ。値は同じ，2026-10-21）")
    p.add_argument("--rollout-steps", type=int, default=0,
                   help="0 以外なら，長さ K の自由なロールアウトの損失で学習する（記憶なしの MLP 型だけ，2026-10-22。run_id は --run-id で分ける）")
    p.add_argument("--input-noise", type=float, default=0.0,
                   help="0 以外なら，1 ステップの損失の入力の状態に，変数ごとの二乗平均平方根のこの倍の標準偏差のガウス雑音を足す（--fast-onestep と併用，2026-10-22）")
    p.add_argument("--pos-loss-weight", type=float, default=1.0,
                   help="1 段系（状態入力）の損失で位置の誤差に掛ける重み（2026-10-18。既定 1 は従来どおり）")
    p.add_argument("--best-val", action="store_true",
                   help="毎エポック検証損失を測り，最良のエポックの重みを最終モデルにする（原著の実験と同じ）")
    p.add_argument("--ltc-unfolds", type=int, default=1,
                   help="LTC の展開回数 L（1 入力あたりの fused 更新の回数。原著は 6）")
    p.add_argument("--ltc-int-time", type=float, default=None,
                   help="LTC の 1 入力あたりの積分時間 T（省略時は物理の刻み。原著は入力の周期 = 1）")
    p.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine"],
        help="学習率スケジュール。cosine: エポックごとに cos で lr → lr × --lr-min-ratio まで下げる（run_id に _cos）",
    )
    p.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.01,
        help="cosine の最終学習率 / 初期学習率（既定 0.01）",
    )
    p.add_argument(
        "--ckpt-every",
        type=int,
        default=50,
        help="途中保存の間隔 [epoch]（0 で無効）。途中保存があれば自動で再開する",
    )
    p.add_argument("--no-resume", action="store_true", help="途中保存があっても最初から学習する")
    p.add_argument(
        "--skip-if-trained",
        action="store_true",
        help="同じ条件で学習済み（train_log.json とモデルがある）なら学習を飛ばす（再実行用）",
    )
    p.add_argument("--core-weight", type=float, default=0.0,
                   help="芯の段の重み W（2026-10-24，ソリトン。|x| > 尺度/2 の段とその両隣の誤差に 1 + W。--fast-onestep と loss mse で。run_id は --run-id で分ける）")
    p.add_argument("--adam-eps", type=float, default=None,
                   help="Adam の ε（省略時は torch の既定 1e-8。2026-10-23 夜，振幅の小さいソリトンのデータ用。run_id は --run-id で分ける）")
    p.add_argument("--no-progress", action="store_true", help="tqdm プログレスバーを無効化")
    args = p.parse_args()
    validate_loss_type(args.loss_type)

    run_id = args.run_id or default_run_id(
        args.loss_type,
        args.lr,
        args.batch_size,
        args.grad_clip_max_norm,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        teacher=args.teacher,
        ltc_variant=args.ltc_variant,
        ltc_activation=args.ltc_activation,
        seed=args.seed,
        lr_schedule=args.lr_schedule,
        ltc_unfolds=args.ltc_unfolds,
        ltc_int_time=args.ltc_int_time,
    )

    args.model, spec = resolve_model_spec(args.model)
    hidden = int(args.hidden_size) if args.hidden_size is not None else int(spec["default_hs"])
    family = spec["family"]

    manifest = load_manifest(args.dataset_id)
    omega = float(manifest["omega"])
    t_end = float(manifest["t_end"])

    train_ics = split_to_array(manifest, "train")
    val_ics = split_to_array(manifest, "val")
    train_idx = np.arange(len(train_ics))
    val_idx = np.arange(len(val_ics))

    vh_train = vh_val = None
    generic = is_generic_system(manifest)
    generic_state = generic and family == "direct_state"
    sysm = None
    if generic:
        from systems import system_from_manifest

        sysm = system_from_manifest(manifest)
        dt_key = "dt_lf" if family == "lf" else "dt_direct"
        if manifest.get(dt_key) is None:
            raise ValueError(f"{manifest['system']} には {family} 系の刻み（{dt_key}）が無い")
        dt = float(manifest[dt_key])
        x_train, v_train = _generic_trajectories_cached(manifest, "train", sysm, train_ics, family, dt, t_end, args.teacher)
        x_val, v_val = _generic_trajectories_cached(manifest, "val", sysm, val_ics, family, dt, t_end, args.teacher)
    elif args.teacher == "fdpos":
        raise ValueError("teacher=fdpos は調和振動子以外の系（system_dataset.py のデータセット）だけに対応")
    elif family == "kdk3":
        dt = float(manifest["dt_lf"])
        x_train, v_train, vh_train = _build_kdk3_trajectories(
            train_ics, omega, dt, t_end,
            teacher=args.teacher,
            progress_desc="train trajectories",
            disable_progress=args.no_progress,
        )
        x_val, v_val, vh_val = _build_kdk3_trajectories(
            val_ics, omega, dt, t_end,
            teacher=args.teacher,
            progress_desc="val trajectories",
            disable_progress=args.no_progress,
        )
    elif family == "lf":
        dt = float(manifest["dt_lf"])
        x_train, v_train = _build_lf_trajectories(
            train_ics, omega, dt, t_end,
            teacher=args.teacher,
            progress_desc="train trajectories",
            disable_progress=args.no_progress,
        )
        x_val, v_val = _build_lf_trajectories(
            val_ics, omega, dt, t_end,
            teacher=args.teacher,
            progress_desc="val trajectories",
            disable_progress=args.no_progress,
        )
    elif family == "direct_state":
        dt = float(manifest["dt_direct"])
        x_train, v_train = _build_direct_state_trajectories(
            train_ics, omega, dt, t_end,
            teacher=args.teacher,
            progress_desc="train trajectories",
            disable_progress=args.no_progress,
        )
        x_val, v_val = _build_direct_state_trajectories(
            val_ics, omega, dt, t_end,
            teacher=args.teacher,
            progress_desc="val trajectories",
            disable_progress=args.no_progress,
        )
    else:
        dt = float(manifest["dt_direct"])
        x_train = _build_direct_trajectories(
            train_ics, omega, dt, t_end,
            teacher=args.teacher,
            progress_desc="train trajectories",
            disable_progress=args.no_progress,
        )
        v_train = None
        x_val = _build_direct_trajectories(
            val_ics, omega, dt, t_end,
            teacher=args.teacher,
            progress_desc="val trajectories",
            disable_progress=args.no_progress,
        )
        v_val = None

    if generic:
        # 尺度はデータセットに保存したもの（変数ごと）。LF 系の損失には位置の尺度を使う
        train_scale = torch.from_numpy(split_scales(manifest, "train")).float()
        val_scale = torch.from_numpy(split_scales(manifest, "val")).float()
        train_amp, val_amp = train_scale[:, 0].contiguous(), val_scale[:, 0].contiguous()
    else:
        train_amp = torch.from_numpy(amplitudes_from_ics(train_ics, omega, args.amp_floor)).float()
        val_amp = torch.from_numpy(amplitudes_from_ics(val_ics, omega, args.amp_floor)).float()

    if args.seed is not None:
        global _SHUFFLE_RNG
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)  # CUDA の乱数も含む
        _SHUFFLE_RNG = np.random.default_rng(args.seed)
    model = build_model(
        args.model, hidden, ltc_variant=args.ltc_variant, ltc_activation=args.ltc_activation,
        ltc_unfolds=args.ltc_unfolds, ltc_int_time=args.ltc_int_time,
        dim=int(manifest["dim"]) if generic else None,
    ).to(DEVICE)
    optimizer = build_optimizer(
        model, name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps
    )
    scheduler = None
    if args.lr_schedule == "cosine":
        # 各 param group の初期 lr から同じ比率まで下げる
        min_ratio = args.lr_min_ratio
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda ep: min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * min(ep, args.epochs) / args.epochs)),
        )

    out_dir = ic_gen_model_dir(args.dataset_id, run_id, args.model, hidden)
    config = {
        "script": "train_ic_generalization.py",
        "model": args.model,
        "dataset_id": args.dataset_id,
        "run_id": run_id,
        "hidden_size": hidden,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "teacher": args.teacher,
        "ltc_variant": args.ltc_variant,
        "ltc_activation": args.ltc_activation,
        "seed": args.seed,
        "n_params": sum(p.numel() for p in model.parameters()),
        **loss_config_for_run(
            args.loss_type,
            amp_floor=args.amp_floor,
            grad_clip_max_norm=args.grad_clip_max_norm,
        ),
        "omega": omega,
        "t_end": t_end,
        "dt": dt,
        "family": family,
        **({"system": manifest["system"], "dim": int(manifest["dim"])} if generic else {}),
        "n_train": len(train_ics),
        "n_val": len(val_ics),
        "device": str(DEVICE),
    }
    if args.window_len:
        config["window_len"] = args.window_len
    import harmonic_models as _hm
    if _hm.LOCAL_INPUT_SCALE != 1.0:  # 段ごとの網の入力の尺度（環境変数 BHIC_LOCAL_INPUT_SCALE，2026-10-23 夜）
        config["local_input_scale"] = _hm.LOCAL_INPUT_SCALE
    if args.core_weight:
        import harmonic_losses as _hl
        if not args.fast_onestep:
            raise ValueError("--core-weight は --fast-onestep と併用する")
        _hl.CORE_WEIGHT = float(args.core_weight)
        config["core_weight"] = float(args.core_weight)
    if args.adam_eps is not None:  # 既定のときはキーを足さない（既存 run の判定を変えない）
        config["adam_eps"] = float(args.adam_eps)
    if args.best_val:
        config["best_val"] = True
    if args.fast_onestep:
        import harmonic_losses
        if not harmonic_losses.supports_flat_onestep(model):
            raise ValueError(f"--fast-onestep は記憶なしの MLP 型だけに使える（{args.model}）")
        globals()["lf_one_step_loss"] = harmonic_losses.lf_one_step_loss_flat
        globals()["state_one_step_loss"] = harmonic_losses.state_one_step_loss_flat
        config["fast_onestep"] = True
    if args.rollout_steps or args.input_noise:
        import harmonic_losses
        if args.rollout_steps and args.input_noise:
            raise ValueError("--rollout-steps と --input-noise は同時に使わない（別々の対照）")
        if not harmonic_losses.supports_flat_onestep(model):
            raise ValueError(f"--rollout-steps / --input-noise は記憶なしの MLP 型だけに使える（{args.model}）")
        if family not in ("lf", "direct_state") or (family == "direct_state" and not generic_state):
            raise ValueError("--rollout-steps / --input-noise は調和振動子以外の系の LF 系・状態入力の 1 段系だけ")
        if args.rollout_steps:
            harmonic_losses.ROLLOUT_STEPS = int(args.rollout_steps)
            globals()["lf_one_step_loss"] = harmonic_losses.lf_rollout_loss
            globals()["state_one_step_loss"] = harmonic_losses.state_rollout_loss
            config["rollout_steps"] = int(args.rollout_steps)
        else:
            if not args.fast_onestep:
                raise ValueError("--input-noise は --fast-onestep と併用する")
            if family == "lf":
                xt = x_train if x_train.dim() == 3 else x_train.unsqueeze(-1)
                vt = v_train if v_train.dim() == 3 else v_train.unsqueeze(-1)
                sx = torch.sqrt((xt.double() ** 2).mean(dim=(0, 1))).float()
                sv = torch.sqrt((vt.double() ** 2).mean(dim=(0, 1))).float()
                harmonic_losses.INPUT_NOISE = (args.input_noise * sx, args.input_noise * sv)
            else:
                harmonic_losses.INPUT_NOISE = args.input_noise * torch.sqrt((x_train.double() ** 2).mean(dim=(0, 1))).float()
            config["input_noise"] = float(args.input_noise)
    if args.pos_loss_weight != 1.0:
        import harmonic_losses
        harmonic_losses.POS_LOSS_WEIGHT = float(args.pos_loss_weight)
        config["pos_loss_weight"] = float(args.pos_loss_weight)
    if args.ltc_unfolds != 1 or args.ltc_int_time is not None:
        # 既定のときはキーを足さない（既存 run の skip-if-trained / 再開の判定を変えない）
        config["ltc_unfolds"] = args.ltc_unfolds
        config["ltc_int_time"] = args.ltc_int_time
    if hasattr(model, "delta_t") and abs(float(model.delta_t) - float(dt)) > 1e-12:
        # データの刻みが既定（0.025 / 0.05）と違うデータセットでは，モデル内部の刻みもデータに合わせる（T = Δt）。
        # 調和振動子も同じ（2026-10-05，状態入力の 1 段系を LF と同じ刻みで学習する比較用）。
        # 既定のときはキーを足さない（既存 run の判定を変えない）
        model.delta_t = float(dt)
        config["model_delta_t"] = float(dt)
    if args.lr_schedule != "constant":
        config["lr_schedule"] = args.lr_schedule
        config["lr_min_ratio"] = args.lr_min_ratio
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if args.skip_if_trained:
        prev = _read_trained_log(out_dir, config)
        if prev is not None:
            print(f"[skip] already trained → {prev['model_path']}")
            print("RESULT_JSON " + json.dumps({
                "model": args.model, "dataset_id": args.dataset_id, "run_id": run_id,
                "loss_type": args.loss_type, "hidden_size": hidden, "train_s": prev.get("train_s"),
                "timing": prev.get("timing"), "final_val_loss": prev.get("final_val_loss"),
                "model_path": prev["model_path"], "out_dir": str(out_dir), "skipped": True,
            }))
            return 0

    train_losses: list[float] = []
    val_losses: list[tuple[int, float]] = []
    diverged_at: int | None = None
    best_state, best_va, best_epoch = None, float("inf"), None
    start_epoch = 0
    elapsed_prev = 0.0
    ckpt_path = out_dir / CHECKPOINT_NAME
    state = None if args.no_resume else _load_checkpoint(ckpt_path, config)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch_done"])
        train_losses = list(state["train_losses"])
        val_losses = [tuple(v) for v in state["val_losses"]]
        elapsed_prev = float(state["elapsed_s"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        if "shuffle_rng" in state:
            _SHUFFLE_RNG.bit_generator.state = state["shuffle_rng"]
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if "best_state" in state:
            best_state, best_va, best_epoch = state["best_state"], state["best_va"], state["best_epoch"]
        print(f"[resume] {ckpt_path} から再開: epoch {start_epoch}/{args.epochs}")
        del state

    wd_msg = f" wd={args.weight_decay}" if args.weight_decay else ""
    print(
        f"Training {args.model} hs={hidden} dataset={args.dataset_id} "
        f"run={run_id} loss={args.loss_type} opt={args.optimizer}{wd_msg}"
    )
    clip_msg = f"grad_clip_max_norm={args.grad_clip_max_norm}" if args.grad_clip_max_norm is not None else "grad_clip=off"
    print(f"  train={len(train_ics)} val={len(val_ics)} batch_size={args.batch_size} dt={dt} device={DEVICE} {clip_msg}")

    train_timer = PhaseTimer("train")
    train_timer.start()
    t0 = time.monotonic()
    last_progress = t0
    _write_progress(out_dir, status="training", epoch=start_epoch, epochs=args.epochs,
                    elapsed_s=elapsed_prev, resumed_from=start_epoch or None)
    epoch_bar = iter_progress(
        range(start_epoch, args.epochs),
        desc=f"train {args.model} hs={hidden}",
        total=args.epochs - start_epoch,
        disable=args.no_progress,
        unit="epoch",
    )
    if args.window_len:
        if family == "kdk3":
            raise ValueError("--window-len は kdk3 系では未対応")
        x_train_full, v_train_full = x_train, v_train
        n_tr, T_full = x_train_full.shape[0], x_train_full.shape[1]
        if args.window_len >= T_full:
            raise ValueError(f"--window-len {args.window_len} が系列長 {T_full} 以上")
        # 検証用の区間は seed によらず固定する
        val_starts = np.random.default_rng(20260930).integers(0, x_val.shape[1] - args.window_len, size=x_val.shape[0])
        x_val, v_val = _windows([x_val, v_val], val_starts, args.window_len)
    for epoch in epoch_bar:
        if args.window_len:
            starts = _SHUFFLE_RNG.integers(0, T_full - args.window_len, size=n_tr)
            x_train, v_train = _windows([x_train_full, v_train_full], starts, args.window_len)
        if generic_state:
            tr_loss = _epoch_state_generic(
                model, x_train, train_scale, train_idx, args.batch_size, args.loss_type, optimizer,
                args.grad_clip_max_norm,
            )
        elif family == "kdk3":
            tr_loss = _epoch_kdk3(
                model, x_train, v_train, vh_train, train_amp, train_idx, args.batch_size, omega, args.loss_type,
                optimizer, args.grad_clip_max_norm,
            )
        elif family == "lf":
            tr_loss = _epoch_lf(
                model, x_train, v_train, train_amp, train_idx, args.batch_size, omega, args.loss_type, optimizer,
                args.grad_clip_max_norm,
            )
        elif family == "direct_state":
            tr_loss = _epoch_direct_state(
                model, x_train, v_train, train_amp, train_idx, args.batch_size, omega, args.loss_type, optimizer,
                args.grad_clip_max_norm,
            )
        else:
            tr_loss = _epoch_direct(
                model, x_train, train_amp, train_idx, args.batch_size, args.loss_type, optimizer,
                args.grad_clip_max_norm,
            )
        train_losses.append(tr_loss)
        if scheduler is not None:
            scheduler.step()
        if not np.isfinite(tr_loss):
            diverged_at = epoch + 1
            progress_write(f"[diverged] train loss = {tr_loss} at epoch {diverged_at} → 学習を打ち切ります",
                           disable=False)
            break

        if args.best_val or (epoch + 1) % args.val_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            with torch.no_grad():
                if generic_state:
                    va_loss = _epoch_state_generic(
                        model, x_val, val_scale, val_idx, args.batch_size, args.loss_type, None
                    )
                elif family == "kdk3":
                    va_loss = _epoch_kdk3(
                        model, x_val, v_val, vh_val, val_amp, val_idx, args.batch_size, omega, args.loss_type, None
                    )
                elif family == "lf":
                    va_loss = _epoch_lf(
                        model, x_val, v_val, val_amp, val_idx, args.batch_size, omega, args.loss_type, None
                    )
                elif family == "direct_state":
                    va_loss = _epoch_direct_state(
                        model, x_val, v_val, val_amp, val_idx, args.batch_size, omega, args.loss_type, None
                    )
                else:
                    va_loss = _epoch_direct(
                        model, x_val, val_amp, val_idx, args.batch_size, args.loss_type, None
                    )
            val_losses.append((epoch + 1, va_loss))
            if args.best_val and np.isfinite(va_loss) and va_loss < best_va:
                best_va, best_epoch = va_loss, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            set_postfix(epoch_bar, train=f"{tr_loss:.2e}", val=f"{va_loss:.2e}")
            if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
                progress_write(
                    f"Epoch {epoch + 1:4d}/{args.epochs}  train={tr_loss:.6e}  val={va_loss:.6e}",
                    disable=args.no_progress,
                )

        # 進捗ファイルは 30 秒に 1 回まで毎エポック更新する（速さ・残り時間・停滞の検知用）
        if time.monotonic() - last_progress >= 30:
            last_progress = time.monotonic()
            _write_progress(out_dir, status="training", epoch=epoch + 1, epochs=args.epochs,
                            elapsed_s=elapsed_prev + (time.monotonic() - t0), train_loss=tr_loss)

        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0 and epoch + 1 < args.epochs:
            elapsed = elapsed_prev + (time.monotonic() - t0)
            _save_checkpoint(
                ckpt_path, model=model, optimizer=optimizer, epoch_done=epoch + 1,
                train_losses=train_losses, val_losses=val_losses, elapsed_s=elapsed, config=config,
                scheduler=scheduler,
                extra=({"best_state": best_state, "best_va": best_va, "best_epoch": best_epoch}
                       if best_state is not None else None),
            )
            _write_progress(out_dir, status="training", epoch=epoch + 1, epochs=args.epochs,
                            elapsed_s=elapsed, train_loss=tr_loss)

    train_timer.stop()
    train_s = elapsed_prev + (time.monotonic() - t0)
    train_timer.print_line()
    if args.best_val and best_state is not None:
        model.load_state_dict(best_state)
        progress_write(f"[best-val] epoch {best_epoch} の重みを採用（val={best_va:.6e}）", disable=False)
    model_path = out_dir / f"{args.model}_hs{hidden}.pth"
    torch.save(model.state_dict(), model_path)

    timing = {"train": {**train_timer.to_dict(), "elapsed_s": train_s, "resumed_from_epoch": start_epoch}}
    log = {
        **config,
        "train_s": train_s,
        "timing": timing,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1][1] if val_losses else None,
        "diverged_at_epoch": diverged_at,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": [{"epoch": e, "loss": v} for e, v in val_losses],
        "model_path": str(model_path),
        "out_dir": str(out_dir),
    }
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    (out_dir / timing_filename("train")).write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    ckpt_path.unlink(missing_ok=True)
    _write_progress(out_dir, status="trained", epoch=args.epochs, epochs=args.epochs, elapsed_s=train_s,
                    train_loss=train_losses[-1] if train_losses else None)

    print(f"Done in {train_s:.1f}s → {model_path}")
    result = {
        "model": args.model,
        "dataset_id": args.dataset_id,
        "run_id": run_id,
        "loss_type": args.loss_type,
        "hidden_size": hidden,
        "train_s": train_s,
        "timing": timing,
        "final_val_loss": log["final_val_loss"],
        "model_path": str(model_path),
        "out_dir": str(out_dir),
    }
    print(f"RESULT_JSON {json.dumps(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
