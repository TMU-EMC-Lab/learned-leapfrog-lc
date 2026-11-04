#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""調和振動子以外の系（system_dataset.py のデータセット）の自己回帰評価（2026-09-30 追加）.

ic_generalization_eval.py から，データセットが調和振動子以外の系のときに呼ばれる（CLI は同じ）。
参照解は systems.py の刻みの細かい RK4。出力の要約は既存と同じ形式（rows の各キー）なので，集計スクリプトをそのまま使える。

初期条件ごとの指標（評価時刻 T まで）:
- x_rmse_rel：位置（第 1 変数）の RMSE ÷ 位置の尺度（データセットの scales）。多自由度の系は位置 k 個の RMS
- state_rmse_rel：全変数を各尺度で割った誤差の RMS
- vpt：全変数を尺度で割った誤差の大きさ（√(変数の平均)）が初めて閾値（既定 0.3）を超える時刻（予測が有効な時間）
- energy_rel：同じ時刻の参照解とのエネルギーの差の時間平均 ÷ 初期エネルギー（単振り子・ダフィングのみ）
LF 系は v を半整数時刻で出すので，状態の指標は隣り合う v の平均で整数時刻にそろえて測る。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import ic_generalization_eval as ev
from harmonic_ic_dataset import split_scales, split_to_array
from harmonic_models import hidden_size_of
from harmonic_ref import ic_metrics_filename, steps_from_t_end, sync_half_step_velocity, t_eval_tag
from harmonic_timing import PhaseTimer, timing_filename
from systems import system_from_manifest

VPT_THRESHOLD = 0.3


def rollout_state(model, s0: np.ndarray, steps: int) -> np.ndarray:
    """状態入力の 1 段系を s0 (N, d) から steps 点（s0 を含む）回す → (N, steps, d)."""
    model.eval()
    n, d = s0.shape
    out = torch.zeros(n, steps, d, device=ev.DEVICE)
    s = torch.as_tensor(np.asarray(s0, dtype=np.float32), device=ev.DEVICE)
    out[:, 0] = s
    h = torch.zeros(n, hidden_size_of(model), device=ev.DEVICE)
    with torch.no_grad():
        for t in range(steps - 1):
            s, h = model(s, h)
            out[:, t + 1] = s
    return out.cpu().numpy().astype(np.float64)


def rollout_lf_multi(model, q0: np.ndarray, v0: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """多自由度の LF 系（2026-10-15）：q0, v0 (N, k)（v0 は v_{−1/2}）から steps 点 → xs, vs (N, steps, k)."""
    model.eval()
    n, k = q0.shape
    xs = torch.zeros(n, steps, k, device=ev.DEVICE)
    vs = torch.zeros(n, steps, k, device=ev.DEVICE)
    xs[:, 0] = torch.as_tensor(np.asarray(q0, dtype=np.float32), device=ev.DEVICE)
    vs[:, 0] = torch.as_tensor(np.asarray(v0, dtype=np.float32), device=ev.DEVICE)
    hidden = hidden_size_of(model)
    h_pp = torch.zeros(n, hidden, device=ev.DEVICE)
    h_p = torch.zeros(n, hidden, device=ev.DEVICE)
    with torch.no_grad():
        for t in range(steps - 1):
            out, h_new, h_p_out = model(torch.cat([xs[:, t], vs[:, t]], dim=1), h_pp, h_p, is_velocity_step=True)
            vs[:, t + 1] = out
            h_pp, h_p = h_p_out, h_new
            out, h_new, h_p_out = model(torch.cat([xs[:, t], vs[:, t + 1]], dim=1), h_pp, h_p, is_velocity_step=False)
            xs[:, t + 1] = out
            h_pp, h_p = h_p_out, h_new
    return xs.cpu().numpy(), vs.cpu().numpy()


def _per_ic(sysm, ref: np.ndarray, pred: np.ndarray, scale: np.ndarray, dt: float, t_eval: float) -> dict[str, np.ndarray]:
    """ref, pred (N, T, d)，scale (N, d) → 初期条件ごとの指標.

    x_rmse_rel は位置の変数（多自由度では k 個）の誤差を各尺度で割った RMS（1 自由度では従来と同じ）。
    """
    k = getattr(sysm, "n_dof", 1) if getattr(sysm, "second_order", False) else 1
    with np.errstate(all="ignore"):
        dx = (pred[:, :, :k] - ref[:, :, :k]) / scale[:, None, :k]
        x_rmse_rel = np.sqrt(np.mean(dx * dx, axis=(1, 2)))
        e = (pred - ref) / scale[:, None, :]
        err_t = np.sqrt(np.mean(e * e, axis=2))  # (N, T)
        state_rmse_rel = np.sqrt(np.mean(err_t * err_t, axis=1))
        bad = ~np.isfinite(err_t) | (err_t > VPT_THRESHOLD)
        first = np.where(bad.any(axis=1), bad.argmax(axis=1), err_t.shape[1])
        vpt = np.minimum(first * dt, t_eval)
        out = {"x_rmse_rel": x_rmse_rel, "state_rmse_rel": state_rmse_rel, "vpt": vpt}
        E_ref = sysm.energy(ref)
        if E_ref is not None:
            # 同じ時刻の参照解のエネルギーと比べ，初期のエネルギーで割る（減衰のある系でも使える）
            E_pred = sysm.energy(pred)
            E0 = np.abs(E_ref[:, 0]) + 1e-12
            out["energy_rel"] = np.mean(np.abs(E_pred - E_ref), axis=1) / E0
    return out


def _agg(v: np.ndarray, limit: float = 100.0) -> dict[str, float]:
    ok = v[np.isfinite(v) & (v <= limit)]
    if len(ok) == 0:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan"), "n_diverged": int(len(v))}
    return {"mean": float(ok.mean()), "median": float(np.median(ok)), "max": float(ok.max()),
            "n_diverged": int(len(v) - len(ok))}


def initial_states(sysm, ics: np.ndarray, dt: float, family: str, teacher: str | None) -> np.ndarray:
    """rollout の初期状態. LF 系は (x_0, v_{−1/2})，1 段系は (x_0, v_0) を (N, 2k) で返す.

    teacher=fdpos（2026-10-17）では学習と同じく位置だけの計測 x_{−1}, x_0, x_1 から差分で作る
    （LF 系 (x_0 − x_{−1})/Δt，1 段系 (x_1 − x_{−1})/(2Δt)。1 段系は x_1 も使うので，比較は 1 段系にやや有利）。
    それ以外は参照解の真の値（LF 系の v_{−1/2} は参照解を半ステップ戻して得る）。
    """
    k = getattr(sysm, "n_dof", 1)
    if teacher == "fdpos":
        x, v = sysm.fd_teacher(ics, dt, 1, family)
        return np.concatenate([x[:, 0], v[:, 0]], axis=1)
    if family == "lf":
        v = sysm.half_step_velocity(ics, dt, 1)[:, 0]
        return np.concatenate([ics[:, :k], v.reshape(len(ics), k)], axis=1)
    return ics


def learned_half_start(model, ics: np.ndarray) -> np.ndarray:
    """参照解の v_{−1/2} を使わずに LF 系を始める（2026-10-28）：v_{−1/2} = v_0 − (Δt/2)·F(x_0, v_0).

    学習した F で半ステップ戻すだけで，真の特性も参照解も使わない。BHIC_LF_LEARNED_START=1 のとき evaluate が使う。
    状態入力に記憶を持つ型（LF-U など hidden_is_dof）は対象外。
    """
    k = ics.shape[1] // 2
    model.eval()
    dt = model.delta_t
    x = torch.as_tensor(np.asarray(ics, dtype=np.float32), device=ev.DEVICE)
    try:
        model.delta_t = -0.5 * dt
        with torch.no_grad():
            v, _, _ = model(x, None, None, is_velocity_step=True)
    finally:
        model.delta_t = dt
    return np.concatenate([ics[:, :k], v.cpu().numpy().astype(np.float64).reshape(len(ics), k)], axis=1)


def cached_reference(sysm, manifest: dict, split: str, ics: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """参照解 (N, steps, d) をデータセットの ref_cache/ に保存して使い回す（2026-10-21：評価の時間の大半が参照解の計算だった）.

    参照解はモデルに依らないので，データセット・分割・刻み・ステップ数ごとに 1 回だけ計算する。保存は float32
    （誤差の相対 1e-7 で，モデルの誤差 1e-3 より十分小さい）。書き込みは一時ファイルからの置き換えで，同時に動くジョブでも壊れない。
    BHIC_NO_REF_CACHE=1 なら保存しない（試験用）。
    """
    import os

    if os.environ.get("BHIC_NO_REF_CACHE") or "dataset_id" not in manifest:
        return sysm.integrate(ics, dt, steps)
    from harmonic_ic_dataset import DATASETS_DIR

    d = Path(DATASETS_DIR) / manifest["dataset_id"] / "ref_cache"
    path = d / f"{split}_dt{dt:g}_n{steps}.npy"
    if path.exists():
        ref = np.load(path).astype(np.float64)
        if ref.shape[:2] == (len(ics), steps):
            return ref
    ref = sysm.integrate(ics, dt, steps)
    d.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
    np.save(tmp, ref.astype(np.float32))
    os.replace(tmp, path)
    return ref


def evaluate(model, family: str, manifest: dict, splits: list[str], t_evals: list[float], out_dir: Path,
             meta: dict[str, Any], teacher: str | None = None) -> list[dict[str, Any]]:
    """t_evals ごとの指標. 参照解とロールアウトは最も長い t_eval で 1 回だけ計算し，短い t_eval では先頭を切り出す
    （2026-10-21。ロールアウトは決定的なので，t_eval ごとに計算し直したときと同じ値になる。tests で確かめた）."""
    sysm = system_from_manifest(manifest)
    dt = float(manifest["dt_lf"] if family == "lf" else manifest["dt_direct"])
    steps_max = max(steps_from_t_end(t, dt) for t in t_evals)
    pre = {}
    for split in splits:
        ics = split_to_array(manifest, split)
        scale = split_scales(manifest, split)
        timer = PhaseTimer(f"ref_{split}")
        timer.start()
        ref_full = cached_reference(sysm, manifest, split, ics, dt, steps_max)
        z0 = initial_states(sysm, ics, dt, family, teacher)
        if family == "lf" and os.environ.get("BHIC_LF_LEARNED_START"):
            z0 = learned_half_start(model, ics)
        if family == "lf" and getattr(sysm, "n_dof", 1) > 1:
            k = sysm.n_dof
            xs, vs = rollout_lf_multi(model, z0[:, :k], z0[:, k:], steps_max)
            roll = (xs.astype(np.float64), vs.astype(np.float64))
        elif family == "lf":
            xs, vs = ev._rollout_lf_batch(model, z0[:, 0], z0[:, 1], steps_max)
            roll = (xs.astype(np.float64), vs.astype(np.float64))
        elif family == "direct_state":
            roll = rollout_state(model, z0, steps_max)
        else:
            raise ValueError(f"{family} 系は調和振動子以外の系に未対応")
        pre[split] = (ics, scale, ref_full, roll, timer.stop())
    rows = []
    for t_eval in t_evals:
        timer = PhaseTimer(f"infer_{t_eval_tag(t_eval)}")
        timer.start()
        steps = steps_from_t_end(t_eval, dt)
        res: dict[str, Any] = {**meta, "t_eval": t_eval, "t_eval_tag": t_eval_tag(t_eval), "dt": dt,
                               "steps_total": steps, "ref_mode": "reference", "splits": {}}
        row: dict[str, Any] = {"t_eval": t_eval, "t_eval_tag": t_eval_tag(t_eval), "steps_total": steps,
                               "metrics_path": ic_metrics_filename(t_eval)}
        for split in splits:
            ics, scale, ref_full, roll, _ = pre[split]
            ref = ref_full[:, :steps]
            if family == "lf" and getattr(sysm, "n_dof", 1) > 1:
                xs, vs = roll[0][:, :steps], roll[1][:, :steps]
                v_sync = 0.5 * (vs[:, :-1] + vs[:, 1:])  # sync_half_step_velocity と同じ（整数時刻 0..steps−2）
                pred = np.concatenate([xs[:, :-1], v_sync], axis=2)
                ref_c = ref[:, :-1]
            elif family == "lf":
                xs, vs = roll[0][:, :steps], roll[1][:, :steps]
                v_sync = np.stack([sync_half_step_velocity(v) for v in vs])  # 整数時刻 0..steps−2
                pred = np.stack([xs[:, :-1], v_sync], axis=2)
                ref_c = ref[:, :-1]
            else:
                pred = roll[:, :steps]
                ref_c = ref
            per = _per_ic(sysm, ref_c, pred, scale, dt, t_eval)
            summ = {k: _agg(v) for k, v in per.items()}
            summ["n"] = int(len(ics))
            res["splits"][split] = {"summary": summ, "per_ic": {k: v.tolist() for k, v in per.items()}}
            for k, v in summ.items():
                if isinstance(v, dict):
                    row[f"{split}_{k}_mean"] = v["mean"]
                    row[f"{split}_{k}_median"] = v["median"]
                    row[f"{split}_{k}_max"] = v["max"]
                    row[f"{split}_{k}_n_diverged"] = v["n_diverged"]
            print(f"[{split}] t={t_eval:g}s  x_RMSE_rel 平均 {summ['x_rmse_rel']['mean']:.4g}  "
                  f"state_RMSE_rel 平均 {summ['state_rmse_rel']['mean']:.4g}  VPT 平均 {summ['vpt']['mean']:.3g}s", flush=True)
        res["infer_s"] = timer.stop() + sum(pre[s_][4] for s_ in splits) / len(t_evals)
        row["infer_s"] = res["infer_s"]
        (out_dir / ic_metrics_filename(t_eval)).write_text(json.dumps(res) + "\n", encoding="utf-8")
        rows.append(row)
    return rows


def main_generic(args, model_name: str, hidden: int, family: str, manifest: dict) -> int:
    """ic_generalization_eval.main から呼ぶ（引数は同じ）."""
    t_evals = ev._resolve_t_eval_list(args.t_eval, args.t_evals, float(manifest["t_end"]))
    model_dir = ev._model_dir(args.dataset_id, args.run_id, model_name, hidden)
    model_path = ev.find_model_checkpoint(model_dir, model_name, hidden)
    if not model_path.exists():
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")
    model, config = ev.load_model_for_eval(model_name, hidden, model_dir, model_path)
    data_dim = int(manifest.get("dim", 2))
    if getattr(model, "length_agnostic", False) and data_dim != int(config.get("dim", data_dim)):
        model.set_length(data_dim // 2)  # 段ごとの網：学習と違う段数の線路で評価する（2026-10-21）
    elif data_dim != int(config.get("dim", data_dim)):
        raise ValueError(f"{model_name} は学習（dim {config.get('dim')}）と違う次元（{data_dim}）では評価できない")
    out_dir = model_dir / args.out_subdir if args.out_subdir else model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": model_name, "dataset_id": args.dataset_id, "run_id": args.run_id, "hidden_size": hidden,
            "data_dataset_id": getattr(args, "data_dataset_id", None) or args.dataset_id,
            "system": manifest["system"], "model_path": str(model_path), "vpt_threshold": VPT_THRESHOLD}
    total = PhaseTimer("infer_total")
    total.start()
    rows = evaluate(model, family, manifest, list(args.splits), t_evals, out_dir, meta, teacher=config.get("teacher"))
    total_s = total.stop()
    timing = {"infer_total": total.to_dict(), "infer_total_s": total_s}
    (out_dir / timing_filename("infer")).write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    summary = {**meta, "dt": float(manifest["dt_lf"] if family == "lf" else manifest["dt_direct"]),
               "ref_mode": "reference", "teacher": config.get("teacher"), "t_evals": t_evals,
               "infer_total_s": total_s, "rows": rows,
               "details": {r["t_eval_tag"]: r["metrics_path"] for r in rows}}
    path = out_dir / "ic_generalization_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary saved → {path}")
    print("RESULT_JSON " + json.dumps({"model": model_name, "run_id": args.run_id, "summary_path": str(path), "rows": rows}))
    return 0
