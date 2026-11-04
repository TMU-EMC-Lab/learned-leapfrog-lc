#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IC汎化パイプラインの一括実行（学習 → 評価 → プロット）。
損失は --loss-type で選択（mse / amp_norm_mse）。評価は絶対・相対 MAE/RMSE 両方。

本番の例:
  python harmonic_ic_dataset.py --dataset-id mix_amp1_v1
  python sweep_ic_generalization.py --dataset-id mix_amp1_v1 --run-id amp_norm_lr1e-2_bs8 \\
      --loss-type amp_norm_mse --models all --epochs 1000

軽い起動確認例:
  python sweep_ic_generalization.py --dataset-id mix_amp1_smoke --run-id smoke_amp_norm \\
      --loss-type amp_norm_mse --models LF_RNN --epochs 5 --plot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from harmonic_common import HARMONIC_DIR
from harmonic_losses import LOSS_REGISTRY, default_run_id, validate_loss_type
from harmonic_models import DEFAULT_LTC_ACTIVATION, DEFAULT_LTC_VARIANT, LTC_ACTIVATIONS, LTC_VARIANTS
from harmonic_ref import TEACHER_CHOICES
from harmonic_models import MODEL_REGISTRY, canonical_model_name
from harmonic_progress import iter_progress, progress_write
from harmonic_ref import DEFAULT_AMP_FLOOR


ALL_MODELS = list(MODEL_REGISTRY.keys())


def _run(cmd: list[str], cwd: Path, *, capture_stdout: bool = True) -> str:
    progress_write(f"[exec] {' '.join(cmd)}")
    if capture_stdout:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        stdout_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            stdout_lines.append(line)
        rc = proc.wait()
        stdout = "".join(stdout_lines)
    else:
        rc = subprocess.run(cmd, cwd=str(cwd)).returncode
        stdout = ""
    if rc != 0:
        raise RuntimeError(f"command failed (rc={rc}): {' '.join(cmd)}")
    return stdout


def _parse_result_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :].strip())
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="IC汎化: 学習・評価・プロット一括")
    p.add_argument("--dataset-id", type=str, default="mix_amp1_v1")
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="実験 run ID（未指定時は loss/lr/batch から自動生成）",
    )
    p.add_argument(
        "--loss-type",
        type=str,
        default="amp_norm_mse",
        choices=list(LOSS_REGISTRY),
        help="学習損失（評価指標は常に絶対+相対）",
    )
    p.add_argument("--models", type=str, default="all",
                   help="all またはカンマ区切り (LF_RNN,LF_LTCs,RNN,LTCs。旧名も可)")
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=None,
                   help="省略時は各モデルの default_hs")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--t-eval", type=float, default=None, help="単一評価時間（--t-evals 未指定時の上書き）")
    p.add_argument(
        "--t-evals",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 50.0],
        help="評価時間のリスト [s]（デフォルト: 10 20 50）",
    )
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--plot", action="store_true", help="ic_plot.py を実行")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--n-test-interp", type=int, default=None)
    p.add_argument("--n-test-extrap", type=int, default=None)
    p.add_argument("--sampling-mode", type=str, default=None, choices=["square", "disk"])
    p.add_argument("--max-amplitude", type=float, default=None,
                   help="disk: test_extrap の振幅 A 上限（dataset 生成時）")
    p.add_argument("--train-amp-max", type=float, default=None,
                   help="disk: train/val/test_interp の A 上限（dataset 生成時）")
    p.add_argument("--train-amp-min", type=float, default=None,
                   help="disk: train/val/test_interp の A 下限（省略時 dataset 側デフォルト 0.05）")
    p.add_argument("--test-extrap-amp-min", type=float, default=None,
                   help="disk: test_extrap の A 下限（dataset 生成時）")
    p.add_argument(
        "--amp-floor",
        type=float,
        default=DEFAULT_AMP_FLOOR,
        help="振幅規格化の下限（学習・評価共通）",
    )
    p.add_argument("--fast-onestep", action="store_true",
                   help="学習の 1 ステップ損失を時刻をまとめて計算する（train_ic_generalization.py へ渡す）")
    p.add_argument("--pos-loss-weight", type=float, default=None,
                   help="1 段系の損失で位置の誤差に掛ける重み（train に渡す。省略時は 1）")
    p.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=None,
        help="勾配クリッピングの max_norm（train に渡す。省略時は無効）",
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
        help="weight decay（train に渡す。Adam の L2）。0 で無効",
    )
    p.add_argument(
        "--teacher",
        type=str,
        default="numerical",  # 2026-09-29: 基本は数値解（モデルの構造に合わせる）。主な評価指標は解析解
        choices=list(TEACHER_CHOICES),
        help="教師軌道（train に渡す）。analytic → _ana、numerical → _num",
    )
    p.add_argument(
        "--ltc-variant",
        type=str,
        default=DEFAULT_LTC_VARIANT,
        choices=list(LTC_VARIANTS),
        help="LTC 更新則（train に渡す）。paper_fixed_tau → _ltcden、paper → _ltctau",
    )
    p.add_argument(
        "--ltc-activation",
        type=str,
        default=DEFAULT_LTC_ACTIVATION,
        choices=list(LTC_ACTIVATIONS),
        help="LTC の f の活性化関数（train に渡す）。sigmoid → _sig",
    )
    p.add_argument("--seed", type=int, default=None, help="乱数 seed（train に渡す。run_id に _s<seed>）")
    p.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"],
                   help="学習率スケジュール（train に渡す）。cosine → _cos")
    p.add_argument("--lr-min-ratio", type=float, default=0.01, help="cosine の最終学習率 / 初期学習率（train に渡す）")
    p.add_argument("--ltc-unfolds", type=int, default=1, help="LTC の展開回数 L（train に渡す）")
    p.add_argument("--window-len", type=int, default=0, help="区間での学習（train に渡す）")
    p.add_argument("--rollout-steps", type=int, default=0, help="K ステップのロールアウトの損失（train に渡す，2026-10-22）")
    p.add_argument("--input-noise", type=float, default=0.0, help="1 ステップの損失の入力の雑音（train に渡す，2026-10-22）")
    p.add_argument("--best-val", action="store_true", help="最良の検証損失のエポックを採用（train に渡す）")
    p.add_argument("--ltc-int-time", type=float, default=None, help="LTC の 1 入力あたりの積分時間 T（train に渡す）")
    p.add_argument(
        "--skip-if-trained",
        action="store_true",
        help="同じ条件で学習済みなら学習を飛ばして評価へ（train に渡す。再実行用）",
    )
    p.add_argument("--adam-eps", type=float, default=None, help="Adam の ε（train に渡す，2026-10-23 夜）")
    p.add_argument("--core-weight", type=float, default=0.0, help="芯の段の重み（train に渡す，2026-10-24）")
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
    clip_msg = f" grad_clip={args.grad_clip_max_norm}" if args.grad_clip_max_norm is not None else ""
    wd_msg = f" wd={args.weight_decay}" if args.weight_decay else ""
    print(
        f"[sweep] dataset={args.dataset_id} run_id={run_id} "
        f"loss={args.loss_type} opt={args.optimizer}{clip_msg}{wd_msg}"
    )

    root = HARMONIC_DIR
    py = sys.executable
    models = ALL_MODELS if args.models == "all" else [
        canonical_model_name(m.strip()) for m in args.models.split(",") if m.strip()
    ]
    progress_flag = ["--no-progress"] if args.no_progress else []
    t_list = [args.t_eval] if args.t_eval is not None else (args.t_evals or [10.0])

    tasks: list[tuple[str, dict[str, Any]]] = []
    if not args.skip_dataset:
        tasks.append(("dataset", {}))
    for model in models:
        spec = MODEL_REGISTRY[model]
        hs_list = args.hidden_sizes if args.hidden_sizes else [int(spec["default_hs"])]
        for hs in hs_list:
            if not args.skip_train:
                tasks.append(("train", {"model": model, "hs": hs}))
            if not args.skip_eval:
                tasks.append(("eval", {"model": model, "hs": hs}))
            if args.plot:
                for t_plot in t_list:
                    tasks.append(("plot", {"model": model, "hs": hs, "t_plot": t_plot}))

    task_bar = iter_progress(
        tasks,
        desc=f"sweep {args.dataset_id}/{run_id}",
        total=len(tasks),
        disable=args.no_progress,
        unit="task",
    )

    rows: list[dict[str, Any]] = []

    for phase, ctx in task_bar:
        if phase == "dataset":
            ds_cmd = [py, str(root / "harmonic_ic_dataset.py"), "--dataset-id", args.dataset_id]
            if args.sampling_mode:
                ds_cmd += ["--sampling-mode", args.sampling_mode]
            for flag, val in [
                ("--n-train", args.n_train),
                ("--n-val", args.n_val),
                ("--n-test-interp", args.n_test_interp),
                ("--n-test-extrap", args.n_test_extrap),
                ("--max-amplitude", args.max_amplitude),
                ("--train-amp-min", args.train_amp_min),
                ("--train-amp-max", args.train_amp_max),
                ("--test-extrap-amp-min", args.test_extrap_amp_min),
            ]:
                if val is not None:
                    ds_cmd += [flag, str(val)]
            _run(ds_cmd, root, capture_stdout=False)
            continue

        model = str(ctx["model"])
        hs = int(ctx["hs"])

        if phase == "train":
            train_cmd = [
                py, str(root / "train_ic_generalization.py"),
                "--model", model,
                "--dataset-id", args.dataset_id,
                "--run-id", run_id,
                "--hidden-size", str(hs),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--loss-type", args.loss_type,
                "--amp-floor", str(args.amp_floor),
                *progress_flag,
            ]
            if args.grad_clip_max_norm is not None:
                train_cmd += ["--grad-clip-max-norm", str(args.grad_clip_max_norm)]
            if args.pos_loss_weight is not None:
                train_cmd += ["--pos-loss-weight", str(args.pos_loss_weight)]
            if args.fast_onestep:
                train_cmd.append("--fast-onestep")
            train_cmd += ["--optimizer", args.optimizer]
            if args.weight_decay:
                train_cmd += ["--weight-decay", str(args.weight_decay)]
            train_cmd += [
                "--teacher", args.teacher,
                "--ltc-variant", args.ltc_variant,
                "--ltc-activation", args.ltc_activation,
            ]
            if args.skip_if_trained:
                train_cmd.append("--skip-if-trained")
            if args.seed is not None:
                train_cmd += ["--seed", str(args.seed)]
            if args.lr_schedule != "constant":
                train_cmd += ["--lr-schedule", args.lr_schedule, "--lr-min-ratio", str(args.lr_min_ratio)]
            if args.ltc_unfolds != 1:
                train_cmd += ["--ltc-unfolds", str(args.ltc_unfolds)]
            if args.window_len:
                train_cmd += ["--window-len", str(args.window_len)]
            if args.rollout_steps:
                train_cmd += ["--rollout-steps", str(args.rollout_steps)]
            if args.input_noise:
                train_cmd += ["--input-noise", str(args.input_noise)]
            if args.best_val:
                train_cmd.append("--best-val")
            if args.adam_eps is not None:
                train_cmd += ["--adam-eps", str(args.adam_eps)]
            if args.core_weight:
                train_cmd += ["--core-weight", str(args.core_weight)]
            if args.ltc_int_time is not None:
                train_cmd += ["--ltc-int-time", str(args.ltc_int_time)]
            out = _run(train_cmd, root)
            r = _parse_result_json(out)
            if r:
                rows.append({"phase": "train", **r})
            continue

        if phase == "eval":
            eval_cmd = [
                py, str(root / "ic_generalization_eval.py"),
                "--model", model,
                "--dataset-id", args.dataset_id,
                "--run-id", run_id,
                "--hidden-size", str(hs),
                "--save-npz",
                "--amp-floor", str(args.amp_floor),
                *progress_flag,
            ]
            if args.t_eval is not None:
                eval_cmd += ["--t-eval", str(args.t_eval)]
            elif args.t_evals:
                eval_cmd += ["--t-evals", *[str(t) for t in args.t_evals]]
            _run(eval_cmd, root, capture_stdout=False)
            continue

        if phase == "plot":
            t_plot = float(ctx["t_plot"])
            plot_cmd = [
                py, str(root / "ic_plot.py"), "all",
                "--model", model,
                "--dataset-id", args.dataset_id,
                "--run-id", run_id,
                "--hidden-size", str(hs),
                "--t-eval", str(t_plot),
            ]
            if args.no_show:
                plot_cmd.append("--no-show")
            try:
                _run(plot_cmd, root, capture_stdout=False)
            except RuntimeError as e:
                # 描画は補助。発散した run（NaN・inf）で図が描けなくても学習・評価の結果は残す
                progress_write(f"[warn] plot failed, continue: {e}")

    if rows:
        print("\n=== train summary ===")
        for r in rows:
            timing = r.get("timing", {}).get("train", {})
            started = timing.get("started_at", "-")
            ended = timing.get("ended_at", "-")
            print(
                f"{r.get('model')} hs={r.get('hidden_size')}  "
                f"val_loss={r.get('final_val_loss')}  train_s={r.get('train_s'):.1f}s  "
                f"train_start={started}  train_end={ended}"
            )
    print(f"\n[sweep_ic_generalization] done run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
