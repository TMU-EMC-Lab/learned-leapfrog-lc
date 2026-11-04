#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初期条件 (x0, v0) をサンプリングし、軌道単位で train/val/test に分割する。

サンプリングモード:
  square  … x0, v0 を矩形 [train_lo, train_hi]^2 から一様（旧 mix_v1）
  disk    … 振幅 A=sqrt(x0^2+(v0/ω)^2) を制限した円盤/円環から一様（面積一様）

disk モード例（train/interp: 0.05<A≤1.0, extrap: 1<A≤1.2）:
  python harmonic_ic_dataset.py --dataset-id mix_amp12k_bs128_v1 --sampling-mode disk \\
      --max-amplitude 1.2 --train-amp-min 0.05 --train-amp-max 1.0 --test-extrap-amp-min 1.0 \\
      --n-train 1024 --n-val 256 --n-test-interp 512 --n-test-extrap 256

出力:
  datasets/<dataset_id>/manifest.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from harmonic_common import HARMONIC_DIR
from harmonic_ref import DEFAULT_AMP_FLOOR, ic_amplitude


DATASETS_DIR = HARMONIC_DIR / "datasets"


@dataclass
class ICSplit:
    x0: list[float]
    v0: list[float]

    def to_dict(self) -> dict[str, list[float]]:
        return {"x0": self.x0, "v0": self.v0}

    @classmethod
    def from_arrays(cls, ics: np.ndarray) -> ICSplit:
        return cls(x0=ics[:, 0].tolist(), v0=ics[:, 1].tolist())


def _is_duplicate(pt: np.ndarray, exclude: np.ndarray | None, tol: float) -> bool:
    if exclude is None or len(exclude) == 0:
        return False
    d = np.linalg.norm(exclude - pt, axis=1)
    return bool(np.any(d < tol))


def _sample_uniform_square(
    n: int,
    lo: float,
    hi: float,
    rng: np.random.Generator,
    exclude: np.ndarray | None = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """[lo, hi]^2 から一様サンプル."""
    out: list[np.ndarray] = []
    max_tries = n * 200
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        pt = rng.uniform(lo, hi, size=2)
        if _is_duplicate(pt, exclude, tol):
            continue
        out.append(pt)
    if len(out) < n:
        raise RuntimeError(f"初期条件を {n} 個生成できませんでした (got {len(out)})")
    return np.array(out, dtype=np.float64)


def _sample_extrap_square(
    n: int,
    inner_lo: float,
    inner_hi: float,
    outer_lo: float,
    outer_hi: float,
    rng: np.random.Generator,
    exclude: np.ndarray | None = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """訓練矩形の外側からサンプル."""
    out: list[np.ndarray] = []
    max_tries = n * 500
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        pt = rng.uniform(outer_lo, outer_hi, size=2)
        x0, v0 = float(pt[0]), float(pt[1])
        if inner_lo <= x0 <= inner_hi and inner_lo <= v0 <= inner_hi:
            continue
        if _is_duplicate(pt, exclude, tol):
            continue
        out.append(pt)
    if len(out) < n:
        raise RuntimeError(f"外挿用初期条件を {n} 個生成できませんでした (got {len(out)})")
    return np.array(out, dtype=np.float64)


def _sample_disk(
    n: int,
    amp_min: float,
    amp_max: float,
    omega: float,
    rng: np.random.Generator,
    exclude: np.ndarray | None = None,
    tol: float = 1e-9,
    *,
    min_open: bool = False,
) -> np.ndarray:
    """円盤/円環から面積一様に (x0, v0) をサンプル.

    半径 r = A * sqrt(u), u~U(0,1) で r in [0, A]。
    円環の場合は r = sqrt(amp_min^2 + u*(amp_max^2 - amp_min^2))。
    min_open=True のとき振幅 A > amp_min のみ採用（A = amp_min は除外）。
    """
    if amp_max <= 0 or amp_min < 0 or amp_min > amp_max:
        raise ValueError(f"invalid amplitude range: [{amp_min}, {amp_max}]")
    if min_open and amp_min >= amp_max:
        raise ValueError(f"open min requires amp_min < amp_max, got [{amp_min}, {amp_max}]")
    out: list[np.ndarray] = []
    max_tries = n * 500
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        u = float(rng.uniform())
        r = float(np.sqrt(amp_min * amp_min + u * (amp_max * amp_max - amp_min * amp_min)))
        if min_open and r <= amp_min + 1e-12:
            continue
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        x0 = r * np.cos(theta)
        v0 = r * omega * np.sin(theta)  # A=r となるよう v0 = r*ω*sinθ
        pt = np.array([x0, v0], dtype=np.float64)
        if _is_duplicate(pt, exclude, tol):
            continue
        out.append(pt)
    if len(out) < n:
        raise RuntimeError(f"ディスクサンプルで {n} 個生成できませんでした (got {len(out)})")
    return np.array(out, dtype=np.float64)


def build_dataset(
    *,
    dataset_id: str,
    omega: float = 1.0,
    t_end: float = 10.0,
    dt_lf: float = 0.05,
    dt_direct: float = 0.025,
    sampling_mode: str = "square",
    train_range: tuple[float, float] = (-2.0, 2.0),
    extrap_outer_range: tuple[float, float] = (-3.0, 3.0),
    max_amplitude: float | None = None,
    train_amp_min: float | None = None,
    train_amp_max: float | None = None,
    test_extrap_amp_min: float | None = None,
    n_train: int = 64,
    n_val: int = 16,
    n_test_interp: int = 32,
    n_test_extrap: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    lo, hi = train_range
    ext_lo, ext_hi = extrap_outer_range

    if sampling_mode == "disk":
        if max_amplitude is None:
            raise ValueError("disk モードでは --max-amplitude が必須です")
        amp_train_max = train_amp_max if train_amp_max is not None else 1.0
        amp_train_min = train_amp_min if train_amp_min is not None else DEFAULT_AMP_FLOOR
        amp_extrap_min = test_extrap_amp_min if test_extrap_amp_min is not None else amp_train_max
        if amp_train_min < 0:
            raise ValueError("train-amp-min は 0 以上にしてください")
        if amp_train_min >= amp_train_max:
            raise ValueError(
                f"train-amp-min は train-amp-max 未満にしてください ({amp_train_min} >= {amp_train_max})"
            )
        if amp_train_max > max_amplitude:
            raise ValueError("train-amp-max は max-amplitude 以下にしてください")
        if amp_extrap_min > max_amplitude:
            raise ValueError("test-extrap-amp-min は max-amplitude 以下にしてください")
        if amp_extrap_min < amp_train_max:
            raise ValueError(
                "test-extrap-amp-min は train-amp-max 以上にしてください"
            )
        interp_min_open = amp_train_min > 1e-12

        train_ics = _sample_disk(
            n_train, amp_train_min, amp_train_max, omega, rng, min_open=interp_min_open
        )
        used = train_ics.copy()
        val_ics = _sample_disk(
            n_val, amp_train_min, amp_train_max, omega, rng, exclude=used, min_open=interp_min_open
        )
        used = np.vstack([used, val_ics])
        test_interp_ics = _sample_disk(
            n_test_interp,
            amp_train_min,
            amp_train_max,
            omega,
            rng,
            exclude=used,
            min_open=interp_min_open,
        )
        used = np.vstack([used, test_interp_ics])
        extrap_min_open = amp_extrap_min <= amp_train_max + 1e-12
        test_extrap_ics = _sample_disk(
            n_test_extrap,
            amp_extrap_min,
            max_amplitude,
            omega,
            rng,
            exclude=used,
            min_open=extrap_min_open,
        )
        sampling_meta = {
            "mode": "disk",
            "max_amplitude": max_amplitude,
            "train_amp_min": amp_train_min,
            "train_amp_min_open": interp_min_open,
            "train_amp_max": amp_train_max,
            "test_extrap_amp_min": amp_extrap_min,
            "test_extrap_amp_min_open": extrap_min_open,
        }
    elif sampling_mode == "square":
        train_ics = _sample_uniform_square(n_train, lo, hi, rng)
        used = train_ics.copy()
        val_ics = _sample_uniform_square(n_val, lo, hi, rng, exclude=used)
        used = np.vstack([used, val_ics])
        test_interp_ics = _sample_uniform_square(n_test_interp, lo, hi, rng, exclude=used)
        used = np.vstack([used, test_interp_ics])
        test_extrap_ics = _sample_extrap_square(
            n_test_extrap, lo, hi, ext_lo, ext_hi, rng, exclude=used
        )
        sampling_meta = {
            "mode": "square",
            "train_range": {"lo": lo, "hi": hi},
            "extrap_outer_range": {"lo": ext_lo, "hi": ext_hi},
        }
    else:
        raise ValueError(f"unknown sampling_mode: {sampling_mode}")

    amps = {
        split: [ic_amplitude(x, v, omega) for x, v in zip(s["x0"], s["v0"])]
        for split, s in {
            "train": ICSplit.from_arrays(train_ics).to_dict(),
            "val": ICSplit.from_arrays(val_ics).to_dict(),
            "test_interp": ICSplit.from_arrays(test_interp_ics).to_dict(),
            "test_extrap": ICSplit.from_arrays(test_extrap_ics).to_dict(),
        }.items()
    }

    manifest: dict[str, Any] = {
        "dataset_id": dataset_id,
        "omega": omega,
        "t_end": t_end,
        "dt_lf": dt_lf,
        "dt_direct": dt_direct,
        "sampling": sampling_meta,
        "seed": seed,
        "counts": {
            "train": n_train,
            "val": n_val,
            "test_interp": n_test_interp,
            "test_extrap": n_test_extrap,
        },
        "amplitude_summary": {
            split: {"min": float(min(a)), "max": float(max(a)), "mean": float(np.mean(a))}
            for split, a in amps.items()
        },
        "splits": {
            "train": ICSplit.from_arrays(train_ics).to_dict(),
            "val": ICSplit.from_arrays(val_ics).to_dict(),
            "test_interp": ICSplit.from_arrays(test_interp_ics).to_dict(),
            "test_extrap": ICSplit.from_arrays(test_extrap_ics).to_dict(),
        },
    }
    return manifest


def save_manifest(manifest: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_manifest(dataset_id: str, datasets_dir: Path | None = None) -> dict[str, Any]:
    base = datasets_dir or DATASETS_DIR
    path = base / dataset_id / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest が見つかりません: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def split_to_array(manifest: dict[str, Any], split: str) -> np.ndarray:
    s = manifest["splits"][split]
    if "ics" in s:  # system_dataset.py（調和振動子以外の系，d 次元）
        return np.asarray(s["ics"], dtype=np.float64)
    return np.column_stack([s["x0"], s["v0"]]).astype(np.float64)


def split_scales(manifest: dict[str, Any], split: str) -> np.ndarray | None:
    """system_dataset.py の尺度 (N, d)（調和振動子の既存データセットには無いので None）."""
    s = manifest["splits"][split]
    return np.asarray(s["scales"], dtype=np.float64) if "scales" in s else None


def is_generic_system(manifest: dict[str, Any]) -> bool:
    """system_dataset.py で作ったデータセット（調和振動子以外の系）か."""
    return "system" in manifest


def main() -> int:
    p = argparse.ArgumentParser(description="IC汎化用データセット (manifest) を生成")
    p.add_argument("--dataset-id", type=str, default="mix_amp1_v1")
    p.add_argument("--sampling-mode", type=str, choices=["square", "disk"], default="disk")
    p.add_argument("--omega", type=float, default=1.0)
    p.add_argument("--t-end", type=float, default=10.0)
    p.add_argument("--dt-lf", type=float, default=0.05)
    p.add_argument("--dt-direct", type=float, default=0.025)
    p.add_argument("--max-amplitude", type=float, default=1.2,
                   help="disk: test_extrap の振幅 A 上限")
    p.add_argument(
        "--train-amp-min",
        type=float,
        default=DEFAULT_AMP_FLOOR,
        help="disk: train/val/test_interp の A 下限（デフォルト 0.05, A>下限）",
    )
    p.add_argument("--train-amp-max", type=float, default=1.0,
                   help="disk: train/val/test_interp の A 上限 (内挿領域)")
    p.add_argument("--test-extrap-amp-min", type=float, default=1.0,
                   help="disk: test_extrap の A 下限 (train-amp-max と同値なら A>下限を厳密化)")
    p.add_argument("--train-lo", type=float, default=-2.0, help="square モード用")
    p.add_argument("--train-hi", type=float, default=2.0, help="square モード用")
    p.add_argument("--extrap-lo", type=float, default=-3.0, help="square モード用")
    p.add_argument("--extrap-hi", type=float, default=3.0, help="square モード用")
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-test-interp", type=int, default=32)
    p.add_argument("--n-test-extrap", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    max_amp = args.max_amplitude if args.sampling_mode == "disk" else None
    manifest = build_dataset(
        dataset_id=args.dataset_id,
        omega=args.omega,
        t_end=args.t_end,
        dt_lf=args.dt_lf,
        dt_direct=args.dt_direct,
        sampling_mode=args.sampling_mode,
        train_range=(args.train_lo, args.train_hi),
        extrap_outer_range=(args.extrap_lo, args.extrap_hi),
        max_amplitude=max_amp,
        train_amp_min=args.train_amp_min if args.sampling_mode == "disk" else None,
        train_amp_max=args.train_amp_max if args.sampling_mode == "disk" else None,
        test_extrap_amp_min=args.test_extrap_amp_min if args.sampling_mode == "disk" else None,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test_interp=args.n_test_interp,
        n_test_extrap=args.n_test_extrap,
        seed=args.seed,
    )
    out_dir = DATASETS_DIR / args.dataset_id
    path = save_manifest(manifest, out_dir)

    print(f"Dataset saved → {path}")
    print(f"- sampling_mode: {args.sampling_mode}")
    if args.sampling_mode == "disk":
        extrap_lo = args.test_extrap_amp_min
        extrap_hi = args.max_amplitude
        if extrap_lo >= extrap_hi:
            extrap_range = f"[{extrap_lo}, {extrap_hi}]"
        elif extrap_lo >= args.train_amp_max - 1e-12:
            extrap_range = f"({extrap_lo}, {extrap_hi}]"
        else:
            extrap_range = f"[{extrap_lo}, {extrap_hi}]"
        interp_range = (
            f"({args.train_amp_min}, {args.train_amp_max}]"
            if args.train_amp_min > 1e-12
            else f"[0, {args.train_amp_max}]"
        )
        print(
            f"- interp_amp={interp_range}, "
            f"extrap_amp={extrap_range}, max_amplitude={args.max_amplitude}"
        )
    else:
        print(f"- train range: [{args.train_lo}, {args.train_hi}]^2")
    print(f"- omega={args.omega}, t_end={args.t_end}, dt_lf={args.dt_lf}, dt_direct={args.dt_direct}")
    key_map = {
        "train": "train",
        "val": "val",
        "test_interp": "test_interp",
        "test_extrap": "test_extrap",
    }
    for split, key in key_map.items():
        s = manifest["amplitude_summary"][split]
        print(
            f"- {split:12s}: n={manifest['counts'][key]:3d}  "
            f"A min/max/mean = {s['min']:.4f} / {s['max']:.4f} / {s['mean']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
