#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_harmonic 周辺スクリプト共通ユーティリティ."""

from __future__ import annotations

from pathlib import Path


HARMONIC_DIR = Path(__file__).resolve().parent
PICTURE_DIR = HARMONIC_DIR / "picture"
EXPERIMENTS_DIR = HARMONIC_DIR / "experiments"


# モデル名リネーム（2026-07）: 旧名 → 新名
#   Leapfrog 系: LF_LTCNNs -> LF_LTCs / LF_RNN はそのまま
#   陽的オイラー更新系: LTCNNs -> LTCs / RNN はそのまま
# 旧名は CLI エイリアスおよび既存成果物ディレクトリの fallback として受け付ける。
LEGACY_MODEL_ALIASES: dict[str, str] = {
    "LF_Harmonic_LTCNNs": "LF_LTCs",
    "LF_Harmonic_RNN": "LF_RNN",
    "Direct_Harmonic_LTCNNs_equal": "LTCs",
    "Direct_Harmonic_RNN_equal": "RNN",
    # 2026-07-10: W_hh full 結合化の移行期に使った一時名。
    # 現在は RNN / LTCs 自体が full 結合（旧 rank-1 実装は廃止）。
    "RNN_full": "RNN",
    "LTCs_full": "LTCs",
}


def canonical_model_name(name: str) -> str:
    """旧モデル名を新名称に正規化する（新名称はそのまま返す）."""
    return LEGACY_MODEL_ALIASES.get(name, name)


def legacy_model_names(model: str) -> list[str]:
    """新名称に対応する旧名の一覧."""
    return [old for old, new in LEGACY_MODEL_ALIASES.items() if new == model]


def ic_gen_model_dir(dataset_id: str, run_id: str, model: str, hidden: int) -> Path:
    """IC汎化結果: picture/ic_gen/<dataset_id>/<run_id>/<model>/hs<hidden>/

    新名称のディレクトリを優先し、存在しない場合は旧名の既存ディレクトリに
    fallback する（過去の run を新名称 CLI から参照できるようにするため）。
    """
    model = canonical_model_name(model)
    run_dir = PICTURE_DIR / "ic_gen" / dataset_id / run_id
    d = run_dir / model / f"hs{hidden}"
    if d.exists():
        return d
    for legacy in legacy_model_names(model):
        legacy_dir = run_dir / legacy / f"hs{hidden}"
        if legacy_dir.exists():
            return legacy_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_model_checkpoint(model_dir: Path, model: str, hidden: int) -> Path:
    """model_dir 内のチェックポイントを新名称→旧名の順で探す。

    見つからない場合は新名称のパスを返す（呼び出し側で存在チェック）。
    """
    model = canonical_model_name(model)
    candidates = [model] + legacy_model_names(model)
    for name in candidates:
        p = model_dir / f"{name}_hs{hidden}.pth"
        if p.exists():
            return p
    return model_dir / f"{model}_hs{hidden}.pth"


def ic_gen_run_dir(dataset_id: str, run_id: str) -> Path:
    return PICTURE_DIR / "ic_gen" / dataset_id / run_id


def x0_dir_name(x0: float) -> str:
    """例: 1.0 -> x01p0, 0.5 -> x00p5"""
    if float(x0).is_integer():
        return f"x0{int(x0)}p0"
    s = str(x0).replace("-", "m").replace(".", "p")
    return f"x0{s}"


def candidate_bases(model: str, x0: float) -> list[Path]:
    xdir = x0_dir_name(x0)
    return [
        PICTURE_DIR / xdir / model,
        PICTURE_DIR / xdir / f"harmonic_{model}",
        PICTURE_DIR / f"harmonic_{model}" / xdir,
        PICTURE_DIR / f"harmonic_{model}" / xdir / model,
        PICTURE_DIR / model,
        PICTURE_DIR / f"harmonic_{model}",
    ]
