#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学習・推論フェーズの実行時間計測ユーティリティ."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any


def now_iso() -> str:
    """ローカルタイムゾーン付き ISO8601 文字列."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class PhaseTimer:
    """1 フェーズの開始・終了時刻と経過秒数を記録する."""

    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self.elapsed_s: float | None = None
        self._t0: float | None = None

    def __enter__(self) -> PhaseTimer:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        self.started_at = now_iso()
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            raise RuntimeError(f"PhaseTimer({self.phase!r}) is not started")
        self.elapsed_s = time.perf_counter() - self._t0
        self.ended_at = now_iso()
        return self.elapsed_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_s": self.elapsed_s,
        }

    def print_line(self) -> None:
        if self.elapsed_s is None:
            raise RuntimeError(f"PhaseTimer({self.phase!r}) has no elapsed time")
        print(
            f"[timing] {self.phase}: "
            f"started_at={self.started_at}  ended_at={self.ended_at}  "
            f"elapsed_s={self.elapsed_s:.4f}"
        )


def timing_filename(prefix: str = "run") -> str:
    return f"{prefix}_timing.json"
