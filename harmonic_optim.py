#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimizer helper for training.

All runs of the ELEX letter use Adam (torch.optim.Adam). This copy keeps only Adam; another optimizer that the original
code base offered for an unrelated study is not included.
"""

from __future__ import annotations

import torch
from torch.optim import Adam, Optimizer


def build_optimizer(
    model: torch.nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
    eps: float | None = None,
) -> Optimizer:
    """Build Adam with the given learning rate.

    eps：Adam の ε（None なら torch の既定 1e-8。2026-10-23 夜，振幅の小さいソリトンのデータで損失が 1e-9 程度になり既定の ε で学習が止まったため）。

    weight_decay: torch.optim.Adam の L2 正則化（論文の学習では 0）。
    """
    name = name.lower()
    if name == "adam":
        if eps is not None:
            return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name!r} (choose adam)")
