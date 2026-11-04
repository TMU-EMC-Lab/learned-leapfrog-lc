#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""既存のデータセットから派生のデータセットを作る（2026-10-02，--dt-direct は調和振動子にも使える）.

検証・テストの初期条件はそのままにして，次だけを変える。
- --n-train N：学習の初期条件を先頭の N 本に減らす（初期条件は乱数で引いた順なので，先頭 N 本は無作為な部分集合）
- --dt-scale K：刻み dt_direct・dt_lf を K 倍にする（参照解は刻み 1e-3 の RK4 から取り直すので，教師もその刻みになる）
- --dt-direct D：1 段系（状態入力）の刻み dt_direct だけを D にする（2026-10-05。LF と刻みをそろえる比較用。調和振動子のデータセットにも使える）
- --dt-lf D：LF 系の刻み dt_lf を D にする（2026-10-05。--dt-direct と合わせて刻みの掃引に使う。調和振動子にも使える）

  python scripts/derive_dataset.py --from duffing_v1 --to duffing_v1_n128 --n-train 128
  python scripts/derive_dataset.py --from duffing_v1 --to duffing_v1_dt4 --dt-scale 4
  python scripts/derive_dataset.py --from mix_amp12k_v1 --to mix_amp12k_v1_dtd2 --dt-direct 0.05
  python scripts/derive_dataset.py --from pend_v1 --to pend_v1_dt0p2 --dt-direct 0.2 --dt-lf 0.2
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonic_ic_dataset import DATASETS_DIR  # noqa: E402


def derive(src: dict, dst_id: str, n_train: int | None, dt_scale: float | None, dt_direct: float | None = None,
           dt_lf: float | None = None) -> dict:
    m = copy.deepcopy(src)
    if "system" not in m and (n_train is not None or dt_scale is not None):
        raise ValueError("--n-train / --dt-scale は調和振動子以外の系（system_dataset.py のデータセット）だけに対応")
    m["dataset_id"] = dst_id
    m["derived_from"] = {"dataset_id": src["dataset_id"]}
    if n_train is not None:
        tr = m["splits"]["train"]
        n_all = len(tr["ics"])
        if not 0 < n_train <= n_all:
            raise ValueError(f"--n-train は 1〜{n_all}")
        for k, v in tr.items():
            if isinstance(v, list) and len(v) == n_all:
                tr[k] = v[:n_train]
        m["counts"]["train"] = n_train
        m["derived_from"]["n_train"] = n_train
    if dt_scale is not None:
        for k in ("dt_direct", "dt_lf"):
            m[k] = round(float(src[k]) * dt_scale, 12)
            steps = m["t_end"] / m[k]
            if abs(steps - round(steps)) > 1e-9:
                raise ValueError(f"t_end {m['t_end']} が {k} = {m[k]} で割り切れない")
        m["derived_from"]["dt_scale"] = dt_scale
    for key, val in (("dt_direct", dt_direct), ("dt_lf", dt_lf)):
        if val is None:
            continue
        m[key] = float(val)
        steps = m["t_end"] / m[key]
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"t_end {m['t_end']} が {key} = {val} で割り切れない")
        m["derived_from"][key] = float(val)
    return m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="src", required=True)
    p.add_argument("--to", dest="dst", required=True)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--dt-scale", type=float, default=None)
    p.add_argument("--dt-direct", type=float, default=None)
    p.add_argument("--dt-lf", type=float, default=None)
    args = p.parse_args()
    src = json.loads((DATASETS_DIR / args.src / "manifest.json").read_text())
    m = derive(src, args.dst, args.n_train, args.dt_scale, args.dt_direct, args.dt_lf)
    out = DATASETS_DIR / args.dst
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {out / 'manifest.json'}  train={m['counts'].get('train')}  dt_direct={m['dt_direct']}  dt_lf={m['dt_lf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
