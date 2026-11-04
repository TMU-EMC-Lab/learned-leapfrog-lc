#!/usr/bin/env python3
"""評価の参照解を前もって計算して datasets/<id>/ref_cache/ に保存する（2026-10-21）.

  python scripts/build_ref_cache.py DATASET_ID SPLIT [--t-eval 50] [--family lf]

system_eval.cached_reference と同じ鍵（分割・刻み・ステップ数）で保存するので，評価のジョブはそれを読むだけになる。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import system_eval as se  # noqa: E402
from harmonic_ic_dataset import load_manifest, split_to_array  # noqa: E402
from harmonic_ref import steps_from_t_end  # noqa: E402
from systems import system_from_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_id")
    ap.add_argument("split")
    ap.add_argument("--t-eval", type=float, default=50.0)
    ap.add_argument("--family", default="lf", choices=["lf", "direct_state"])
    a = ap.parse_args()
    m = load_manifest(a.dataset_id)
    dt = float(m["dt_lf"] if a.family == "lf" else m["dt_direct"])
    steps = steps_from_t_end(a.t_eval, dt)
    t0 = time.time()
    ref = se.cached_reference(system_from_manifest(m), m, a.split, split_to_array(m, a.split), dt, steps)
    print(f"{a.dataset_id} {a.split} dt={dt:g} steps={steps} shape={ref.shape} {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
