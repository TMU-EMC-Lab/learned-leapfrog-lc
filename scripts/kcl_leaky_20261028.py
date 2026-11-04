#!/usr/bin/env python3
"""漏れのある 8 段の輪での LF-KCL（分路の項なし，LF_KCL_Loc）と LF の比（2026-10-28，ELEX）.

  python3 scripts/kcl_leaky_20261028.py --out OUT.json [--root RUNS_ROOT]

学習はほかの _gpu の run と同じ設定（seed 0〜9，2 つの大きさ × 2 つの学習率の 40 本）。誤差は t = 50 の内挿の位置の相対 RMSE。
比は同じ seed・大きさ・学習率の対の幾何平均（両方の中央値が 0.3 以上の層は除く）。分路の電流を学ぶ LF-KCLD も並べる。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learned_start_20261028 import geo, runs  # noqa: E402
from lllc_paths import RUNS_ROOT  # noqa: E402

DS = "nltlg8_v1_dtd2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lf, kcl, kcld = (runs(a.root, DS, m) for m in ("LF_Free_Loc", "LF_KCL_Loc", "LF_KCLD_Loc"))
    res = {"n": {"LF": len(lf), "LF-KCL": len(kcl), "LF-KCLD": len(kcld)},
           "LF-KCL/LF": geo(kcl, lf), "LF-KCLD/LF": geo(kcld, lf),
           "median": {"LF": float(np.median(list(lf.values()))), "LF-KCL": float(np.median(list(kcl.values()))),
                      "LF-KCLD": float(np.median(list(kcld.values())))},
           "pairs_KCL_worse": int(sum(kcl[k] > lf[k] for k in kcl if k in lf)),
           "n_failed_KCL": int(sum(v >= 0.3 for v in kcl.values()))}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
