#!/usr/bin/env python3
"""参照解の半ステップの磁束を使わずに LF を始めたときの比（2026-10-28，ELEX）.

  python3 scripts/learned_start_20261028.py --out OUT.json [--root RUNS_ROOT]

評価は BHIC_LF_LEARNED_START=1 の ic_generalization_eval.py（v_{-1/2} = v_0 - (dt/2) F(x_0, v_0)，結果は各 LF の run の eval_learnedstart/）。
回路（nlqf3_v1_dtd2，LF_Free_MLP）と 8 段の線路（nltl8_v1_dtd2，LF_Free_Loc），seed 0〜9，2 つの大きさ × 2 つの学習率の 40 本。
比は同じ seed・大きさ・学習率の対の幾何平均（両方の中央値が 0.3 以上の層は除く。scripts/letter_stats.py と同じ組み方）。
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lllc_paths import RUNS_ROOT  # noqa: E402

RID = re.compile(r"abs_mse_lr(1e2|3e2)_bs64_gc1_num_ltctau_cos_s(\d+)_gpu$")
CASES = {"circuit": ("nlqf3_v1_dtd2", "LF_Free_MLP", "Res_MLP_state", "RK4_MLP_state"),
         "line": ("nltl8_v1_dtd2", "LF_Free_Loc", "Res_Loc_state", "RK4_Loc_state")}


def err(f):
    d = json.load(open(f))
    rows = d["rows"] if "rows" in d else d
    v = next(x for x in rows if x.get("t_eval_tag") == "t50s")["test_interp_x_rmse_rel_mean"]
    return v if v is not None and math.isfinite(v) and v < 1e3 else 1e3


def runs(root, ds, m, sub=""):
    out = {}
    for rd in glob.glob(f"{root}/{ds}/abs_mse_*/{m}/hs*"):
        mm = RID.match(Path(rd).parent.parent.name)
        f = Path(rd) / sub / "ic_generalization_summary.json"
        if mm and int(mm[2]) <= 9 and f.exists():
            out[(mm[2], mm[1], int(Path(rd).name[2:]))] = err(f)
    hs = sorted({k[2] for k in out})
    return {(s, lr, "small" if h == hs[0] else "medium"): v for (s, lr, h), v in out.items()}


def geo(a, b):
    logs = []
    for lr in ("1e2", "3e2"):
        for sz in ("small", "medium"):
            ks = [k for k in b if k[1] == lr and k[2] == sz and k in a]
            if ks and not (np.median([a[k] for k in ks]) >= 0.3 and np.median([b[k] for k in ks]) >= 0.3):
                logs += [math.log(a[k] / b[k]) for k in ks]
    return {"ratio": float(np.exp(np.mean(logs))), "pairs": len(logs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for case, (ds, lf_m, res_m, rk_m) in CASES.items():
        lf, ls = runs(a.root, ds, lf_m), runs(a.root, ds, lf_m, "eval_learnedstart")
        R, K = runs(a.root, ds, res_m), runs(a.root, ds, rk_m)
        res[case] = {"dataset": ds, "n": {"LF": len(lf), "LF_learned": len(ls), "Res": len(R), "RK4": len(K)},
                     "LF_learned/LF_ref": geo(ls, lf), "Res/LF_ref": geo(R, lf), "Res/LF_learned": geo(R, ls),
                     "RK4/LF_ref": geo(K, lf), "RK4/LF_learned": geo(K, ls),
                     "median_LF_ref": float(np.median(list(lf.values()))), "median_LF_learned": float(np.median(list(ls.values())))}
        print(case, json.dumps(res[case]))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
