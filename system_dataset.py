#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""調和振動子以外の系のデータセット（初期条件と尺度）を作る（2026-09-30 追加）.

  python system_dataset.py --system pendulum  --dataset-id pend_v1
  python system_dataset.py --system vanderpol --dataset-id vdp_v1
  python system_dataset.py --system duffing   --dataset-id duffing_v1
  python system_dataset.py --system lorenz    --dataset-id lorenz_v1
  python system_dataset.py --system doublewell --dataset-id dwell_v1_dtd2 --dt-direct 0.05   （2026-10-15 追加：morse / charged / fpu も同様）

既定の本数は調和振動子と同じ（学習 1024，検証 256，内挿 512，外挿 256）。領域は systems.DEFAULT_REGIONS。
出力は datasets/<dataset_id>/manifest.json（既存と同じ置き場）。各分割に初期条件 ics (N, d) と尺度 scales (N, d)。
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from harmonic_ic_dataset import DATASETS_DIR, save_manifest
from systems import DEFAULT_REGIONS, Lorenz, make_system


def build(args) -> dict:
    params = {}
    if args.system == "pendulum" and args.omega0 is not None:
        params["omega0"] = args.omega0
    if args.system in ("vanderpol", "vdp") and args.mu is not None:
        params["mu"] = args.mu
    if args.system == "lorenz":
        for k in ("sigma", "rho", "beta"):
            if getattr(args, k) is not None:
                params[k] = getattr(args, k)
    if args.system == "duffing":
        for k in ("alpha", "beta", "delta"):
            if getattr(args, f"duff_{k}") is not None:
                params[k] = getattr(args, f"duff_{k}")
    if args.system == "harmonic" and args.omega is not None:
        params["omega"] = args.omega
    if args.system == "fpu" and getattr(args, "fpu_n", None) is not None:
        params["n"] = args.fpu_n
    if args.system in ("nltl", "nltlg", "nltlv") and getattr(args, "nltl_n", None) is not None:
        params["n"] = args.nltl_n
    if args.system == "nltlg" and getattr(args, "nltl_g", None) is not None:
        params["G"] = args.nltl_g
    if args.system == "charged" and getattr(args, "bfield", None) is not None:
        params["B"] = args.bfield
    sysm = make_system(args.system, **params)
    if args.t_end is not None:
        sysm.t_end = args.t_end
    if args.dt_direct is not None:
        sysm.dt_direct = args.dt_direct
    if args.dt_lf is not None:
        sysm.dt_lf = args.dt_lf

    reg = DEFAULT_REGIONS[sysm.name]
    tr_lo, tr_hi = (args.train_lo, args.train_hi) if args.train_lo is not None else reg["train"]
    ex_lo, ex_hi = (args.extrap_lo, args.extrap_hi) if args.extrap_lo is not None else reg["extrap"]
    rng = np.random.default_rng(args.seed)
    counts = {"train": args.n_train, "val": args.n_val, "test_interp": args.n_test_interp, "test_extrap": args.n_test_extrap}
    used: set = set()
    splits, summary = {}, {}
    for split, n in counts.items():
        lo, hi = (ex_lo, ex_hi) if split == "test_extrap" else (tr_lo, tr_hi)
        kw = {"exclude": used} if isinstance(sysm, Lorenz) and split != "test_extrap" else {}
        ics = sysm.sample_ics(n, lo, hi, rng, **kw)
        sc = sysm.scales(ics, floor=args.scale_floor)
        rv = sysm.region_value(ics)
        entry = {"ics": ics.tolist(), "scales": sc.tolist()}
        if sysm.dim == 2:
            entry["x0"] = ics[:, 0].tolist()
            entry["v0"] = ics[:, 1].tolist()
        splits[split] = entry
        summary[split] = {"region_min": float(rv.min()), "region_max": float(rv.max()), "region_mean": float(rv.mean()),
                          "scale_x_mean": float(sc[:, 0].mean())}
    manifest = {
        "dataset_id": args.dataset_id,
        "system": sysm.name,
        "params": sysm.manifest_params(),
        "dim": sysm.dim,
        "var_names": list(sysm.var_names),
        "second_order": sysm.second_order,
        "omega": sysm.omega,
        "n_dof": sysm.n_dof,
        "t_end": sysm.t_end,
        "dt_direct": sysm.dt_direct,
        "dt_lf": sysm.dt_lf,
        "t_evals_default": list(sysm.t_evals),
        "seed": args.seed,
        "scale_floor": args.scale_floor,
        "sampling": {"train": [tr_lo, tr_hi], "extrap": [ex_lo, ex_hi],
                     "region": {"pendulum": "最大振れ角 θ_max [rad]", "vanderpol": "初期点の半径 √(x0² + v0²)",
                                "lorenz": "アトラクタまでの距離 / アトラクタの広がり", "harmonic": "振幅 A",
                                "duffing": "振幅 A（折り返し点の |x|）", "doublewell": "エネルギー E",
                                "morse": "エネルギー E", "charged": "エネルギー E", "fpu": "全エネルギー E",
                                "nlcap": "エネルギー E", "nllc": "エネルギー E", "nlind": "エネルギー E",
                                "nllcqf": "エネルギー E", "nlqf3": "エネルギー E", "nlcapqf": "エネルギー E", "nlindqf": "エネルギー E", "nlqf3vi": "エネルギー E", "nltl": "1 段あたりのエネルギー E/n", "nltlg": "初期の 1 段あたりのエネルギー E/n", "nltlv": "初期の電圧の最大 max|u(q)|（抽出は波の 1 段あたりの二乗平均平方根）"}[sysm.name]},
        "counts": counts,
        "region_summary": summary,
        "splits": splits,
    }
    if isinstance(sysm, Lorenz):
        manifest["attractor_std"] = sysm.attractor_std().tolist()
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True, choices=["pendulum", "vanderpol", "vdp", "duffing", "lorenz", "harmonic",
                                                            "doublewell", "morse", "charged", "fpu", "nlcap", "nllc", "nlind", "nllcqf", "nlqf3",
                                                            "nlcapqf", "nlindqf", "nlqf3vi", "nltl", "nltlg", "nltlv"])
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--n-train", type=int, default=1024)
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--n-test-interp", type=int, default=512)
    p.add_argument("--n-test-extrap", type=int, default=256)
    p.add_argument("--train-lo", type=float, default=None)
    p.add_argument("--train-hi", type=float, default=None)
    p.add_argument("--extrap-lo", type=float, default=None)
    p.add_argument("--extrap-hi", type=float, default=None)
    p.add_argument("--omega", type=float, default=None, help="harmonic の ω")
    p.add_argument("--omega0", type=float, default=None, help="pendulum の ω0 = √(g/L)（既定 1）")
    p.add_argument("--mu", type=float, default=None, help="vanderpol の μ（既定 1）")
    p.add_argument("--duff-alpha", type=float, default=None, help="duffing の α（既定 1）")
    p.add_argument("--duff-beta", type=float, default=None, help="duffing の β（既定 1）")
    p.add_argument("--duff-delta", type=float, default=None, help="duffing の減衰 δ（既定 0：保存系）")
    p.add_argument("--fpu-n", type=int, default=None, help="fpu の質点の数（既定 4）")
    p.add_argument("--nltl-n", type=int, default=None, help="nltl（非線形伝送線路）の段数（既定 8，8 の倍数）")
    p.add_argument("--nltl-g", type=float, default=None, help="nltlg（損失のある線路）の各段の漏れのコンダクタンス G（既定 0.03）")
    p.add_argument("--bfield", type=float, default=None, help="charged の磁場の強さ B（既定 1）")
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--t-end", type=float, default=None, help="学習に使う軌道の長さ [s]")
    p.add_argument("--dt-direct", type=float, default=None)
    p.add_argument("--dt-lf", type=float, default=None)
    p.add_argument("--scale-floor", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    manifest = build(args)
    path = save_manifest(manifest, DATASETS_DIR / args.dataset_id)
    print(f"saved {path}")
    for split, s in manifest["region_summary"].items():
        print(f"  {split:12s} n={manifest['counts'][split]:4d}  region {s['region_min']:.3f}–{s['region_max']:.3f}"
              f"  scale_x 平均 {s['scale_x_mean']:.3f}")
    print(json.dumps({k: manifest[k] for k in ("system", "params", "dim", "t_end", "dt_direct", "dt_lf", "t_evals_default")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
