#!/usr/bin/env python3
"""位置の網と速度の網の 1 ステップの誤差と，それがエネルギーを系統的に動かす量（2026-10-20，ELEX）.

  python3 scripts/energy_projection_20261020.py --dataset-id nlqf3_v1_dtd2 --size small --out OUT.json [--root RUNS_ROOT]

2 節の説明：位置の網が力の項 −τΔt·a を割合 ε_G だけ学び損ねると，位置の更新の速さに e = ε_G τ Δt a の誤差が乗る。
エネルギーの変化の速さは (∂H/∂x)·e = −ε_G τ Δt a² で符号が決まり（τ > 0 で減る，τ < 0 で増える），ロールアウトで積み重なる。
ここでは参照の軌道の状態（テスト内挿の初期条件 64 個 × 200 ステップ，教師強制）で，
  位置の網：e_G = (G(x_n, u_n) − G*(x_n, u_n))，u_n は型に合った参照の速度，G* は参照の (x_{n+1} − x_n)/Δt
  速度の網：e_F = (F(x_n, w_n) − F*(x_n, w_n))，w_n は型に合った参照の速度
を求め，誤差の二乗平均平方根（rms_G，rms_F）と，1 ステップあたりのエネルギーの変化 Δt·⟨∂H/∂x · e_G⟩ と Δt·⟨∂H/∂v · e_F⟩ を
初期条件ごとのエネルギーで割って平均した値（符号つき，1000 ステップ分）を出す。∂H は sysm.energy の中心差分。
あわせて，位置の網の誤差を力の項の向き f = τΔt·ẍ（ẍ は参照の軌道の位置の 2 階差分）に当てはめた係数 ε_G = ⟨e_G·f⟩/⟨f·f⟩ を出す（τ ≠ 0 の型だけ）。
網は学習率 1e-2・3e-2 × seed 0〜9 の 20 本ずつ（大きさは --size）。最後に，型ごとの中央値と，その LF に対する比を印字する。
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from lllc_paths import RUNS_ROOT  # noqa: E402

MODELS = {"SIE": ("SIE_MLP_state", 0.5), "SIE-M": ("SIEMid_MLP_state", 0.0), "LF-U": ("LF_Par_MLP", -1.0),
          "LF": ("LF_Free_MLP", 0.0)}


def main():
    import ic_generalization_eval as ev
    from harmonic_ic_dataset import split_to_array
    from harmonic_models import memoryless_width
    from systems import system_from_manifest

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--size", default="small")
    ap.add_argument("--n-ics", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lrs", nargs="+", default=["1e2", "3e2"])
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--tag", default="_gpu", help="run_id の印")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ds = args.dataset_id
    teacher = "ana" if ds.startswith("mix_") else "num"
    manifest = json.load(open(os.path.join(ROOT, "datasets", ds, "manifest.json")))
    dim = int(manifest.get("dim", 2))
    sysm = system_from_manifest(manifest)
    dt = float(manifest["dt_lf"])
    T = args.steps
    ics = split_to_array(manifest, "test_interp")[:args.n_ics]
    k = sysm.n_dof
    N = len(ics)
    ref = sysm.integrate(ics, dt, T + 2)  # (N, T+2, 2k)
    vh = sysm.half_step_velocity(ics, dt, T + 2)  # vh[:, n] = v_{n−1/2}
    if vh.ndim == 2:
        vh = vh[:, :, None]
    x, v = ref[:, :T + 1, :k], ref[:, :T + 1, k:]
    e_floor = -sysm.params["a"] ** 2 / (4 * sysm.params["b"]) if sysm.name == "doublewell" else 0.0
    H0 = sysm.energy(ics) - e_floor  # (N,)：ポテンシャルの最小からのエネルギー
    h = 1e-6
    s = ref[:, :T].reshape(-1, 2 * k)
    grad = np.empty_like(s)
    for i in range(2 * k):
        e_i = np.zeros(2 * k)
        e_i[i] = h
        grad[:, i] = (sysm.energy(s + e_i) - sysm.energy(s - e_i)) / (2 * h)
    dHx = grad[:, :k].reshape(N, T, k)
    dHv = grad[:, k:].reshape(N, T, k)
    Gstar = (x[:, 1:T + 1] - x[:, :T]) / dt
    xdd = (ref[:, 2:T + 2, :k] - 2 * ref[:, 1:T + 1, :k] + ref[:, :T, :k]) / dt ** 2
    inputs = {  # 位置の網と速度の網に入る参照の速度，速度の網の目標
        "SIE": (v[:, 1:T + 1], v[:, :T], (v[:, 1:T + 1] - v[:, :T]) / dt),
        "SIE-M": (0.5 * (v[:, :T] + v[:, 1:T + 1]), v[:, :T], (v[:, 1:T + 1] - v[:, :T]) / dt),
        "LF-U": (vh[:, :T], vh[:, :T], (vh[:, 1:T + 1] - vh[:, :T]) / dt),
        "LF": (vh[:, 1:T + 1], vh[:, :T], (vh[:, 1:T + 1] - vh[:, :T]) / dt),
    }

    out = {"dataset": ds, "size": args.size, "runs": []}
    for short, (name, tau) in MODELS.items():
        hs = memoryless_width(name, args.size, dim)
        u, w, Fstar = inputs[short]
        for lr in args.lrs:
            for sd in range(10):
                rid = f"abs_mse_lr{lr}_bs64_gc1_{teacher}_ltctau_cos_s{sd}{args.tag}"
                d = Path(args.root) / ds / rid / name / f"hs{hs}"
                p = ev.find_model_checkpoint(d, name, hs)
                if not p.exists():
                    continue
                model, _ = ev.load_model_for_eval(name, hs, d, p)
                model = copy.deepcopy(model).double().cpu().eval()
                with torch.no_grad():
                    xg = torch.as_tensor(np.concatenate([x[:, :T].reshape(-1, k), u.reshape(-1, k)], 1))
                    xf = torch.as_tensor(np.concatenate([x[:, :T].reshape(-1, k), w.reshape(-1, k)], 1))
                    eG = model.G(xg).numpy().reshape(N, T, k) - Gstar
                    eF = model.F(xf).numpy().reshape(N, T, k) - Fstar
                # 1000 ステップあたりのエネルギーの相対変化（初期条件ごとに割ってから平均）
                dG = float(np.mean(1000 * dt * np.mean(np.sum(dHx * eG, axis=2), axis=1) / H0))
                dF = float(np.mean(1000 * dt * np.mean(np.sum(dHv * eF, axis=2), axis=1) / H0))
                rG = float(np.sqrt(np.mean(eG ** 2)))
                if tau != 0:
                    f = tau * dt * 0.5 * (xdd + np.concatenate([xdd[:, :1], xdd[:, :-1]], 1))  # ẍ を t_n + Δt/2 の近くに
                    epsG = float(np.sum(eG * f) / np.sum(f * f))
                else:
                    epsG = float("nan")
                rF = float(np.sqrt(np.mean(eF ** 2)))
                out["runs"].append({"model": short, "tau": tau, "lr": lr, "seed": sd, "dH_G": dG, "dH_F": dF,
                                    "rms_G": rG, "rms_F": rF, "eps_G": epsG})
    med = {}
    for short in MODELS:
        rs = [r for r in out["runs"] if r["model"] == short]
        if rs:
            med[short] = {key: float(np.median([r[key] for r in rs])) for key in ("dH_G", "dH_F", "rms_G", "rms_F")}
            med[short]["n"] = len(rs)
            print(f"{short:5s} τ={MODELS[short][1]:+.1f} n={len(rs):2d}  1000 ステップのエネルギーの相対変化：位置の網 {med[short]['dH_G']:+.2e}"
                  f"  速度の網 {med[short]['dH_F']:+.2e}  | 誤差 RMS 位置の網 {med[short]['rms_G']:.2e} 速度の網 {med[short]['rms_F']:.2e}", flush=True)
    out["medians"] = med
    if "LF" in med:
        out["rms_ratio_to_LF"] = {s: {"G": med[s]["rms_G"] / med["LF"]["rms_G"], "F": med[s]["rms_F"] / med["LF"]["rms_F"]}
                                  for s in med if s != "LF"}
        for s, r in out["rms_ratio_to_LF"].items():
            print(f"{s:5s}/LF  誤差 RMS の中央値の比：位置の網 {r['G']:.2f}  速度の網 {r['F']:.2f}", flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
