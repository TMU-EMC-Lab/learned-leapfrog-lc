#!/usr/bin/env python3
"""反射のある終端のバラクタの線路（パルスが学習したブロックを何度も通る）.

開いた線路（nltlv_open_20261022.py）と同じ回路・同じ学習済みの網・同じ埋め込みで，終端だけを変える。
電源の内部抵抗 R_s = 0.1（5 Ω），負荷 R_L = 10（500 Ω）。反射係数は電源で −0.82，負荷で +0.82 なので，
パルスは減衰しながら線路を往復し，10 ns（約 394 t0）の間に学習したブロックを 9〜10 回通る（古典の方法の下見で，負荷の電圧のピークは 5 往復ぶん）。

  python3 scripts/nltlv_reflect_20261022.py --out OUT.json [--models LF SIE LF-U Res RK4 LF-KCL] [--lrs 1e2 3e2] [--seeds 0 9]
  python3 scripts/nltlv_reflect_20261022.py --models True-full --out FLOOR.json   （つなぎ方の床：真の特性を整数の時刻の型で進める）
  python3 scripts/nltlv_reflect_20261022.py --reference --out REF.json            （古典の方法の精度：刻み 0.005 の RK4 の解との差）

入力：電源の開放電圧 V_s(t) = a (1 + R_s) p(t)。a は線路への入射の振幅（V），p は nltlv_open と同じ台形（上昇余弦の辺 t_r，平らな部分 10 t0）。
試験のパルス 64 本：a ∈ [0.5, 1.0] V，t_r ∈ [4, 12] t0 の一様乱数（seed 20261023）。図用に a = 0.8，t_r = 8 を 1 本足す（統計には入れない）。
長さ 7874 ステップ（10.0 ns）。物差し：負荷の電圧 u(q_39) の，すべて古典の解との差の二乗平均平方根を a で割った値。
全体（0〜10 ns），前（最初の 2000 ステップ，0〜2.54 ns：開いた線路と同じ長さ）と後（2000 ステップ以降，2.54〜10 ns）の 3 つ。
パルスごとに求め，発散（有限でないか 100 超）を除いて平均し，発散の数を数える。
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import nltlv_open_20261022 as O  # noqa: E402
from harmonic_models import hidden_size_of, memoryless_width  # noqa: E402
from lllc_paths import RUNS_ROOT  # noqa: E402

N, EDGE, DT = O.N, O.EDGE, O.DT
RS, RL = 0.1, 10.0
STEPS, SPLIT = 7874, 2000
u_of, p_of, flux_update = O.u_of, O.p_of, O.flux_update


def pulses(seed=20261023, n=64):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 1.0, n)
    tr = rng.uniform(4.0, 12.0, n)
    return np.concatenate([a, [0.8]]), np.concatenate([tr, [8.0]])


def charge_update(q, f_new, dt, vs_mid, rs=RS, rl=RL):
    """古典の電荷の更新（全節点）。電源と負荷の項は ū で中点にそろえる（nltlv_open と同じ。抵抗だけを別にした）。"""
    i = f_new
    base = q + dt * (i - torch.roll(i, -1, dims=1))
    qn = base.clone()
    for _ in range(3):
        ub = 0.5 * (u_of(q) + u_of(qn))
        qn = base.clone()
        qn[:, 0] = base[:, 0] + dt * (vs_mid - ub[:, 0]) / rs
        qn[:, -1] = base[:, -1] - dt * ub[:, -1] / rl
    return qn


def run(model, kind, a, tr, steps=STEPS, dt=DT, record=False):
    """nltlv_open_20261022.run と同じ埋め込み（境目・つなぎ方・発散の扱い）で，終端の抵抗と物差しの時間窓だけを変える。"""
    dev = a.device
    B = len(a)
    q = torch.zeros(B, N, device=dev, dtype=torch.float32)
    f_half = torch.zeros_like(q)
    f_int = torch.zeros_like(q)
    qc = torch.zeros(B, N, device=dev, dtype=torch.float64)
    fc = torch.zeros_like(qc)
    m = torch.zeros(N, dtype=torch.bool, device=dev)
    m[EDGE:N - EDGE] = True
    h = torch.zeros(B, max(1, hidden_size_of(model)) if model is not None else 1, device=dev)
    h_pp, h_p = h.clone(), h.clone()
    div = torch.zeros(B, dtype=torch.bool, device=dev)
    sq_e = torch.zeros(B, dtype=torch.float64, device=dev)
    sq_l = torch.zeros(B, dtype=torch.float64, device=dev)
    eL = torch.zeros(B, dtype=torch.float64, device=dev)
    eLc = torch.zeros(B, dtype=torch.float64, device=dev)
    uL, uLc = [], []
    for n in range(steps):
        t_mid = (n + 0.5) * dt
        vs = a * (1.0 + RS) * p_of(torch.full_like(a, t_mid), tr)
        fc_new = flux_update(qc, fc, dt)
        qc_new = charge_update(qc, fc_new, dt, vs.double())
        f_c = flux_update(q, f_half, dt)
        if model is None:
            f_new = f_c
            q_new = charge_update(q, f_new, dt, vs.float())
            f_int_new = f_int
        elif kind == "stag":
            out_v, h_new, h_p_out = model(torch.cat([q, f_half], 1), h_pp, h_p, is_velocity_step=True)
            h_pp, h_p = h_p_out, h_new
            f_new = torch.where(m, out_v, f_c)
            f_new[:, 0] = 0.0
            q_c = charge_update(q, f_new, dt, vs.float())
            out_x, h_new, h_p_out = model(torch.cat([q, f_new], 1), h_pp, h_p, is_velocity_step=False)
            h_pp, h_p = h_p_out, h_new
            q_new = torch.where(m, out_x, q_c)
            f_int_new = f_int
        else:
            phi_n = torch.where(m, f_int, 0.5 * (f_half + f_c))
            z_new, _ = model(torch.cat([q, phi_n], 1), h)
            q_l, f_l = z_new[:, :N], z_new[:, N:]
            f_new = torch.where(m, 0.5 * (f_int + f_l), f_c)
            f_new[:, 0] = 0.0
            q_c = charge_update(q, f_new, dt, vs.float())
            q_new = torch.where(m, q_l, q_c)
            f_int_new = torch.where(m, f_l, f_int)
        bad = ~torch.isfinite(q_new).all(1) | (q_new.abs().amax(1) > 100)
        div |= bad
        q_new = torch.where(div[:, None], torch.zeros_like(q_new), q_new)
        f_new = torch.where(div[:, None], torch.zeros_like(f_new), f_new)
        f_int_new = torch.where(div[:, None], torch.zeros_like(f_int_new), f_int_new)
        q, f_half, f_int, qc, fc = q_new, f_new, f_int_new, qc_new, fc_new
        ul, ulc = u_of(q[:, -1]).double(), u_of(qc[:, -1])
        if n < SPLIT:
            sq_e += (ul - ulc) ** 2
        else:
            sq_l += (ul - ulc) ** 2
        eL += ul ** 2 * dt / RL
        eLc += ulc ** 2 * dt / RL
        if record:
            uL.append(ul.cpu())
            uLc.append(ulc.cpu())
    ad = a.double()
    out = {"err": (torch.sqrt((sq_e + sq_l) / steps) / ad).cpu().numpy(),
           "err_early": (torch.sqrt(sq_e / SPLIT) / ad).cpu().numpy(),
           "err_late": (torch.sqrt(sq_l / (steps - SPLIT)) / ad).cpu().numpy(),
           "div": div.cpu().numpy(), "eratio": (eL / eLc).cpu().numpy()}
    if record:
        out["uL"] = torch.stack(uL, 1).numpy()
        out["uLc"] = torch.stack(uLc, 1).numpy()
    return out


def reference(a, tr):
    """古典の方法の精度：同じ回路の ODE を刻み 0.005 の RK4 で解き，Δt = 0.05 の古典の解と負荷の電圧を比べる（倍精度，CPU）。"""
    a, tr = a.cpu().double(), tr.cpu().double()
    B = len(a)

    def rhs(t, q, f):
        vs = a * (1.0 + RS) * p_of(torch.full_like(a, t), tr)
        uq = u_of(q)
        dq = f - torch.roll(f, -1, dims=1)
        dq[:, 0] += (vs - uq[:, 0]) / RS
        dq[:, -1] -= uq[:, -1] / RL
        df = torch.roll(uq, 1, dims=1) - uq
        df[:, 0] = 0.0
        return dq, df

    hh, sub = 0.005, 10
    q = torch.zeros(B, N, dtype=torch.float64)
    f = torch.zeros_like(q)
    uL_ref = []
    t = 0.0
    for n in range(STEPS):
        for _ in range(sub):
            k1 = rhs(t, q, f)
            k2 = rhs(t + hh / 2, q + hh / 2 * k1[0], f + hh / 2 * k1[1])
            k3 = rhs(t + hh / 2, q + hh / 2 * k2[0], f + hh / 2 * k2[1])
            k4 = rhs(t + hh, q + hh * k3[0], f + hh * k3[1])
            q = q + hh / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            f = f + hh / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            t += hh
        uL_ref.append(u_of(q[:, -1]))
    uL_ref = torch.stack(uL_ref, 1)
    cl = run(None, "stag", a, tr, record=True)
    err = torch.sqrt(((torch.as_tensor(cl["uLc"]) - uL_ref) ** 2).mean(1)) / a
    return {"classical_vs_rk4": err.numpy(), "uL_ref": uL_ref.numpy(), "uL_classical": cl["uLc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--models", nargs="+", default=["LF", "SIE", "LF-U", "Res", "RK4", "LF-KCL"])
    ap.add_argument("--lrs", nargs="+", default=["1e2", "3e2"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 9])
    ap.add_argument("--size", default="medium")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS, help="試しの実行用（主の解析は既定の 7874）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_np, tr_np = pulses()
    a = torch.as_tensor(a_np, device=dev, dtype=torch.float32)
    tr = torch.as_tensor(tr_np, device=dev, dtype=torch.float32)
    res = {"N": N, "edge": EDGE, "dt": DT, "steps": args.steps, "split": SPLIT, "rs": RS, "rl": RL,
           "a": a_np.tolist(), "tr": tr_np.tolist(), "runs": []}
    if args.reference:
        with torch.no_grad():
            r = reference(a, tr)
        e = r["classical_vs_rk4"][:-1]
        res["classical_vs_rk4"] = {"mean": float(e.mean()), "max": float(e.max())}
        res["show_ref"] = {"uL_ref": r["uL_ref"][-1].tolist(), "uL_classical": r["uL_classical"][-1].tolist()}
        print("classical vs RK4:", res["classical_vs_rk4"], flush=True)
        json.dump(res, open(args.out, "w"))
        return
    for short in args.models:
        if short == "True-full":
            name, kind, hs = None, "full", 0
        else:
            name, kind = (O.STAG[short], "stag") if short in O.STAG else (O.FULL[short], "full")
            hs = memoryless_width(name, args.size, 16)
        for lr in (args.lrs if name else ["-"]):
            for sd in (range(args.seeds[0], args.seeds[1] + 1) if name else [0]):
                if name is None:
                    model = O.TrueFull().to(dev).eval()
                else:
                    rid = f"abs_mse_lr{lr}_bs64_gc1_num_ltctau_cos_s{sd}_gpu"
                    d = Path(args.root) / O.TRAIN_DS / rid / name / f"hs{hs}"
                    p = O.ev.find_model_checkpoint(d, name, hs)
                    if not p.exists():
                        print("missing", short, lr, sd, flush=True)
                        continue
                    model, _ = O.ev.load_model_for_eval(name, hs, d, p)
                    model.set_length(N)
                    model = model.to(dev).eval()
                with torch.no_grad():
                    r = run(model, kind, a, tr, steps=args.steps, record=True)
                dv = r["div"][:-1]
                ok = np.isfinite(r["err"][:-1]) & (r["err"][:-1] <= 100) & ~dv
                rec = {"model": short, "lr": lr, "seed": sd, "n_div": int((~ok).sum())}
                for key in ("err", "err_early", "err_late"):
                    rec[key] = float(r[key][:-1][ok].mean()) if ok.any() else 1e3
                rec["eratio"] = float(np.median(r["eratio"][:-1][ok])) if ok.any() else float("nan")
                if (lr == "1e2" and sd == 0) or name is None:
                    rec["show"] = {"uL": r["uL"][-1].tolist(), "uLc": r["uLc"][-1].tolist(), "div": bool(r["div"][-1])}
                res["runs"].append(rec)
                print(short, lr, sd, {k: v for k, v in rec.items() if k != "show"}, flush=True)
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
