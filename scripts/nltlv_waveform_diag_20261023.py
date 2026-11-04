#!/usr/bin/env python3
"""駆動したバラクタの線路で，ブロックのエネルギーでは入力の時刻の利点が出るのに，負荷の電圧の波形では小さいのはなぜかを，
負荷の電圧の誤差を成分に分けて調べる（2026-10-23）。計画した検定 W1〜W6（docs/planned_tests.csv，本文には載せていない）は，
2 組目の網（seed 10〜19）の open・train・reflect の出力の e_act・e_qui・e_W を使う。

nltlv_diag_20261022.py の run_line と同じ埋め込み（境目・つなぎ方・発散の扱い）で，負荷の電圧とブロックのエネルギーの
時系列を，網と古典の線路の両方で記録し，信号ごとに次を求める（A は信号の振幅）。

  e_tot     ：負荷の電圧の誤差の二乗平均平方根 / A（従来の物差しと同じ）
  e_align   ：網の負荷の電圧を ±3 ステップの範囲で 0.05 ステップずつずらし（線形補間），最も小さくした誤差 / A。
              時刻のずれ（到着の遅れ・進み）を除いた誤差。shift はそのときのずらし量（ステップ）
  e_act, e_qui：古典の負荷の電圧が 0.05 A を超える時刻（パルスが負荷にある）とそれ以外での誤差の寄与（二乗の和が e_tot²）
  e_slow, e_fast：誤差を 800 ステップ（40 t0）の移動平均で分けた，ゆっくりした成分（直流のずれ）と速い成分
  e_W       ：負荷に届いたエネルギー（Σ u_load²）の相対誤差
  e_E       ：ブロックのエネルギーの診断量（表 IV のブロックのエネルギーと同じ：ずれの二乗平均平方根 / 古典の最大）
  e_Ebias   ：ブロックのエネルギーのずれの時間平均（符号つき）/ 古典の最大（網がエネルギーを足すか減らすか）
  e_Eact, e_Equi：ブロックのエネルギーのずれを，古典のブロックのエネルギーが最大の 5% を超える時刻とそれ以外に分けた寄与
  dc_q      ：ブロックの電荷の平均（段の平均）のずれの時間平均の大きさ（偽の直流の電荷）

  python3 scripts/nltlv_waveform_diag_20261023.py --mode open --models LF --seeds 10 19 --out $O/waveforms/open_new_LF.json
  （train には --sig-seed 20261031 を付ける。コマンドの一覧は README.md の表にある）
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import nltlv_diag_20261022 as D  # noqa: E402
from harmonic_models import hidden_size_of  # noqa: E402
from lllc_paths import RUNS_ROOT  # noqa: E402

O, RF = D.O, D.RF
N, EDGE, DT = D.N, D.EDGE, D.DT
u_of, p_of, flux_update, energy = D.u_of, D.p_of, D.flux_update, D.energy


def run_trace(model, kind, src, rs, rl, steps, dev):
    """D.run_line と同じ更新で，負荷の電圧・ブロックのエネルギー・ブロックの電荷の平均の時系列を返す（補正なし）。"""
    B = src(torch.zeros(1, device=dev)).shape[0]
    q = torch.zeros(B, N, device=dev, dtype=torch.float32)
    f_half = torch.zeros_like(q)
    f_int = torch.zeros_like(q)
    qc = torch.zeros(B, N, device=dev, dtype=torch.float64)
    fc = torch.zeros_like(qc)
    m = torch.zeros(N, dtype=torch.bool, device=dev)
    m[EDGE:N - EDGE] = True
    blk = slice(EDGE, N - EDGE)
    h = torch.zeros(B, max(1, hidden_size_of(model)) if model is not None else 1, device=dev)
    h_pp, h_p = h.clone(), h.clone()
    div = torch.zeros(B, dtype=torch.bool, device=dev)
    tr = {k: torch.zeros(B, steps, dtype=torch.float64, device=dev) for k in ("uL", "uLc", "Eb", "Ebc", "qm", "qmc")}
    for n in range(steps):
        t_mid = torch.full((B,), (n + 0.5) * DT, device=dev)
        vs = src(t_mid)
        fc_new = flux_update(qc, fc, DT)
        qc_new = RF.charge_update(qc, fc_new, DT, vs.double(), rs=rs, rl=rl)
        f_c = flux_update(q, f_half, DT)
        if kind == "stag":
            out_v, h_new, h_p_out = model(torch.cat([q, f_half], 1), h_pp, h_p, is_velocity_step=True)
            h_pp, h_p = h_p_out, h_new
            f_new = torch.where(m, out_v, f_c)
            f_new[:, 0] = 0.0
            q_c = RF.charge_update(q, f_new, DT, vs.float(), rs=rs, rl=rl)
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
            q_c = RF.charge_update(q, f_new, DT, vs.float(), rs=rs, rl=rl)
            q_new = torch.where(m, q_l, q_c)
            f_int_new = torch.where(m, f_l, f_int)
        bad = ~torch.isfinite(q_new).all(1) | (q_new.abs().amax(1) > 100)
        div |= bad
        q_new = torch.where(div[:, None], torch.zeros_like(q_new), q_new)
        f_new = torch.where(div[:, None], torch.zeros_like(f_new), f_new)
        f_int_new = torch.where(div[:, None], torch.zeros_like(f_int_new), f_int_new)
        q, f_half, f_int, qc, fc = q_new, f_new, f_int_new, qc_new, fc_new
        qd, fd = q.double(), f_half.double()
        tr["uL"][:, n] = u_of(qd[:, -1])
        tr["uLc"][:, n] = u_of(qc[:, -1])
        tr["Eb"][:, n] = energy(qd, fd)[:, blk].sum(1)
        tr["Ebc"][:, n] = energy(qc, fc)[:, blk].sum(1)
        tr["qm"][:, n] = qd[:, blk].mean(1)
        tr["qmc"][:, n] = qc[:, blk].mean(1)
    out = {k: v.cpu().numpy() for k, v in tr.items()}
    out["div"] = div.cpu().numpy()
    return out


def movavg(x, w):
    k = np.ones(w) / w
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def metrics(tr, amp):
    B, T = tr["uL"].shape
    t = np.arange(T, dtype=np.float64)
    shifts = np.arange(-3.0, 3.0 + 1e-9, 0.05)
    lo, hi = 4, T - 4  # ずらしたときに端の外を使わない範囲
    res = {k: np.full(B, np.nan) for k in ("e_tot", "e_align", "shift", "e_act", "e_qui", "e_slow", "e_fast", "e_W",
                                           "e_E", "e_Ebias", "e_Eact", "e_Equi", "dc_q")}
    for b in range(B):
        if tr["div"][b]:
            continue
        A = float(amp[b])
        uL, uC = tr["uL"][b], tr["uLc"][b]
        d = uL - uC
        res["e_tot"][b] = math.sqrt(np.mean(d * d)) / A
        best, bs = np.inf, 0.0
        uc = uC[lo:hi]
        for s in shifts:
            ui = np.interp(t[lo:hi] + s, t, uL)
            e = np.mean((ui - uc) ** 2)
            if e < best:
                best, bs = e, s
        res["e_align"][b] = math.sqrt(best) / A
        res["shift"][b] = bs
        act = np.abs(uC) > 0.05 * A
        res["e_act"][b] = math.sqrt(np.sum(d[act] ** 2) / T) / A
        res["e_qui"][b] = math.sqrt(np.sum(d[~act] ** 2) / T) / A
        sl = movavg(d, 800)
        res["e_slow"][b] = math.sqrt(np.mean(sl * sl)) / A
        res["e_fast"][b] = math.sqrt(np.mean((d - sl) ** 2)) / A
        Wm, Wc = np.sum(uL * uL), np.sum(uC * uC)
        res["e_W"][b] = abs(Wm - Wc) / Wc
        Eb, Ebc = tr["Eb"][b], tr["Ebc"][b]
        sc = max(np.abs(Ebc).max(), 1e-12)
        dE = Eb - Ebc
        res["e_E"][b] = math.sqrt(np.mean(dE * dE)) / sc
        res["e_Ebias"][b] = np.mean(dE) / sc
        actE = np.abs(Ebc) > 0.05 * sc
        res["e_Eact"][b] = math.sqrt(np.sum(dE[actE] ** 2) / T) / sc
        res["e_Equi"][b] = math.sqrt(np.sum(dE[~actE] ** 2) / T) / sc
        res["dc_q"][b] = abs(np.mean(tr["qm"][b] - tr["qmc"][b]))
    return res


def sources(mode, dev, sig_seed=None):
    """nltlv_diag_20261022.py の open・reflect・cw・prbs・train と同じ信号。返り値：src，rs，rl，steps，amp。"""
    if mode == "open":
        A_np, tr_np = O.pulses()
        A_np, tr_np = A_np[:-1], tr_np[:-1]
        A = torch.as_tensor(A_np, device=dev, dtype=torch.float32)
        trt = torch.as_tensor(tr_np, device=dev, dtype=torch.float32)
        return (lambda t: 2.0 * A * p_of(t, trt)), 1.0, 1.0, 2000, A_np
    if mode == "reflect":
        a_np, tr_np = RF.pulses()
        a_np, tr_np = a_np[:-1], tr_np[:-1]
        a = torch.as_tensor(a_np, device=dev, dtype=torch.float32)
        trt = torch.as_tensor(tr_np, device=dev, dtype=torch.float32)
        return (lambda t: a * (1.0 + RF.RS) * p_of(t, trt)), RF.RS, RF.RL, RF.STEPS, a_np
    n = 64
    rng = np.random.default_rng(sig_seed if sig_seed is not None else {"cw": 20261026, "prbs": 20261027, "train": 20261028}[mode])
    if mode == "cw":
        A_np = rng.uniform(0.5, 1.5, n)
        f_np = rng.uniform(0.03, 0.12, n)
        th_np = rng.uniform(0, 2 * math.pi, n)
        A_t, f_t, th_t = (torch.as_tensor(x, device=dev, dtype=torch.float32) for x in (A_np, f_np, th_np))

        def src(t):
            ramp = 0.5 * (1 - torch.cos(math.pi * torch.clamp(t / 10.0, 0, 1)))
            return 2.0 * ramp * A_t * torch.sin(2 * math.pi * f_t * t + th_t)
        return src, 1.0, 1.0, 4000, A_np
    if mode == "train":
        A_np = rng.uniform(1.0, 3.0, n)
        tr_np = rng.uniform(4.0, 12.0, n)
        per_np = rng.uniform(60.0, 120.0, n)
        A_t, tr_t, per_t = (torch.as_tensor(x, device=dev, dtype=torch.float32) for x in (A_np, tr_np, per_np))
        return (lambda t: 2.0 * A_t * p_of(torch.remainder(t, per_t), tr_t)), 1.0, 1.0, 4000, A_np
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["open", "reflect", "cw", "train"], required=True)
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--models", nargs="+", default=["LF", "SIE", "LF-U"])
    ap.add_argument("--lrs", nargs="+", default=["1e2", "3e2"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 9])
    ap.add_argument("--size", default="medium")
    ap.add_argument("--sig-seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, rs, rl, steps, amp = sources(args.mode, dev, args.sig_seed)
    res = {"mode": args.mode, "runs": [], "steps": steps, "amp": np.asarray(amp).tolist()}
    for short in args.models:
        for lr in args.lrs:
            for sd in range(args.seeds[0], args.seeds[1] + 1):
                model, kind = D.load(short, lr, sd, args.root, args.size, dev, N)
                if model is None:
                    print("missing", short, lr, sd, flush=True)
                    continue
                with torch.no_grad():
                    tr = run_trace(model, kind, src, rs, rl, steps, dev)
                mt = metrics(tr, amp)
                res["runs"].append({"model": short, "lr": lr, "seed": sd, "div": tr["div"].tolist(),
                                    **{k: v.tolist() for k, v in mt.items()}})
                ok = ~tr["div"]
                print(short, lr, sd, " ".join(f"{k} {np.nanmean(mt[k][ok]):.4g}" for k in ("e_tot", "e_align", "e_act", "e_qui", "e_E")), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
