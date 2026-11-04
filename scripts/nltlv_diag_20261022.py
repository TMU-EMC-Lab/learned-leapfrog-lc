#!/usr/bin/env python3
"""駆動したバラクタの線路で入力の時刻の利点が出ない理由の切り分け.

開いた線路と反射のある線路の試験（nltlv_open_20261022.py，nltlv_reflect_20261022.py）の学習済みの網（nltlv8_v1_dtd2，約 2400 パラメータ，20 本ずつ）をそのまま使い，学習し直さない。
次の 2 つの候補を調べる。

  候補 1（ずれが働く場所と時間）：2 節の式では，学習し残した補正によるエネルギーのずれの速さは，線路では
    隣の段との電圧の差の 2 乗 (u(q_{k-1}) - u(q_k))^2 = dφ_k/dt^2 と u'(q_k) (i(φ_k) - i(φ_{k+1}))^2 = u' dq_k/dt^2 に比例する。
    環では全段がずっと揺れているが，パルスで駆動した線路ではほとんどの段が止まっているのではないか。
    → (a) 古典の解で，この 2 つの量の時間・段の平均（以下「露出」）を環と駆動した線路で比べる。
      (b) 環の試験の初期条件を波の振幅で分け，振幅の小さい初期条件で SIE/LF・LF-U/LF が 1 に近づくかを見る。
      (c) 線路を連続した信号（直流の偏り＋3 つの正弦波，整合した終端）で駆動し，ブロックの全段が揺れ続けるようにして，
          比が環の値（1.49・1.59）に近づくかを見る。
  候補 3（物差しの違い）：駆動の物差しは負荷の電圧という 1 点の波形で，ブロックの中のずれに鈍いのではないか。
    → 同じ実行で，ブロックの中（32 段）と全 40 段の電荷の二乗平均平方根の誤差（すべて古典の解との差），
      ブロックの中の磁束の誤差とエネルギーの差も測り，比を負荷の電圧の比と並べる。

  python3 scripts/nltlv_diag_20261022.py --mode ring   --out RING.json   [--models ...]
  python3 scripts/nltlv_diag_20261022.py --mode open   --out OPEN.json   （nltlv_open_20261022.py と同じ 64 本のパルス，整合，2000 ステップ）
  python3 scripts/nltlv_diag_20261022.py --mode reflect --out REFL.json  （nltlv_reflect_20261022.py と同じ 64 本，反射，7874 ステップ）
  python3 scripts/nltlv_diag_20261022.py --mode busy   --out BUSY.json   （連続した信号，整合，4000 ステップ）
  （--models classical だけを指定すると，学習なしの古典の線路で露出だけを測る）
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

import ic_generalization_eval as ev  # noqa: E402
import nltlv_open_20261022 as O  # noqa: E402
import nltlv_reflect_20261022 as RF  # noqa: E402
from harmonic_models import hidden_size_of, memoryless_width  # noqa: E402
from lllc_paths import RUNS_ROOT  # noqa: E402

N, EDGE, ALPHA, DT = O.N, O.EDGE, O.ALPHA, O.DT
NR = 8  # 環の段数
u_of, p_of, flux_update = O.u_of, O.p_of, O.flux_update


def du_of(q):
    return 1.0 + 2.0 * ALPHA * q


def energy(q, f):
    return 0.5 * q * q + ALPHA * q ** 3 / 3.0 + 0.5 * f * f


REST_CORRECT = False  # --rest-correct：各段の出力から，その段の値を全段に並べた一様な状態での増分を引く


def rest_inc_stag(model, a, b, velocity):
    """ずらした型：入力 (a, b)（(B, n) ずつ）の各段の値を全段に並べた一様な線路での，その更新の増分（真の更新は 0）。(B, n) を返す。"""
    B, n = a.shape
    aU = a.reshape(B * n, 1).expand(B * n, n)
    bU = b.reshape(B * n, 1).expand(B * n, n)
    hU = torch.zeros(B * n, max(1, hidden_size_of(model)), device=a.device)
    out, _, _ = model(torch.cat([aU, bU], 1), hU, hU, is_velocity_step=velocity)
    mid = n // 2
    base = bU[:, mid] if velocity else aU[:, mid]
    return (out[:, mid] - base).reshape(B, n)


def rest_inc_full(model, q, phi):
    """整数の時刻の型：各段の (q, φ) を全段に並べた一様な線路での 1 ステップの増分。(B, 2n) を返す。"""
    B, n = q.shape
    qU = q.reshape(B * n, 1).expand(B * n, n)
    fU = phi.reshape(B * n, 1).expand(B * n, n)
    hU = torch.zeros(B * n, max(1, hidden_size_of(model)), device=q.device)
    zU, _ = model(torch.cat([qU, fU], 1), hU)
    mid = n // 2
    dq = (zU[:, mid] - qU[:, mid]).reshape(B, n)
    df = (zU[:, n + mid] - fU[:, mid]).reshape(B, n)
    return torch.cat([dq, df], 1)


def busy_drives(seed=20261024, n=64, f_lo=0.1, f_hi=0.25):
    """連続した信号：V_s = 2 ramp(t) [V_dc + a Σ_j c_j sin(2π f_j t + θ_j)]（整合した終端なので入射の電圧は括弧の中）.
    V_dc ∈ [0, 1.5] V，a ∈ [0.2, 0.8] V（3 つの正弦波の和の二乗平均平方根），f_j ∈ [0.1, 0.25] / t0（環の波長 4〜8 段に当たる），
    立ち上がり 10 t0 の上昇余弦。環の学習の範囲（直流の偏り＋1 段あたりの二乗平均平方根 0.05〜1.0 の波，電圧は最大 6.5 V）に収まる。"""
    rng = np.random.default_rng(seed)
    vdc = rng.uniform(0.0, 1.5, n)
    a = rng.uniform(0.2, 0.8, n)
    f = rng.uniform(f_lo, f_hi, (n, 3))
    th = rng.uniform(0, 2 * math.pi, (n, 3))
    c = rng.uniform(0.5, 1.0, (n, 3))
    c = c / np.sqrt(0.5 * (c ** 2).sum(1, keepdims=True))  # 和の二乗平均平方根を 1 に（周波数が違えば）
    return vdc, a, f, th, c


def run_line(model, kind, src, rs, rl, steps, dev):
    """nltlv_reflect_20261022.run と同じ埋め込み（境目・つなぎ方・発散の扱い）。src(t) は (B,) の開放電圧。
    返り値：パルスごとの各物差しの二乗和（あとで平均と比にする）と，古典の解の露出。"""
    B = src(torch.zeros(1, device=dev)).shape[0] if callable(src) else None
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
    acc = {k: torch.zeros(B, dtype=torch.float64, device=dev) for k in
           ("load", "q_blk", "q_all", "f_blk", "e_blk", "expo_f", "expo_q", "act")}
    emax = torch.zeros(B, dtype=torch.float64, device=dev)
    for n in range(steps):
        t_mid = torch.full((B,), (n + 0.5) * dt_g, device=dev)
        vs = src(t_mid)
        fc_new = flux_update(qc, fc, dt_g)
        qc_new = RF.charge_update(qc, fc_new, dt_g, vs.double(), rs=rs, rl=rl)
        if model is None:
            q_new, f_new, f_int_new = qc_new.float(), fc_new.float(), f_int
        else:
            f_c = flux_update(q, f_half, dt_g)
            if kind == "stag":
                out_v, h_new, h_p_out = model(torch.cat([q, f_half], 1), h_pp, h_p, is_velocity_step=True)
                h_pp, h_p = h_p_out, h_new
                if REST_CORRECT:
                    out_v = out_v - rest_inc_stag(model, q, f_half, True)
                f_new = torch.where(m, out_v, f_c)
                f_new[:, 0] = 0.0
                q_c = RF.charge_update(q, f_new, dt_g, vs.float(), rs=rs, rl=rl)
                out_x, h_new, h_p_out = model(torch.cat([q, f_new], 1), h_pp, h_p, is_velocity_step=False)
                h_pp, h_p = h_p_out, h_new
                if REST_CORRECT:
                    out_x = out_x - rest_inc_stag(model, q, f_new, False)
                q_new = torch.where(m, out_x, q_c)
                f_int_new = f_int
            else:
                phi_n = torch.where(m, f_int, 0.5 * (f_half + f_c))
                z_new, _ = model(torch.cat([q, phi_n], 1), h)
                if REST_CORRECT:
                    z_new = z_new - rest_inc_full(model, q, phi_n)
                q_l, f_l = z_new[:, :N], z_new[:, N:]
                f_new = torch.where(m, 0.5 * (f_int + f_l), f_c)
                f_new[:, 0] = 0.0
                q_c = RF.charge_update(q, f_new, dt_g, vs.float(), rs=rs, rl=rl)
                q_new = torch.where(m, q_l, q_c)
                f_int_new = torch.where(m, f_l, f_int)
        bad = ~torch.isfinite(q_new).all(1) | (q_new.abs().amax(1) > 100)
        div |= bad
        q_new = torch.where(div[:, None], torch.zeros_like(q_new), q_new)
        f_new = torch.where(div[:, None], torch.zeros_like(f_new), f_new)
        f_int_new = torch.where(div[:, None], torch.zeros_like(f_int_new), f_int_new)
        q, f_half, f_int, qc, fc = q_new, f_new, f_int_new, qc_new, fc_new
        qd, fd = q.double(), f_half.double()
        acc["load"] += (u_of(qd[:, -1]) - u_of(qc[:, -1])) ** 2
        acc["q_blk"] += ((qd - qc)[:, blk] ** 2).mean(1)
        acc["q_all"] += ((qd - qc) ** 2).mean(1)
        acc["f_blk"] += ((fd - fc)[:, blk] ** 2).mean(1)
        eb, ebc = energy(qd, fd)[:, blk].sum(1), energy(qc, fc)[:, blk].sum(1)
        acc["e_blk"] += (eb - ebc) ** 2
        emax = torch.maximum(emax, ebc.abs())
        # 露出（古典の解）：dφ_k/dt = u(q_{k-1}) - u(q_k)，dq_k/dt = φ_k - φ_{k+1}（i(φ) = φ）
        uc = u_of(qc)
        dphi = (torch.roll(uc, 1, dims=1) - uc)[:, blk]
        dq = (fc - torch.roll(fc, -1, dims=1))[:, blk]
        acc["expo_f"] += (dphi ** 2).mean(1)
        acc["expo_q"] += (du_of(qc[:, blk]) * dq ** 2).mean(1)
        acc["act"] += ((dphi.abs() > 0.05) | (dq.abs() > 0.05)).double().mean(1)
    out = {k: (v / steps).cpu().numpy() for k, v in acc.items()}
    out["e_scale"] = emax.cpu().numpy()
    out["div"] = div.cpu().numpy()
    return out


dt_g = DT


def ring_reference(ics, steps, dev, sub=50):
    """環の参照解：刻み Δt/sub の RK4（倍精度）。φ_{-1/2} も逆向きに Δt/2 だけ進めて求める。"""
    s = torch.as_tensor(ics, device=dev, dtype=torch.float64)

    def rhs(z):
        q, f = z[:, :NR], z[:, NR:]
        uq = u_of(q)
        return torch.cat([f - torch.roll(f, -1, dims=1), torch.roll(uq, 1, dims=1) - uq], 1)

    def rk4(z, h, k):
        for _ in range(k):
            k1 = rhs(z)
            k2 = rhs(z + 0.5 * h * k1)
            k3 = rhs(z + 0.5 * h * k2)
            k4 = rhs(z + h * k3)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z

    z_half = rk4(s, -DT / sub, sub // 2)
    traj = [s]
    z = s
    for _ in range(steps):
        z = rk4(z, DT / sub, sub)
        traj.append(z)
    return torch.stack(traj, 1), z_half[:, NR:]


def run_ring(model, kind, ics, ref, f_mhalf, steps, dev):
    z0 = torch.as_tensor(ics, device=dev, dtype=torch.float32)
    q, f = z0[:, :NR], z0[:, NR:]
    B = q.shape[0]
    h = torch.zeros(B, max(1, hidden_size_of(model)), device=dev)
    h_pp, h_p = h.clone(), h.clone()
    qs = [q]
    if kind == "stag":
        f_half = f_mhalf.float()
        for _ in range(steps):
            out_v, h_new, h_p_out = model(torch.cat([q, f_half], 1), h_pp, h_p, is_velocity_step=True)
            h_pp, h_p = h_p_out, h_new
            if REST_CORRECT:
                out_v = out_v - rest_inc_stag(model, q, f_half, True)
            f_half = out_v
            out_x, h_new, h_p_out = model(torch.cat([q, f_half], 1), h_pp, h_p, is_velocity_step=False)
            h_pp, h_p = h_p_out, h_new
            if REST_CORRECT:
                out_x = out_x - rest_inc_stag(model, q, f_half, False)
            q = out_x
            qs.append(q)
    else:
        z = z0
        for _ in range(steps):
            zn, _ = model(z, h)
            if REST_CORRECT:
                zn = zn - rest_inc_full(model, z[:, :NR], z[:, NR:])
            z = zn
            qs.append(z[:, :NR])
    qp = torch.stack(qs, 1).double()
    qr = ref[:, :, :NR]
    err = torch.sqrt(((qp[:, 1:] - qr[:, 1:]) ** 2).mean((1, 2))) / qr.abs().amax((1, 2))
    err = torch.where(torch.isfinite(err), err, torch.full_like(err, 1e9))
    return err.cpu().numpy()


def load(short, lr, sd, root, size, dev, n_len):
    name, kind = (O.STAG[short], "stag") if short in O.STAG else (O.FULL[short], "full")
    hs = memoryless_width(name, size, 16)
    rid = f"abs_mse_lr{lr}_bs64_gc1_num_ltctau_cos_s{sd}_gpu"
    d = Path(root) / O.TRAIN_DS / rid / name / f"hs{hs}"
    p = ev.find_model_checkpoint(d, name, hs)
    if not p.exists():
        return None, kind
    model, _ = ev.load_model_for_eval(name, hs, d, p)
    model.set_length(n_len)
    return model.to(dev).eval(), kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ring", "open", "reflect", "busy", "busy_low", "burst", "cw", "prbs", "train", "static"], required=True)
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--models", nargs="+", default=["LF", "SIE", "LF-U", "Res", "RK4", "LF-KCL"])
    ap.add_argument("--lrs", nargs="+", default=["1e2", "3e2"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 9])
    ap.add_argument("--size", default="medium")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--rest-correct", action="store_true", help="一様な状態での網の増分を各段で引く（仕組みの確かめ）")
    ap.add_argument("--sig-seed", type=int, default=None, help="cw・prbs・train の信号の乱数を替える")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global REST_CORRECT
    REST_CORRECT = args.rest_correct
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"mode": args.mode, "runs": [], "rest_correct": REST_CORRECT}
    if args.mode == "static":
        # 一様な状態（真の更新は 0）での網の 1 ステップの増分。整合した線路の直流の状態（u(q) = V，φ = V）と，電流のない状態（φ = 0）
        V = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        qv = (-1.0 + np.sqrt(1.0 + 4.0 * ALPHA * V)) / (2.0 * ALPHA)
        res["V"] = V.tolist()
        for short in args.models:
            for lr in args.lrs:
                for sd in range(args.seeds[0], args.seeds[1] + 1):
                    model, kind = load(short, lr, sd, args.root, args.size, dev, N)
                    if model is None:
                        continue
                    rec = {"model": short, "lr": lr, "seed": sd}
                    for fam, fv in (("locus", V), ("nocurrent", np.zeros_like(V))):
                        q0 = torch.as_tensor(qv, device=dev, dtype=torch.float32)[:, None].expand(len(V), N).contiguous()
                        f0 = torch.as_tensor(fv, device=dev, dtype=torch.float32)[:, None].expand(len(V), N).contiguous()
                        with torch.no_grad():
                            if kind == "stag":
                                dv = rest_inc_stag(model, q0[:, :1].expand(len(V), N).contiguous(), f0, True)[:, 0]
                                dx = rest_inc_stag(model, q0, f0, False)[:, 0]
                            else:
                                d = rest_inc_full(model, q0, f0)
                                dx, dv = d[:, 0], d[:, N]
                        rec[fam] = {"dq": dx.cpu().numpy().tolist(), "dphi": dv.cpu().numpy().tolist()}
                    res["runs"].append(rec)
                    print(short, lr, sd, "locus |dq|", np.round(np.abs(rec["locus"]["dq"]), 6).tolist(), flush=True)
        json.dump(res, open(args.out, "w"))
        return
    if args.mode == "ring":
        man = json.load(open(Path(HERE).parent / "datasets" / O.TRAIN_DS / "manifest.json"))
        ics = np.asarray(man["splits"]["test_interp"]["ics"], dtype=np.float64)
        steps = args.steps or 1000
        with torch.no_grad():
            ref, f_mhalf = ring_reference(ics, steps, dev)
        qr, fr = ref[:, :, :NR], ref[:, :, NR:]
        ur = u_of(qr)
        dphi = torch.roll(ur, 1, dims=2) - ur
        dq = fr - torch.roll(fr, -1, dims=2)
        res["ic_wave_q"] = ics[:, :NR].std(1).tolist()
        res["ic_wave_f"] = ics[:, NR:].std(1).tolist()
        res["ic_region"] = np.abs(ics[:, :NR] + ALPHA * ics[:, :NR] ** 2).max(1).tolist()
        res["expo_f"] = (dphi[:, 1:] ** 2).mean((1, 2)).cpu().numpy().tolist()
        res["expo_q"] = (du_of(qr[:, 1:]) * dq[:, 1:] ** 2).mean((1, 2)).cpu().numpy().tolist()
        res["act"] = ((dphi[:, 1:].abs() > 0.05) | (dq[:, 1:].abs() > 0.05)).double().mean((1, 2)).cpu().numpy().tolist()
        res["steps"] = steps
        for short in args.models:
            for lr in args.lrs:
                for sd in range(args.seeds[0], args.seeds[1] + 1):
                    model, kind = load(short, lr, sd, args.root, args.size, dev, NR)
                    if model is None:
                        print("missing", short, lr, sd, flush=True)
                        continue
                    with torch.no_grad():
                        err = run_ring(model, kind, ics, ref, f_mhalf, steps, dev)
                    res["runs"].append({"model": short, "lr": lr, "seed": sd, "err": err.tolist()})
                    ok = err <= 100
                    print(short, lr, sd, "mean err", float(err[ok].mean()), "div", int((~ok).sum()), flush=True)
        json.dump(res, open(args.out, "w"))
        return
    if args.mode == "open":
        A_np, tr_np = O.pulses()
        A_np, tr_np = A_np[:-1], tr_np[:-1]
        A = torch.as_tensor(A_np, device=dev, dtype=torch.float32)
        tr = torch.as_tensor(tr_np, device=dev, dtype=torch.float32)
        src = lambda t: 2.0 * A * p_of(t, tr)  # noqa: E731
        rs, rl, steps, amp = 1.0, 1.0, args.steps or 2000, A_np
    elif args.mode == "reflect":
        a_np, tr_np = RF.pulses()
        a_np, tr_np = a_np[:-1], tr_np[:-1]
        a = torch.as_tensor(a_np, device=dev, dtype=torch.float32)
        tr = torch.as_tensor(tr_np, device=dev, dtype=torch.float32)
        src = lambda t: a * (1.0 + RF.RS) * p_of(t, tr)  # noqa: E731
        rs, rl, steps, amp = RF.RS, RF.RL, args.steps or RF.STEPS, a_np
    elif args.mode in ("cw", "prbs", "train"):
        # 実用の信号（2026-10-23，学習した網の結果を見る前に決めた。整合した終端，4000 ステップ = 5.1 ns，64 本）
        #   cw：周波数逓倍器・コム発生器の入力。バイアス点のまわりの正弦波，入射の振幅 0.5〜1.5 V，f 0.03〜0.12 / t0（1.2〜4.7 GHz，遮断の 0.1〜0.38）
        #   prbs：信号の品質の解析のビット列。0 と A（0.5〜1.5 V）の NRZ，ビットの長さ 16〜40 t0（1〜2.5 Gb/s），切り替わり 4〜8 t0 の上昇余弦
        #   train：サンプリングの衝撃波の入力。nltlv_open_20261022.py と同じ形のパルス（1〜3 V，立ち上がり 4〜12 t0，平ら 10 t0）を 60〜120 t0 ごとに繰り返す
        n = 64
        rng = np.random.default_rng(args.sig_seed if args.sig_seed is not None else {"cw": 20261026, "prbs": 20261027, "train": 20261028}[args.mode])
        if args.mode == "cw":
            A_np = rng.uniform(0.5, 1.5, n)
            f_np = rng.uniform(0.03, 0.12, n)
            th_np = rng.uniform(0, 2 * math.pi, n)
            A_t, f_t, th_t = (torch.as_tensor(x, device=dev, dtype=torch.float32) for x in (A_np, f_np, th_np))

            def src(t):
                ramp = 0.5 * (1 - torch.cos(math.pi * torch.clamp(t / 10.0, 0, 1)))
                return 2.0 * ramp * A_t * torch.sin(2 * math.pi * f_t * t + th_t)
            res["f"] = f_np.tolist()
        elif args.mode == "prbs":
            A_np = rng.uniform(0.5, 1.5, n)
            tb_np = rng.uniform(16.0, 40.0, n)
            tr_np = rng.uniform(4.0, 8.0, n)
            bits_np = rng.integers(0, 2, (n, 64)).astype(np.float64)
            A_t, tb_t, tr_t = (torch.as_tensor(x, device=dev, dtype=torch.float32) for x in (A_np, tb_np, tr_np))
            bits_t = torch.as_tensor(bits_np, device=dev, dtype=torch.float32)

            def src(t):
                b = torch.floor(t / tb_t).long().clamp(0, 63)
                cur = bits_t.gather(1, b[:, None])[:, 0]
                prev = torch.where(b > 0, bits_t.gather(1, (b - 1).clamp(min=0)[:, None])[:, 0], torch.zeros_like(cur))
                frac = t - b.float() * tb_t
                w = 0.5 * (1 - torch.cos(math.pi * torch.clamp(frac / tr_t, 0, 1)))
                return 2.0 * A_t * (prev + (cur - prev) * w)
            res["tb"] = tb_np.tolist()
        else:
            A_np = rng.uniform(1.0, 3.0, n)
            tr_np = rng.uniform(4.0, 12.0, n)
            per_np = rng.uniform(60.0, 120.0, n)
            A_t, tr_t, per_t = (torch.as_tensor(x, device=dev, dtype=torch.float32) for x in (A_np, tr_np, per_np))

            def src(t):
                return 2.0 * A_t * p_of(torch.remainder(t, per_t), tr_t)
            res["period"] = per_np.tolist()
        rs, rl, steps, amp = 1.0, 1.0, args.steps or 4000, A_np
    else:
        # busy：連続した信号（f 0.1〜0.25 / t0，波長 4〜10 段）。busy_low：同じで周波数だけ低い（0.02〜0.05 / t0，波長 20〜50 段）。
        # burst：busy と同じ信号を，80 t0 ごとに 24 t0 だけの上昇余弦の窓で区切る（揺れている時間と範囲がパルスの試験に近い）
        f_lo, f_hi = (0.02, 0.05) if args.mode == "busy_low" else (0.1, 0.25)
        vdc, a_np, f_np, th_np, c_np = busy_drives(f_lo=f_lo, f_hi=f_hi)
        ph_np = np.random.default_rng(20261025).uniform(0, 80.0, len(vdc))
        ph_t = torch.as_tensor(ph_np, device=dev, dtype=torch.float32)
        burst = args.mode == "burst"
        vdc_t = torch.as_tensor(vdc, device=dev, dtype=torch.float32)
        a_t = torch.as_tensor(a_np, device=dev, dtype=torch.float32)
        f_t = torch.as_tensor(f_np, device=dev, dtype=torch.float32)
        th_t = torch.as_tensor(th_np, device=dev, dtype=torch.float32)
        c_t = torch.as_tensor(c_np, device=dev, dtype=torch.float32)

        def src(t):
            ramp = 0.5 * (1 - torch.cos(math.pi * torch.clamp(t / 10.0, 0, 1)))
            tones = (c_t * torch.sin(2 * math.pi * f_t * t[:, None] + th_t)).sum(1)
            if burst:
                x = torch.remainder(t + ph_t, 80.0)
                env = torch.where(x < 24.0, 0.5 * (1 - torch.cos(2 * math.pi * x / 24.0)), torch.zeros_like(x))
                tones = env * tones
            return 2.0 * ramp * (vdc_t + a_t * tones)
        rs, rl, steps, amp = 1.0, 1.0, args.steps or 4000, a_np
        res["vdc"], res["wave_rms"] = vdc.tolist(), a_np.tolist()
    res.update({"steps": steps, "rs": rs, "rl": rl, "amp": np.asarray(amp).tolist()})
    for short in args.models:
        if short == "classical":
            with torch.no_grad():
                r = run_line(None, "stag", src, rs, rl, steps, dev)
            res["classical"] = {k: v.tolist() for k, v in r.items()}
            print("classical expo_f", float(r["expo_f"].mean()), "expo_q", float(r["expo_q"].mean()), "act", float(r["act"].mean()), flush=True)
            continue
        for lr in args.lrs:
            for sd in range(args.seeds[0], args.seeds[1] + 1):
                model, kind = load(short, lr, sd, args.root, args.size, dev, N)
                if model is None:
                    print("missing", short, lr, sd, flush=True)
                    continue
                with torch.no_grad():
                    r = run_line(model, kind, src, rs, rl, steps, dev)
                rec = {"model": short, "lr": lr, "seed": sd, "div": r["div"].tolist()}
                for k in ("load", "q_blk", "q_all", "f_blk", "e_blk"):
                    rec[k] = r[k].tolist()
                rec["e_scale"] = r["e_scale"].tolist()
                res["runs"].append(rec)
                amp_np = np.asarray(amp)
                ok = ~r["div"]
                print(short, lr, sd, "load", float((np.sqrt(r["load"]) / amp_np)[ok].mean()),
                      "q_blk", float((np.sqrt(r["q_blk"]) / amp_np)[ok].mean()), "div", int((~ok).sum()), flush=True)
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
