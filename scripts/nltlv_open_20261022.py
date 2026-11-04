#!/usr/bin/env python3
"""電源と負荷のある 40 段の開いた線路（物理の単位のバラクタの線路）に，学習した網を埋め込む.

  python3 scripts/nltlv_open_20261022.py --out OUT.json [--models LF SIE LF-U Res RK4 LF-KCL] [--lrs 1e2 3e2] [--seeds 0 9]
  python3 scripts/nltlv_open_20261022.py --reference --out REF.json   （古典の方法の精度：刻み 0.005 の RK4 の解との差）

線路：節点 0..39 のバラクタ（電荷 q_k，u(q) = q + α q²），節点 k−1 と k をつなぐインダクタ（磁束 φ_k，k = 1..39，i(φ) = φ）。
環の表し方のまま，切れ目のインダクタ φ_0 を 0 に固定する。節点 0 には電源（開放電圧 V_s(t)，内部抵抗 R = 1 = 50 Ω）から，
節点 39 からは負荷（R = 1）へ電流が流れる。その 2 つの項は節点の電圧の台形 ū = (u(q_n) + u(q_{n+1}))/2 で中点にそろえ，不動点の反復 3 回で解く。
両端の 4 段ずつ（節点 0〜3，36〜39 と，その磁束 φ_1〜φ_3，φ_36〜φ_39）は真の特性の古典のリープフロッグ，中の 32 段（4〜35）を学習した網に置き換える。
境目は hybrid_line_20261021.py と同じ（古典の隣の段は真の式，整数の時刻の型は磁束の半ステップの平均でつなぐ）。初期状態はバイアス点（すべて 0）。
入力：V_s(t) = 2A p(t)，p は立ち上がり t_r の上昇余弦の辺，平らな部分 10，下がりも t_r の台形（時間の単位 t0 = 25.4 ps）。
試験のパルス 64 本：A ∈ [1, 3]，t_r ∈ [4, 12] の一様乱数（seed 20261022）。図用に A = 2.5，t_r = 8 のパルスを 1 本足す（統計には入れない）。
長さ 100（2000 ステップ）。主の物差し：負荷の電圧 u(q_39) の，すべて古典の解との差の二乗平均平方根 / A（パルスごと，発散を除いて平均）。
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
sys.path.insert(0, os.path.join(HERE, ".."))

import ic_generalization_eval as ev  # noqa: E402
from harmonic_models import hidden_size_of, memoryless_width  # noqa: E402
from lllc_paths import RUNS_ROOT  # noqa: E402

TRAIN_DS = "nltlv8_v1_dtd2"
STAG = {"LF": "LF_Free_Loc", "LF-U": "LF_Par_Loc", "LF-KCL": "LF_KCL_Loc"}
FULL = {"Res": "Res_Loc_state", "RK4": "RK4_Loc_state", "SIE": "SIE_Loc_state"}
N, EDGE, ALPHA, DT, STEPS, R = 40, 4, 0.0925, 0.05, 2000, 1.0
HOLD = 10.0


def u_of(q):
    return q + ALPHA * q * q


def pulses(seed=20261022, n=64):
    rng = np.random.default_rng(seed)
    A = rng.uniform(1.0, 3.0, n)
    tr = rng.uniform(4.0, 12.0, n)
    return np.concatenate([A, [2.5]]), np.concatenate([tr, [8.0]])


def p_of(t, tr):
    """台形（上昇余弦の辺）：0..tr で上がり，tr..tr+HOLD で 1，次の tr で下がる。t, tr は (B,) か broadcast できる形."""
    up = 0.5 * (1 - torch.cos(math.pi * torch.clamp(t / tr, 0, 1)))
    down = 0.5 * (1 + torch.cos(math.pi * torch.clamp((t - tr - HOLD) / tr, 0, 1)))
    return torch.where(t < tr + HOLD, up, down)


class TrueFull(torch.nn.Module):
    """基準（2026-10-22）：真の特性を整数の時刻の型（速度ベルレ法）で進め，整数の時刻の網と同じつなぎ方で埋め込む。
    つなぎ方だけで生じる誤差の床を測る。塊の中の段だけを使うので，電源と負荷の項は入れない。"""

    hidden_size = 1

    def forward(self, z, h):
        n = z.shape[1] // 2
        q, f = z[:, :n], z[:, n:]
        vq = u_of(q)
        fh = f + 0.5 * DT * (torch.roll(vq, 1, dims=1) - vq)
        q1 = q + DT * (fh - torch.roll(fh, -1, dims=1))
        vq1 = u_of(q1)
        f1 = fh + 0.5 * DT * (torch.roll(vq1, 1, dims=1) - vq1)
        return torch.cat([q1, f1], 1), h

    def set_length(self, n):
        pass


def flux_update(q, f_half, dt):
    vq = u_of(q)
    f = f_half + dt * (torch.roll(vq, 1, dims=1) - vq)
    f[:, 0] = 0.0
    return f


def charge_update(q, f_new, dt, vs_mid):
    """古典の電荷の更新（全節点）。両端の電源と負荷の項は ū で中点にそろえる。"""
    i = f_new
    base = q + dt * (i - torch.roll(i, -1, dims=1))
    qn = base.clone()
    for _ in range(3):
        ub = 0.5 * (u_of(q) + u_of(qn))
        qn = base.clone()
        qn[:, 0] = base[:, 0] + dt * (vs_mid - ub[:, 0]) / R
        qn[:, -1] = base[:, -1] - dt * ub[:, -1] / R
    return qn


def zero_offsets(model, kind, dev):
    """バイアス点（すべて 0）での網の 1 ステップの出力。真の更新は 0 なので，これは学習の誤差で，
    線路の中で小さな電源のように働き，基線をずらす。--anchor のときはこれを各段の出力から引く（ゼロ点の校正）。"""
    z = torch.zeros(1, 2 * N, device=dev)
    h = torch.zeros(1, max(1, hidden_size_of(model)), device=dev)
    if kind == "stag":
        v0, _, _ = model(z, h, h, is_velocity_step=True)
        x0, _, _ = model(z, h, h, is_velocity_step=False)
        return v0, x0
    z0, _ = model(z, h)
    return z0[:, N:], z0[:, :N]


def run(model, kind, A, tr, steps=STEPS, dt=DT, record=False, anchor=False):
    dev = A.device
    B = len(A)
    v_off, x_off = zero_offsets(model, kind, dev) if (anchor and model is not None) else (0.0, 0.0)
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
    sq = torch.zeros(B, dtype=torch.float64, device=dev)
    eL = torch.zeros(B, dtype=torch.float64, device=dev)
    eLc = torch.zeros(B, dtype=torch.float64, device=dev)
    uL, uLc = [], []
    for n in range(steps):
        t_mid = (n + 0.5) * dt
        vs = 2.0 * A * p_of(torch.full_like(A, t_mid), tr)
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
            f_new = torch.where(m, out_v - v_off, f_c)
            f_new[:, 0] = 0.0
            q_c = charge_update(q, f_new, dt, vs.float())
            out_x, h_new, h_p_out = model(torch.cat([q, f_new], 1), h_pp, h_p, is_velocity_step=False)
            h_pp, h_p = h_p_out, h_new
            q_new = torch.where(m, out_x - x_off, q_c)
            f_int_new = f_int
        else:
            phi_n = torch.where(m, f_int, 0.5 * (f_half + f_c))
            z_new, _ = model(torch.cat([q, phi_n], 1), h)
            q_l, f_l = z_new[:, :N] - x_off, z_new[:, N:] - v_off
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
        sq += (ul - ulc) ** 2
        eL += ul ** 2 * dt / R
        eLc += ulc ** 2 * dt / R
        if record:
            uL.append(ul.cpu())
            uLc.append(ulc.cpu())
    err = torch.sqrt(sq / steps) / A.double()
    out = {"err": err.cpu().numpy(), "div": div.cpu().numpy(), "eratio": (eL / eLc).cpu().numpy()}
    if record:
        out["uL"] = torch.stack(uL, 1).numpy()
        out["uLc"] = torch.stack(uLc, 1).numpy()
    return out


def edge_times(w, dt=DT):
    """負荷の電圧の立ち上がりの 10%・50%・90% を横切る時刻（最大値に対して，最初の横切り）。"""
    out = []
    for x in w:
        mx = x.max()
        ts = []
        for frac in (0.1, 0.5, 0.9):
            k = int(np.argmax(x >= frac * mx))
            ts.append(k * dt)
        out.append(ts)
    return np.array(out)


def reference(A, tr):
    """古典の方法の精度：同じ回路の ODE を刻み 0.005 の RK4 で解き，Δt = 0.05 の古典の解と負荷の電圧を比べる（倍精度，CPU）。"""
    A, tr = A.cpu().double(), tr.cpu().double()
    B = len(A)

    def rhs(t, q, f):
        vs = 2.0 * A * p_of(torch.full_like(A, t), tr)
        uq = u_of(q)
        dq = f - torch.roll(f, -1, dims=1)
        dq[:, 0] += (vs - uq[:, 0]) / R
        dq[:, -1] -= uq[:, -1] / R
        df = torch.roll(uq, 1, dims=1) - uq
        df[:, 0] = 0.0
        return dq, df

    h, sub = 0.005, 10
    q = torch.zeros(B, N, dtype=torch.float64)
    f = torch.zeros_like(q)
    uL_ref = []
    t = 0.0
    for n in range(STEPS):
        for _ in range(sub):
            k1 = rhs(t, q, f)
            k2 = rhs(t + h / 2, q + h / 2 * k1[0], f + h / 2 * k1[1])
            k3 = rhs(t + h / 2, q + h / 2 * k2[0], f + h / 2 * k2[1])
            k4 = rhs(t + h, q + h * k3[0], f + h * k3[1])
            q = q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            f = f + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            t += h
        uL_ref.append(u_of(q[:, -1]))
    uL_ref = torch.stack(uL_ref, 1)
    cl = run(None, "stag", A, tr, record=True)
    err = torch.sqrt(((torch.as_tensor(cl["uLc"]) - uL_ref) ** 2).mean(1)) / A
    return {"classical_vs_rk4": err.numpy(), "uL_ref": uL_ref.numpy(), "uL_classical": cl["uLc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--models", nargs="+", default=["LF", "SIE", "LF-U", "Res", "RK4", "LF-KCL"])
    ap.add_argument("--lrs", nargs="+", default=["1e2", "3e2"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 9])
    ap.add_argument("--size", default="medium")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--anchor", action="store_true", help="バイアス点での網の出力を引く（ゼロ点の校正）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_np, tr_np = pulses()
    A = torch.as_tensor(A_np, device=dev, dtype=torch.float32)
    tr = torch.as_tensor(tr_np, device=dev, dtype=torch.float32)
    res = {"N": N, "edge": EDGE, "alpha": ALPHA, "dt": DT, "steps": STEPS, "A": A_np.tolist(), "tr": tr_np.tolist(), "runs": []}
    if args.reference:
        with torch.no_grad():
            r = reference(A, tr)
        e = r["classical_vs_rk4"][:-1]
        res["classical_vs_rk4"] = {"mean": float(e.mean()), "max": float(e.max())}
        res["show_ref"] = {"uL_ref": r["uL_ref"][-1].tolist(), "uL_classical": r["uL_classical"][-1].tolist()}
        res["edges_classical"] = edge_times(r["uL_classical"][:-1]).tolist()
        print("classical vs RK4:", res["classical_vs_rk4"], flush=True)
        json.dump(res, open(args.out, "w"))
        return
    for short in args.models:
        if short == "True-full":  # つなぎ方の床（学習なし，1 本だけ）
            name, kind, hs = None, "full", 0
        else:
            name, kind = (STAG[short], "stag") if short in STAG else (FULL[short], "full")
            hs = memoryless_width(name, args.size, 16)
        for lr in (args.lrs if name else ["-"]):
            for sd in (range(args.seeds[0], args.seeds[1] + 1) if name else [0]):
                if name is None:
                    model = TrueFull().to(dev).eval()
                else:
                    rid = f"abs_mse_lr{lr}_bs64_gc1_num_ltctau_cos_s{sd}_gpu"
                    d = Path(args.root) / TRAIN_DS / rid / name / f"hs{hs}"
                    p = ev.find_model_checkpoint(d, name, hs)
                    if not p.exists():
                        print("missing", short, lr, sd, flush=True)
                        continue
                    model, _ = ev.load_model_for_eval(name, hs, d, p)
                    model.set_length(N)
                    model = model.to(dev).eval()
                with torch.no_grad():
                    r = run(model, kind, A, tr, record=True, anchor=args.anchor)
                err, dv = r["err"][:-1], r["div"][:-1]
                ok = np.isfinite(err) & (err <= 100) & ~dv
                ed = edge_times(r["uL"][:-1][ok]) - edge_times(r["uLc"][:-1][ok])
                rec = {"model": short, "lr": lr, "seed": sd, "err": float(err[ok].mean()) if ok.any() else 1e3,
                       "n_div": int((~ok).sum()), "err_med": float(np.median(np.where(ok, err, np.inf))),
                       "eratio": float(np.median(r["eratio"][:-1][ok])) if ok.any() else float("nan"),
                       "dt50_med_abs": float(np.median(np.abs(ed[:, 1]))) if ok.any() else float("nan"),
                       "drise_med_abs": float(np.median(np.abs((ed[:, 2] - ed[:, 0])))) if ok.any() else float("nan"),
                       "show": {"uL": r["uL"][-1].tolist(), "uLc": r["uLc"][-1].tolist(), "div": bool(r["div"][-1])}}
                res["runs"].append(rec)
                print(short, lr, sd, {k: v for k, v in rec.items() if k != "show"}, flush=True)
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
