#!/usr/bin/env python3
"""学習した段を古典のリープフロッグの線路に埋め込む試験（2026-10-21 夜，ELEX）.

  python3 scripts/hybrid_line_20261021.py --block 8 --steps 10000 --out OUT.json [--models ...] [--seeds 0 9]

32 段の非線形の LC 線路（nltl32_v1_dtd2，内挿の初期条件 512 個）を，真の特性 v(q) = q + q³，i(φ) = φ + φ³ を入れた古典のリープフロッグ
（時間方向は 1 次元の FDTD と同じ組み立て，Δt = 0.05）で解き，連続した block 段だけを 8 段の線路で学習した網（nltl8_v1_dtd2 の _gpu の run）に置き換える。
- ずらした型（LF，LF-U，LF-Sep）：古典の状態 (q_n, φ_{n−1/2}) をそのまま網に渡し，置き換える段の出力だけを使う。
- 整数の時刻の型（Res，RK4，Res2，SIE，SIE-M）：古典の段の φ_n は半ステップの平均 (φ_{n−1/2} + φ_{n+1/2})/2，置き換える段の φ_n は網の状態を使う。
  置き換えた段の φ_{n+1/2} は (φ_n + φ_{n+1})/2 にして古典の隣の段に渡す（2 次の精度のつなぎ方）。
同じ初期条件（φ_{−1/2} は参照解を半ステップ戻して作る）で，すべての段を古典で解いた解（倍精度）を同時に進め，次を測る。
- ずれ：古典の解との位置の差の二乗平均平方根を，初期条件ごとに古典の解の |q| の最大値（下限 0.05）で割り，初期条件で平均（t = 50，100，500）。
- エネルギーのずれ：|⟨H⟩_最後の 1/10 − H(0)|/H(0) の初期条件の平均（H は真の特性。φ_n は半ステップの平均）。
発散した初期条件は，主の評価（system_eval._agg）と同じく平均から除き，数を n_div に残す（2026-10-21 夜の直し：
最初の版はどれか 1 つの初期条件が発散すると，その 1 本の全体を発散として打ち切っていた）。生き残りだけの平均の偏りを見るため，
発散を無限大とした中央値（*_med）と，1 で頭打ちにして発散を 1 とした平均（*_cap）も出す。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import ic_generalization_eval as ev  # noqa: E402
from harmonic_ic_dataset import split_to_array  # noqa: E402
from harmonic_models import hidden_size_of, memoryless_width  # noqa: E402
from systems import system_from_manifest  # noqa: E402
from lllc_paths import RUNS_ROOT  # noqa: E402

TRAIN_DS = "nltl8_v1_dtd2"
TEST_DS = "nltl32_v1_dtd2"
STAG = {"LF": "LF_Free_Loc", "LF-U": "LF_Par_Loc", "LF-Sep": "LF_Sep_Loc", "LF-SepD": "LF_SepD_Loc", "LF-SepDx": "LF_SepDx_Loc",
        "LF-KCL": "LF_KCL_Loc", "LF-KCLD": "LF_KCLD_Loc"}  # LF-KCL・LF-KCLD は 2026-10-22（素子の特性を学びキルヒホッフの法則でつなぐ）  # LF-SepD は 2026-10-21 夜
FULL = {"Res": "Res_Loc_state", "RK4": "RK4_Loc_state", "Res2": "Res2_Loc_state", "SIE": "SIE_Loc_state", "SIE-M": "SIEMid_Loc_state"}
HORIZONS = (1000, 2000, 10000)
DT = 0.05  # 基準の部品の刻み。main でデータの dt_lf に合わせる（2026-10-21 夜：粗い刻みの埋め込みのため）


def v_of(q):
    return q + q ** 3


def i_of(f):
    return f + f ** 3


def energy(q, f):
    return (0.5 * q * q + 0.25 * q ** 4 + 0.5 * f * f + 0.25 * f ** 4).sum(-1)


def classical_velocity(q, f_half, dt):
    vq = v_of(q)
    return f_half + dt * (torch.roll(vq, 1, dims=1) - vq)


LOSS_G = 0.0  # 損失のある線路（nltlg）の漏れのコンダクタンス。--loss-g で設定する（2026-10-21 夜）


def classical_position(q, f_half_new, dt):
    i = i_of(f_half_new)
    q_new = q + dt * (i - torch.roll(i, -1, dims=1))
    if LOSS_G:  # 損失の項 −G u(q̄) は q̄ = (q_n + q_{n+1})/2 で中点にそろえ，不動点の反復 3 回で解く
        base = q_new
        q_new = base - dt * LOSS_G * v_of(q)
        for _ in range(3):
            q_new = base - dt * LOSS_G * v_of(0.5 * (q + q_new))
    return q_new


class ExactStag(torch.nn.Module):
    """基準：真の式のリープフロッグを LF 型の呼び出し方で返す（全段）。置き換えても古典の解と同じになるはず."""

    hidden_size = 1

    def forward(self, x, h_pp, h_p, is_velocity_step):
        n = x.shape[1] // 2
        q, f = x[:, :n], x[:, n:]
        out = classical_velocity(q, f, DT) if is_velocity_step else classical_position(q, f, DT)
        return out, h_p, h_pp


class ExactFull(torch.nn.Module):
    """基準：整数の時刻の状態を真の式の速度ベルレ法で 1 ステップ進める（全段）。半ステップの平均でつなぐときの差だけを見る."""

    hidden_size = 1

    def forward(self, z, h):
        n = z.shape[1] // 2
        q, f = z[:, :n], z[:, n:]
        vq = v_of(q)
        fh = f + 0.5 * DT * (torch.roll(vq, 1, dims=1) - vq)
        q1 = classical_position(q, fh, DT)
        vq1 = v_of(q1)
        f1 = fh + 0.5 * DT * (torch.roll(vq1, 1, dims=1) - vq1)
        return torch.cat([q1, f1], 1), h


def run(model, kind, q0, f_half0, f0, mask, steps, dt, trace=None, mask_x=None):
    """kind = 'stag' か 'full'（model が None なら全段が古典）。初期条件ごとに発散を扱う（system_eval._agg と同じ決まり：
    有限で 100 以下の初期条件だけで平均し，発散した数を数える。どの初期条件も残らなければ 1e3）."""
    dev = q0.device
    n_ic, n = q0.shape
    qc, fc = q0.double(), f_half0.double()  # 古典の参照解（倍精度）
    q, f_half = q0.float(), f_half0.float()  # 埋め込みの解（単精度）
    f_int = f0.float()  # 整数の時刻の型が置き換えた段で持つ φ_n
    m = mask.to(dev)  # 磁束（速度）の更新を網に任せる段
    mx = (mask if mask_x is None else mask_x).to(dev)  # 電荷（位置）の更新を網に任せる段（既定は同じ）
    sq0, sf0 = q0.double().sum(1), f_half0.double().sum(1)
    nrm_q = torch.sqrt((q0.double() ** 2).sum(1)).clamp(min=1e-12)
    nrm_f = torch.sqrt((f_half0.double() ** 2).sum(1)).clamp(min=1e-12)
    dq_snap, df_snap = {}, {}
    h = torch.zeros(n_ic, max(1, hidden_size_of(model)) if model is not None else 1, device=dev)
    h_pp, h_p = h.clone(), h.clone()
    sq = torch.zeros(n_ic, dtype=torch.float64, device=dev)
    qmax = q0.double().abs().amax(1)
    e_sum = {T: torch.zeros(n_ic, dtype=torch.float64, device=dev) for T in HORIZONS if T <= steps}
    ec_sum = {T: torch.zeros(n_ic, dtype=torch.float64, device=dev) for T in e_sum}  # 古典の解のエネルギー（損失のある線路で比べる）
    e_cnt = {T: 0 for T in e_sum}
    snap = {}
    anchor = None
    for t in range(steps):
        fc_new = classical_velocity(qc, fc, dt)
        qc_new = classical_position(qc, fc_new, dt)
        f_c = classical_velocity(q, f_half, dt)
        if model is None:
            f_new = f_c
            q_new = classical_position(q, f_new, dt)
            f_int_new = f_int
        elif kind in ("kclc", "kclc_raw"):  # 境目をそろえた LF-KCL(D)：素子ごとに式を 1 回だけ評価し（塊の中は網，外は真の式），両隣に同じ値を渡す
            # 輪の中の差の形では u_θ・i_ψ の定数の足し分は打ち消し合って学習で決まらない（40 本の学習後の u_θ(0) は −0.62〜0.40，i_ψ(0) は −1.25〜0.14）。
            # 境目ではこの定数が電源のように働くので，kclc では u_θ(0) = i_ψ(0) = 0 に固定する（kclc_raw は固定しない対照）
            if anchor is None:
                z0 = torch.zeros(1, 1, device=dev, dtype=q.dtype)
                anchor = (model.U(z0), model.I(z0)) if kind == "kclc" else (0.0, 0.0)
            uq = torch.where(m, model.U(q) - anchor[0], v_of(q))
            f_new = f_half + dt * (torch.roll(uq, 1, dims=1) - uq)
            ib = torch.where(m, model.I(f_new) - anchor[1], i_of(f_new))
            base = q + dt * (ib - torch.roll(ib, -1, dims=1))
            q_new = base
            if hasattr(model, "D") or LOSS_G:  # 漏れ：塊の中は学習した d_χ(q̄)，外は真の G u(q̄)，どちらも不動点の反復 3 回
                def leak(qb):
                    true_leak = LOSS_G * v_of(qb)
                    return torch.where(m, model.D(qb) if hasattr(model, "D") else torch.zeros_like(qb), true_leak)
                q_new = base - dt * leak(q)
                for _ in range(3):
                    q_new = base - dt * leak(0.5 * (q + q_new))
            f_int_new = f_int
        elif kind == "stag":
            out_v, h_new, h_p_out = model(torch.cat([q, f_half], 1), h_pp, h_p, is_velocity_step=True)
            h_pp, h_p = h_p_out, h_new
            f_new = torch.where(m, out_v, f_c)
            q_c = classical_position(q, f_new, dt)
            out_x, h_new, h_p_out = model(torch.cat([q, f_new], 1), h_pp, h_p, is_velocity_step=False)
            h_pp, h_p = h_p_out, h_new
            q_new = torch.where(mx, out_x, q_c)
            f_int_new = f_int
        else:
            phi_n = torch.where(m, f_int, 0.5 * (f_half + f_c))
            z_new, _ = model(torch.cat([q, phi_n], 1), h)
            q_l, f_l = z_new[:, :n], z_new[:, n:]
            f_new = torch.where(m, 0.5 * (f_int + f_l), f_c)
            q_c = classical_position(q, f_new, dt)
            q_new = torch.where(mx, q_l, q_c)
            f_int_new = torch.where(m, f_l, f_int)
        phi_t = 0.5 * (f_half + f_new)
        if kind == "full" and model is not None:
            phi_t = torch.where(m, f_int, phi_t)
        for T in e_sum:
            if T - max(1, T // 10) <= t < T:
                e_sum[T] += energy(q.double(), phi_t.double())
                ec_sum[T] += energy(qc, 0.5 * (fc + fc_new))
                e_cnt[T] += 1
        sq += ((q.double() - qc) ** 2).mean(1)
        qmax = torch.maximum(qmax, qc.abs().amax(1))
        if t + 1 in HORIZONS:
            snap[t + 1] = (sq.clone(), qmax.clone())
            dq_snap[t + 1] = torch.abs(q_new.double().sum(1) - sq0) / nrm_q  # 総電荷のずれ（真の輪では保存）
            df_snap[t + 1] = torch.abs(f_new.double().sum(1) - sf0) / nrm_f  # 総磁束のずれ
        if trace is not None and (t + 1) % 100 == 0:
            ok = torch.isfinite(q).all(1) & (q.abs().amax(1) < 1e3)
            trace.append((t + 1, int(ok.sum()), float(torch.nan_to_num(q.abs().amax(1), nan=1e9, posinf=1e9).max())))
        q, f_half, f_int = q_new, f_new, f_int_new
        qc, fc = qc_new, fc_new
    e0 = energy(q0.double(), f0.double())

    def agg(v):
        ok = torch.isfinite(v) & (v <= 100)
        nd = int((~ok).sum())
        return (float(v[ok].mean()) if ok.any() else 1e3), nd

    def robust(v):  # 発散（有限でない・100 超）を無限大として扱った中央値と，1 で頭打ちにした平均（発散は 1）
        w = torch.where(torch.isfinite(v) & (v <= 100), v, torch.full_like(v, float("inf")))
        return float(w.median()), float(torch.clamp(w, max=1.0).mean())

    res = {}
    for T in HORIZONS:
        if T > steps:
            continue
        sqT, qmT = snap[T]
        dv = torch.sqrt(sqT / T) / torch.clamp(qmT, min=0.05)
        dr = torch.abs(e_sum[T] / max(1, e_cnt[T]) - e0) / e0
        dev_mean, nd_dev = agg(dv)
        drift_mean, nd_e = agg(dr)
        drift_med, drift_cap = robust(dr)
        dev_med, dev_cap = robust(dv)
        edev_mean, _ = agg(torch.abs(e_sum[T] - ec_sum[T]) / max(1, e_cnt[T]) / e0)  # 古典の解とのエネルギーの差（損失のある線路の物差し）
        res[str(T)] = {"dev": dev_mean, "drift": drift_mean, "n_div": max(nd_dev, nd_e),
                       "drift_med": drift_med, "drift_cap": drift_cap, "dev_med": dev_med, "dev_cap": dev_cap, "edev": edev_mean,
                       "dq": agg(dq_snap[T])[0], "dphi": agg(df_snap[T])[0]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--n-ics", type=int, default=512)
    ap.add_argument("--models", nargs="+", default=list(STAG) + list(FULL))
    ap.add_argument("--sizes", nargs="+", default=["small", "medium"])
    ap.add_argument("--lrs", nargs="+", default=["1e2", "3e2"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 9])
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-ds", default=TRAIN_DS)
    ap.add_argument("--test-ds", default=TEST_DS)
    ap.add_argument("--loss-g", type=float, default=0.0, help="損失のある線路（nltlg）の漏れのコンダクタンス G。古典の段と基準の部品に入れる")
    ap.add_argument("--debug", action="store_true", help="1 本目だけ，100 ステップごとに生きている初期条件の数と |q| の最大を出す")
    ap.add_argument("--references", action="store_true", help="基準だけを測る：全段が古典，真の式のずらした部品，真の式の整数の時刻の部品")
    ap.add_argument("--run-tag", default="_gpu", help="学習の run_id の末尾（2026-10-22：学習の手順を変えた run は _ro10_gpu，_nz1_gpu）")
    ap.add_argument("--interface", choices=["true", "learned", "consistent", "consistent-raw"], default="true",
                    help="true：古典の隣の段が境目の特性を真の式で計算（従来）。learned：境目の外側の磁束 1 つと電荷 1 つも網の出力で更新し，学習した段の状態に真の式を使わない。"
                         "consistent（LF-KCL・LF-KCLD だけ，2026-10-22）：素子ごとに特性を 1 回だけ評価し，"
                         "塊の中の素子は学習した u_θ・i_ψ（・d_χ），外は真の式を両隣の更新に渡す。輪全体で総電荷・総磁束が丸め誤差まで保存される。"
                         "学習で決まらない定数は u_θ(0) = i_ψ(0) = 0 に固定する（consistent-raw は固定しない対照）")
    ap.add_argument("--horizons", nargs="+", type=int, default=None,
                    help="エネルギーと古典の解とのずれを記録するステップ（既定 1000 2000 10000。2026-10-21 夜：時間変化の図のために足した）")
    args = ap.parse_args()
    global LOSS_G, HORIZONS, DT
    LOSS_G = float(args.loss_g)
    if args.horizons:
        HORIZONS = tuple(sorted(set(args.horizons)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.load(open(os.path.join(HERE, "..", "datasets", args.test_ds, "manifest.json")))
    assert float(manifest.get("params", {}).get("G", 0.0)) == LOSS_G, "試験の線路の G と --loss-g が違う"
    sysm = system_from_manifest(manifest)
    dt = float(manifest["dt_lf"])
    DT = dt
    n = int(manifest["params"]["n"])
    ics = split_to_array(manifest, "test_interp")[:args.n_ics]
    f_half0 = sysm.half_step_velocity(ics, dt, 1)[:, 0]
    q0 = torch.as_tensor(ics[:, :n], device=dev)
    f0 = torch.as_tensor(ics[:, n:], device=dev)
    fh0 = torch.as_tensor(np.asarray(f_half0).reshape(len(ics), n), device=dev)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:args.block] = True
    mask_x = None
    if args.interface == "learned":  # 置き換える段 0..block−1 の右の段の磁束と，左の段（輪の n−1）の電荷も網で更新する
        mask_x = mask.clone()
        mask_x[n - 1] = True
        mask = mask.clone()
        mask[args.block] = True
    results = {"block": args.block, "steps": args.steps, "n_ics": len(ics), "dt": dt, "interface": args.interface, "runs": []}
    if args.references:
        for key, (m, kind) in {"classical": (None, "stag"), "exact_stag": (ExactStag(), "stag"), "exact_full": (ExactFull(), "full")}.items():
            with torch.no_grad():
                results[key] = run(m, kind, q0, fh0, f0, mask, args.steps, dt, mask_x=mask_x)
            print(key, json.dumps(results[key]), flush=True)
    else:
        for short in args.models:
            name, kind = (STAG[short], "stag") if short in STAG else (FULL[short], "full")
            if args.interface in ("consistent", "consistent-raw"):
                assert short in ("LF-KCL", "LF-KCLD"), "--interface consistent は LF-KCL・LF-KCLD だけ"
                kind = "kclc" if args.interface == "consistent" else "kclc_raw"
            for sz in args.sizes:
                hs = memoryless_width(name, sz, 16)
                for lr in args.lrs:
                    for sd in range(args.seeds[0], args.seeds[1] + 1):
                        rid = f"abs_mse_lr{lr}_bs64_gc1_num_ltctau_cos_s{sd}{args.run_tag}"
                        d = Path(args.root) / args.train_ds / rid / name / f"hs{hs}"
                        p = ev.find_model_checkpoint(d, name, hs)
                        if not p.exists():
                            continue
                        model, _ = ev.load_model_for_eval(name, hs, d, p)
                        model.set_length(n)
                        model = model.to(dev).eval()
                        tr = [] if args.debug else None
                        with torch.no_grad():
                            r = run(model, kind, q0, fh0, f0, mask, args.steps, dt, trace=tr, mask_x=mask_x)
                        if args.debug:
                            print(short, sz, lr, sd, "trace", tr, flush=True)
                        results["runs"].append({"model": short, "size": sz, "lr": lr, "seed": sd, **{f"T{T}": v for T, v in r.items()}})
                        print(short, sz, lr, sd, json.dumps(r), flush=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
