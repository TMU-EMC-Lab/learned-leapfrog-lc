#!/usr/bin/env python3
"""駆動したバラクタの線路の物差しの基準（すべて古典の線路，Δt = 0.05）が，細かい刻みの RK4 の解からどれだけずれるか（2026-10-29，ELEX の 4.4 節）.

  python3 scripts/nltlv_fine_rk_ref_20261029.py --out OUT.json [--drives open reflect train prbs cw train_new cwrep] [--runs-csv results/letter_runs.csv]

表 IV の信号の組ごとに，nltlv_diag_20261022.py の評価と同じ 64 本の信号・終端・ステップ数で，古典の線路と，同じ回路の ODE を
刻み 0.005（Δt の 1/10）の RK4 で解いた解（倍精度，CPU，右辺は nltlv_open・nltlv_reflect の --reference と同じ）を比べる。
負荷の電圧は，差の二乗平均平方根を評価と同じ振幅（入射の振幅）で割る。ブロックのエネルギーは，中の 32 段の Σ_k [U(q_k) + K(φ_k)]
（バイアス点から）の差の二乗平均平方根を，RK4 の解でのその最大で割る（評価と同じく電荷は t_{n+1}，磁束は t_{n+1/2} の組で取り，
最大が 0 の空の信号は除く）。主の古典の線路は電源の電圧を倍精度で計算する（--reference と同じ）。評価と同じく単精度で計算した線路の値も並べる。
整合したパルスと反射のあるパルスの負荷の電圧の平均は，nltlv_open_20261022.py・nltlv_reflect_20261022.py の --reference の
classical_vs_rk4 と同じ量である。--runs-csv を渡すと，各組の負荷の電圧の平均を，その組の信号を使う表 IV の行での
LF の誤差の中央値（results/letter_runs.csv）で割った比も出す。出力の要約は OUT.txt に書く。
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import nltlv_diag_20261022 as D  # noqa: E402
import nltlv_open_20261022 as O  # noqa: E402
import nltlv_reflect_20261022 as RF  # noqa: E402

N, EDGE, DT = O.N, O.EDGE, O.DT
BLK = slice(EDGE, N - EDGE)
H, SUB = 0.005, 10  # RK4 の刻みと，Δt あたりの刻みの数（--reference と同じ）
u_of, p_of, flux_update, energy = O.u_of, O.p_of, O.flux_update, D.energy

# 組の名前：(nltlv_diag の --mode, --sig-seed, 表 IV の行)
DRIVES = {
    "open": ("open", None, "Pulses, matched (also the matched pulses of the 2nd and 3rd sets)"),
    "reflect": ("reflect", None, "Pulses, reflected (also the reflected pulses of the 2nd and 3rd sets)"),
    "train": ("train", None, "Pulse trains"),
    "prbs": ("prbs", None, "Bit streams"),
    "cw": ("cw", None, "Sinusoids"),
    "train_new": ("train", 20261031, "Pulse trains of the 2nd and 3rd pulse sets (--sig-seed 20261031)"),
    "cwrep": ("cw", 20261029, "Sinusoids, 2nd set (--sig-seed 20261029)"),
}
ROWS = {"Pulses, matched": ["open"], "Pulses, reflected": ["reflect"], "Pulse trains": ["train"], "Bit streams": ["prbs"],
        "Sinusoids": ["cw"], "Pulses, 2nd set": ["open", "reflect", "train_new"],
        "Pulses, 3rd set": ["open", "reflect", "train_new"], "Sinusoids, 2nd set": ["cwrep"]}
# 組の信号を使う results/letter_runs.csv の負荷の電圧の source（scripts/letter_data.py）
CSV_SOURCES = {"open": ("pulses", "pulses-2", "pulses-3"), "reflect": ("reflected", "reflected-2", "reflected-3"),
               "train": ("trains",), "train_new": ("trains-2", "trains-3"), "prbs": ("bits",), "cw": ("sinusoids",),
               "cwrep": ("sinusoids-2",)}


def signals(mode, sig_seed, n=64):
    """nltlv_diag_20261022.main と同じ乱数の引き方の信号のパラメータ（倍精度の numpy）と，終端・ステップ数・振幅。"""
    if mode == "open":
        A, tr = O.pulses()
        P = {"A": A[:-1], "tr": tr[:-1]}
        return P, 1.0, 1.0, 2000, P["A"]
    if mode == "reflect":
        a, tr = RF.pulses()
        P = {"a": a[:-1], "tr": tr[:-1]}
        return P, RF.RS, RF.RL, RF.STEPS, P["a"]
    rng = np.random.default_rng(sig_seed if sig_seed is not None else {"cw": 20261026, "prbs": 20261027, "train": 20261028}[mode])
    if mode == "cw":
        P = {"A": rng.uniform(0.5, 1.5, n), "f": rng.uniform(0.03, 0.12, n), "th": rng.uniform(0, 2 * math.pi, n)}
    elif mode == "prbs":
        P = {"A": rng.uniform(0.5, 1.5, n), "tb": rng.uniform(16.0, 40.0, n), "tr": rng.uniform(4.0, 8.0, n),
             "bits": rng.integers(0, 2, (n, 64)).astype(np.float64)}
    else:
        P = {"A": rng.uniform(1.0, 3.0, n), "tr": rng.uniform(4.0, 12.0, n), "per": rng.uniform(60.0, 120.0, n)}
    return P, 1.0, 1.0, 4000, P["A"]


def source(mode, P, dtype):
    """評価の src と同じ式の電源の開放電圧 V_s(t)。パラメータは評価と同じく単精度に丸めてから dtype で計算する（評価は単精度）。"""
    T = {k: torch.as_tensor(v, dtype=torch.float32).to(dtype) for k, v in P.items()}
    if mode == "open":
        return lambda t: 2.0 * T["A"] * p_of(t, T["tr"])
    if mode == "reflect":
        return lambda t: T["a"] * (1.0 + RF.RS) * p_of(t, T["tr"])
    if mode == "train":
        return lambda t: 2.0 * T["A"] * p_of(torch.remainder(t, T["per"]), T["tr"])
    if mode == "cw":
        def src(t):
            ramp = 0.5 * (1 - torch.cos(math.pi * torch.clamp(t / 10.0, 0, 1)))
            return 2.0 * ramp * T["A"] * torch.sin(2 * math.pi * T["f"] * t + T["th"])
        return src

    def src(t):
        b = torch.floor(t / T["tb"]).long().clamp(0, 63)
        cur = T["bits"].gather(1, b[:, None])[:, 0]
        prev = torch.where(b > 0, T["bits"].gather(1, (b - 1).clamp(min=0)[:, None])[:, 0], torch.zeros_like(cur))
        frac = t - b.to(dtype) * T["tb"]
        w = 0.5 * (1 - torch.cos(math.pi * torch.clamp(frac / T["tr"], 0, 1)))
        return 2.0 * T["A"] * (prev + (cur - prev) * w)
    return src


def classical(src, rs, rl, steps, B, t_dtype):
    """評価の古典の線路（nltlv_diag_20261022.run_line の qc・fc と同じ更新）。電源の電圧は時刻 (n + 1/2)Δt に t_dtype で計算する。
    負荷の電圧 u(q_39^{n+1}) と，ブロックのエネルギー E(q^{n+1}, φ^{n+1/2}) を (B, steps) で返す。"""
    qc = torch.zeros(B, N, dtype=torch.float64)
    fc = torch.zeros_like(qc)
    uL = torch.empty(B, steps, dtype=torch.float64)
    eB = torch.empty_like(uL)
    for n in range(steps):
        vs = src(torch.full((B,), (n + 0.5) * DT, dtype=t_dtype))
        fc = flux_update(qc, fc, DT)
        qc = RF.charge_update(qc, fc, DT, vs.double(), rs=rs, rl=rl)
        uL[:, n] = u_of(qc[:, -1])
        eB[:, n] = energy(qc, fc)[:, BLK].sum(1)
    return uL, eB


def rk4(src, rs, rl, steps, B):
    """同じ回路の ODE を刻み H の RK4 で解く（倍精度，CPU）。右辺は nltlv_open・nltlv_reflect の reference と同じで，終端の抵抗だけを引数にした。
    負荷の電圧 u(q_39(t_{n+1})) と，古典の線路と同じ時刻の組のブロックのエネルギー E(q(t_{n+1}), φ(t_{n+1/2})) を返す。"""
    def rhs(t, q, f):
        vs = src(torch.full((B,), t, dtype=torch.float64))
        uq = u_of(q)
        dq = f - torch.roll(f, -1, dims=1)
        dq[:, 0] += (vs - uq[:, 0]) / rs
        dq[:, -1] -= uq[:, -1] / rl
        df = torch.roll(uq, 1, dims=1) - uq
        df[:, 0] = 0.0
        return dq, df

    q = torch.zeros(B, N, dtype=torch.float64)
    f = torch.zeros_like(q)
    uL = torch.empty(B, steps, dtype=torch.float64)
    eB = torch.empty_like(uL)
    for n in range(steps):
        for j in range(SUB):
            t = (n * SUB + j) * H
            k1 = rhs(t, q, f)
            k2 = rhs(t + H / 2, q + H / 2 * k1[0], f + H / 2 * k1[1])
            k3 = rhs(t + H / 2, q + H / 2 * k2[0], f + H / 2 * k2[1])
            k4 = rhs(t + H, q + H * k3[0], f + H * k3[1])
            q = q + H / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            f = f + H / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if j == SUB // 2 - 1:
                f_half = f  # φ(t_{n+1/2})
        uL[:, n] = u_of(q[:, -1])
        eB[:, n] = energy(q, f_half)[:, BLK].sum(1)
    return uL, eB


def deviations(uc, ec, ur, er, amp):
    """信号ごとの，負荷の電圧のずれ / 振幅と，ブロックのエネルギーのずれ / RK4 の解でのその最大（空の信号は nan）。"""
    load = torch.sqrt(((uc - ur) ** 2).mean(1)) / torch.as_tensor(amp, dtype=torch.float64)
    sc = er.abs().amax(1)
    e = torch.sqrt(((ec - er) ** 2).mean(1)) / sc.clamp(min=1e-12)
    e = torch.where(sc > 1e-9, e, torch.full_like(e, float("nan")))
    return load.numpy(), e.numpy()


def stats(x):
    ok = np.isfinite(x)
    return {"mean_pct": float(100 * x[ok].mean()), "max_pct": float(100 * x[ok].max()), "n": int(ok.sum())}


def as_list(x):
    return [float(v) if np.isfinite(v) else None for v in x]


def lf_ratios(res, csv_path):
    """各組の負荷の電圧の平均を，その組の信号を使う行での LF の誤差の中央値（run ごとの値の中央値）で割る。"""
    vals = {}
    for r in csv.DictReader(open(csv_path)):
        if r["model"] == "LF":
            vals.setdefault(r["source"], []).append(float(r["value"]))
    out = {}
    for name, rec in res["drives"].items():
        for s in CSV_SOURCES[name]:
            if s in vals:
                lf = float(np.median(vals[s]))
                out[f"{name}/{s}"] = {"classical_mean": rec["summary"]["main"]["load"]["mean_pct"] / 100, "lf_median": lf,
                                      "n_lf": len(vals[s]), "ratio": rec["summary"]["main"]["load"]["mean_pct"] / 100 / lf}
    return out


def write_summary(res, path):
    L = ["# All-classical varactor line (dt = 0.05) against RK4 with substep 0.005 (double precision, CPU): "
         "scripts/nltlv_fine_rk_ref_20261029.py",
         "# Load voltage: RMS deviation / incident amplitude, mean and max over the 64 signals (%)",
         "# Block energy: RMS deviation of the interior 32 sections' sum[U(q)+K(phi)] / its maximum in the RK4 solution, "
         "mean and max over the non-empty signals (%)",
         "# Main: source voltage in double precision. In brackets: source voltage in single precision, as in the evaluation", ""]
    for name, r in res["drives"].items():
        m, s = r["summary"]["main"], r["summary"]["eval_src32"]
        n_e = m["energy"]["n"]
        L.append(f"- {name} ({r['row']}, {r['steps']} steps): load voltage mean {m['load']['mean_pct']:.3g}%, max {m['load']['max_pct']:.3g}%"
                 f" [{s['load']['mean_pct']:.3g}%, {s['load']['max_pct']:.3g}%]; block energy mean {m['energy']['mean_pct']:.3g}%, "
                 f"max {m['energy']['max_pct']:.3g}% [{s['energy']['mean_pct']:.3g}%, {s['energy']['max_pct']:.3g}%]"
                 + (f" ({len(r['amp']) - n_e} empty signals left out, {n_e} used)" if n_e < len(r["amp"]) else ""))
    L += ["", "Drives of README.md and signal groups: " + "; ".join(f"{k} = {' + '.join(v)}" for k, v in res["table4_rows"].items())]
    if "lf_ratio" in res:
        L += ["", "Mean load-voltage deviation of the classical line / LF's median load-voltage error in each row using the group "
              "(results/letter_runs.csv, model LF)"]
        for k, v in res["lf_ratio"].items():
            L.append(f"- {k}: {100 * v['classical_mean']:.3g}% / {100 * v['lf_median']:.3g}% ({v['n_lf']} runs) = {v['ratio']:.3f}")
        L.append(f"- largest: {max(v['ratio'] for v in res['lf_ratio'].values()):.3f}")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drives", nargs="+", choices=list(DRIVES), default=list(DRIVES))
    ap.add_argument("--steps", type=int, default=None, help="試しの実行用：ステップ数を切り詰める")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs-csv", default=None, help="results/letter_runs.csv：LF の誤差の中央値との比を出す")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    res = {"dt": DT, "rk_h": H, "N": N, "block": [EDGE, N - EDGE], "torch": torch.__version__, "table4_rows": ROWS, "drives": {}}
    for name in args.drives:
        mode, sig_seed, row = DRIVES[name]
        P, rs, rl, steps, amp = signals(mode, sig_seed)
        steps = args.steps or steps
        B = len(amp)
        t0 = time.time()
        with torch.no_grad():
            ur, er = rk4(source(mode, P, torch.float64), rs, rl, steps, B)
            u64, e64 = classical(source(mode, P, torch.float64), rs, rl, steps, B, torch.float64)
            u32, e32 = classical(source(mode, P, torch.float32), rs, rl, steps, B, torch.float32)
        rec = {"mode": mode, "sig_seed": sig_seed, "row": row, "steps": steps, "rs": rs, "rl": rl, "amp": amp.tolist(),
               "e_scale_rk": er.abs().amax(1).tolist(), "per_signal": {}, "summary": {}}
        for key, (uc, ec) in (("main", (u64, e64)), ("eval_src32", (u32, e32))):
            load, e = deviations(uc, ec, ur, er, amp)
            rec["per_signal"][key] = {"load": as_list(load), "energy": as_list(e)}
            rec["summary"][key] = {"load": stats(load), "energy": stats(e)}
        rec["elapsed_s"] = round(time.time() - t0, 1)
        res["drives"][name] = rec
        s = rec["summary"]["main"]
        print(name, f"load mean {s['load']['mean_pct']:.4g}% max {s['load']['max_pct']:.4g}%",
              f"energy mean {s['energy']['mean_pct']:.4g}% max {s['energy']['max_pct']:.4g}% (n={s['energy']['n']})",
              f"{rec['elapsed_s']} s", flush=True)
        json.dump(res, open(out, "w"))
    if args.runs_csv:
        res["lf_ratio"] = lf_ratios(res, args.runs_csv)
        json.dump(res, open(out, "w"))
    write_summary(res, out.with_suffix(".txt"))


if __name__ == "__main__":
    main()
