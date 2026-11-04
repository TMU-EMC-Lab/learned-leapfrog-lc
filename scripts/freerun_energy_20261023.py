#!/usr/bin/env python3
"""自由に走る回路と線路で，学習した網を長く回したときのエネルギーの変化（2026-10-23，ELEX の 4.1 節）.

  python3 scripts/freerun_energy_20261023.py --out OUT [--root RUNS_ROOT] [--workers 24] [--cache RUNS.jsonl]

3 次の回路（nlqf3_v1_dtd2）と 8 段の線路（nltl8_v1_dtd2）の学習した網（LF，SIE，SIE-M，LF-U，Res，RK4-net。
大きさ small・medium × 学習率 1e-2・3e-2 × seed 0〜9 の 40 本ずつ，run_id の末尾 _gpu）を，試験（内挿）の 512 の初期条件から
1 万ステップ回す（CPU，学習はしない）。初期状態は評価（system_eval.initial_states）と同じで，ずらした型（LF，LF-U）は
参照解の φ_{−1/2} から，ほかは (q_0, φ_0) から始める。
エネルギーの物差し：初期条件ごとの r(t) = (H(t) − H_0)/H_0（H_0 は初期条件の真のエネルギー）の，整数の時刻 0..N での二乗平均平方根
（N = 1000 と 10^4）。ずらした型の磁束は隣り合う半ステップの平均で整数の時刻にそろえる。二乗平均平方根が有限でないか 100 を超えた
初期条件は発散として数え，残りの中央値を網の値とする（全部が発散した網は 1e3）。
比 X/LF は同じ seed・大きさ・学習率の対の幾何平均（両方の中央値が 0.3 以上の層は除く。scripts/letter_stats.py と同じ組み方）。
出力：OUT.json（網ごとの値，型ごとの中央値，比）と OUT.txt（要約）。途中の結果は --cache（jsonl）に 1 本ずつ書き，
やり直すときは済んだ網を飛ばす。所要は 24 プロセス（各 2 スレッド）で約 6 分。
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU で回す（torch を読む前に）
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from learned_start_20261028 import geo  # noqa: E402
from lllc_paths import RUNS_ROOT  # noqa: E402

SYSTEMS = {
    "circuit": ("nlqf3_v1_dtd2", {"LF": "LF_Free_MLP", "SIE": "SIE_MLP_state", "SIE-M": "SIEMid_MLP_state", "LF-U": "LF_Par_MLP",
                                  "Res": "Res_MLP_state", "RK4": "RK4_MLP_state"}),
    "line": ("nltl8_v1_dtd2", {"LF": "LF_Free_Loc", "SIE": "SIE_Loc_state", "SIE-M": "SIEMid_Loc_state", "LF-U": "LF_Par_Loc",
                               "Res": "Res_Loc_state", "RK4": "RK4_Loc_state"}),
}
SHORTS = ("LF", "SIE", "SIE-M", "LF-U", "Res", "RK4")
RID = re.compile(r"^abs_mse_lr(?P<lr>1e2|3e2)_bs64_gc1_num_ltctau_cos_s(?P<seed>\d+)_gpu$")
SPLIT = "test_interp"
WINDOWS = (1000, 10000)
DIV = 100.0  # 発散のしきい値（有限でないか 100 を超える）
CAP = 1e3  # 全部の初期条件が発散した網の値


def manifest(ds):
    return json.load(open(ROOT / "datasets" / ds / "manifest.json"))


def run_job(job):
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    import ic_generalization_eval as ev
    import system_eval as se
    from harmonic_ic_dataset import split_to_array
    from harmonic_models import MODEL_REGISTRY
    from systems import system_from_manifest

    ev.DEVICE = torch.device("cpu")
    t0 = time.time()
    man = manifest(job["ds"])
    sysm = system_from_manifest(man)
    ics = split_to_array(man, SPLIT)
    dt = float(man["dt_lf"])
    family = MODEL_REGISTRY[job["model"]]["family"]
    model_dir = Path(job["root"]) / job["ds"] / job["rid"] / job["model"] / f"hs{job['hs']}"
    model, config = ev.load_model_for_eval(job["model"], job["hs"], model_dir, ev.find_model_checkpoint(model_dir, job["model"], job["hs"]))
    k = getattr(sysm, "n_dof", 1)
    n_long = max(WINDOWS)
    z0 = se.initial_states(sysm, ics, dt, family, config.get("teacher"))
    e0 = sysm.energy(ics)
    with np.errstate(all="ignore"):
        if family == "lf":
            # xs[:, n] = q_n，vs[:, m] = φ_{m−1/2}。磁束は隣り合う半ステップの平均で整数の時刻 0..n_long にそろえる
            if k > 1:
                xs, vs = se.rollout_lf_multi(model, z0[:, :k], z0[:, k:], n_long + 2)
            else:
                xs, vs = ev._rollout_lf_batch(model, z0[:, 0], z0[:, 1], n_long + 2)
                xs, vs = xs[:, :, None], vs[:, :, None]
            xs, vs = xs.astype(np.float64), vs.astype(np.float64)
            states = np.concatenate([xs[:, :n_long + 1], 0.5 * (vs[:, :n_long + 1] + vs[:, 1:n_long + 2])], axis=2)
            del xs, vs
        elif family == "direct_state":
            states = se.rollout_state(model, z0, n_long + 1)
        else:
            raise ValueError(family)
        r = (sysm.energy(states) - e0[:, None]) / e0[:, None]  # (初期条件, 時刻 0..n_long)
    del states
    out = {}
    for W in WINDOWS:
        with np.errstate(all="ignore"):
            rms = np.sqrt(np.mean(r[:, :W + 1] ** 2, axis=1))
        ok = np.isfinite(rms) & (rms <= DIV)
        out[str(W)] = {"median": float(np.median(rms[ok])) if ok.any() else float("nan"), "n_div": int((~ok).sum())}
    keep = ("system", "ds", "rid", "lr", "seed", "model", "short", "hs", "size")
    return {**{k_: job[k_] for k_ in keep}, "energy": out, "t_s": round(time.time() - t0, 1)}


def list_jobs(root, systems):
    from harmonic_models import memoryless_width

    jobs = []
    for system in systems:
        ds, models = SYSTEMS[system]
        dim = int(manifest(ds).get("dim", 2))
        widths = {(name, size): memoryless_width(name, size, dim) for name in models.values() for size in ("small", "medium")}
        for rd in sorted((Path(root) / ds).iterdir()):
            m = RID.match(rd.name)
            if not m or not 0 <= int(m["seed"]) <= 9:
                continue
            for short, name in models.items():
                for size in ("small", "medium"):
                    hs = widths[(name, size)]
                    if (rd / name / f"hs{hs}").is_dir():
                        jobs.append({"root": str(root), "system": system, "ds": ds, "rid": rd.name, "lr": m["lr"],
                                     "seed": int(m["seed"]), "model": name, "short": short, "hs": hs, "size": size})
    return jobs


def job_key(j):
    return f"{j['ds']}|{j['rid']}|{j['model']}|{j['hs']}"


def run_all(jobs, cache, workers):
    done = set()
    if cache.exists():
        done = {job_key(json.loads(line)) for line in open(cache) if line.strip()}
    todo = [j for j in jobs if job_key(j) not in done]
    todo.sort(key=lambda j: (j["system"] != "line", j["short"] not in ("RK4", "LF", "LF-U"), j["rid"]))  # 重い網を先に
    print(f"{len(jobs)} networks, {len(done)} done, {len(todo)} to run ({workers} workers)", flush=True)
    if not todo:
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex, open(cache, "a") as f:
        futs = {ex.submit(run_job, j): j for j in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                res = fu.result()
            except Exception as e:  # noqa: BLE001
                print(f"failed: {job_key(futs[fu])}: {e!r}", flush=True)
                continue
            f.write(json.dumps(res) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done ({time.time() - t0:.0f} s)", flush=True)


def fin(v):
    return v if v is not None and math.isfinite(v) and v < CAP else CAP


def analyze(rows):
    res = {"per_model": {}, "ratios": {}}
    L = ["# Energy change of free-running learned networks: scripts/freerun_energy_20261023.py",
         "# Networks: sizes small and medium x learning rates 1e-2 and 3e-2 x seeds 0-9 per model; 512 test (interpolation) initial conditions",
         "# Measure: RMS over integer times 0..N of r = (H - H0)/H0, with staggered models' flux averaged over adjacent half steps;",
         "#   a network's value is the median over initial conditions whose RMS is finite and at most 100 (the others count as diverged),",
         "#   or 1e3 if all diverged",
         "# X/LF: geometric mean of ratios of runs paired by seed, size, and learning rate "
         "(a stratum is left out when both medians are 0.3 or more)", ""]
    for system, (ds, _) in SYSTEMS.items():
        rs = [r for r in rows if r["system"] == system]
        if not rs:
            continue
        val = {(s, W): {(str(r["seed"]), r["lr"], r["size"]): fin(r["energy"][str(W)]["median"]) for r in rs if r["short"] == s}
               for s in SHORTS for W in WINDOWS}
        res["per_model"][system], res["ratios"][system] = {}, {}
        L += [f"## {system} ({ds})", "",
              "| Model | Runs | Energy change, 1000 steps | 10^4 steps | Diverged of 512, 1000 steps | 10^4 steps | X/LF, 1000 steps | 10^4 steps |",
              "|---|---|---|---|---|---|---|---|"]
        for s in SHORTS:
            mine = [r for r in rs if r["short"] == s]
            if not mine:
                continue
            pm = {"n": len(mine)}
            for W in WINDOWS:
                pm[f"energy_{W}_median"] = float(np.median(list(val[(s, W)].values())))
                pm[f"div_{W}_median"] = float(np.median([r["energy"][str(W)]["n_div"] for r in mine]))
            res["per_model"][system][s] = pm
            if s != "LF":
                res["ratios"][system][f"{s}/LF"] = {str(W): geo(val[(s, W)], val[("LF", W)]) for W in WINDOWS}
            rt = res["ratios"][system].get(f"{s}/LF")
            L.append(f"| {s} | {len(mine)} | {pm['energy_1000_median']:.3g} | {pm['energy_10000_median']:.3g} | "
                     f"{pm['div_1000_median']:g} | {pm['div_10000_median']:g} | "
                     + (f"{rt['1000']['ratio']:.4g} ({rt['1000']['pairs']} pairs) | {rt['10000']['ratio']:.4g} ({rt['10000']['pairs']} pairs) |"
                        if rt else "— | — |"))
        L.append("")
    return res, "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--root", default=RUNS_ROOT)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cache", default=None, help="網ごとの結果の jsonl（既定は OUT_runs.jsonl）")
    ap.add_argument("--out", required=True, help="出力の接頭辞（.json と .txt を書く）")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    cache = Path(args.cache or f"{args.out}_runs.jsonl")
    jobs = list_jobs(args.root, args.systems)
    if not args.analyze_only:
        run_all(jobs, cache, args.workers)
    keys = {job_key(j) for j in jobs}
    rows = {}
    for line in open(cache):
        if line.strip():
            r = json.loads(line)
            if job_key(r) in keys:
                rows[job_key(r)] = r
    rows = [rows[k] for k in sorted(rows)]
    res, text = analyze(rows)
    meta = {"script": "scripts/freerun_energy_20261023.py", "windows": WINDOWS, "div": DIV, "cap": CAP, "split": SPLIT, "device": "cpu",
            "missing": len(keys) - len(rows)}
    json.dump({"meta": meta, **res, "runs": rows}, open(out.with_suffix(".json"), "w"), indent=1)
    out.with_suffix(".txt").write_text(text + "\n")
    print(text)
    return 0 if len(rows) == len(keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
