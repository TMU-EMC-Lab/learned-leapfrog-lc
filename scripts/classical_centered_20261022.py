# 真の特性の古典の方法を，LF の並び（自分の変数は区間の始め）と LF-C の並び（自分の変数も中点，不動点の反復）で，
# 電圧・電流と電荷・磁束の両方の変数で回し，電圧の誤差（t = 50，参照の |V| の最大〔下限 0.05〕で割る）を比べる。学習はしない。
import json, os, sys
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "scripts"))
import system_eval as se
from harmonic_ic_dataset import split_to_array
from systems import system_from_manifest
from lllc_paths import WORK_DIR
ROOT = os.path.join(_REPO, "datasets")
PAIRS = {"circuit1": ("nlcapqf_v1_dtd2", "nlcap_v1_dtd2", lambda q: np.tan(q)), "circuit2": ("nllcqf_v1_dtd2", "nllc_e1_v1_dtd2", lambda q: np.tan(q)),
         "circuit3": ("nlindqf_v1_dtd2", "nlind_v1_dtd2", lambda q: q / 2.0), "cubic": ("nlqf3_v1_dtd2", "nlqf3vi_v1_dtd2", lambda q: q + q ** 3)}
def man(ds): return json.load(open(f"{ROOT}/{ds}/manifest.json"))
def verr(p, r):
    sc = np.maximum(np.abs(r).max(axis=1), 0.05)
    with np.errstate(all="ignore"):
        e = np.sqrt(np.mean((p - r) ** 2, axis=1)) / sc
    ok = np.isfinite(e) & (e <= 100)
    return float(e[ok].mean()), int((~ok).sum())
def run(sysm, ics, dt, steps, centered, iters=30):
    z0 = se.initial_states(sysm, ics, dt, "lf", None)
    x, v = z0[:, 0].astype(np.float64), z0[:, 1].astype(np.float64)
    out = [x.copy()]
    for _ in range(steps - 1):
        if centered:
            vn = v.copy()
            for _ in range(iters):
                vn = v + dt * sysm.rhs(np.stack([x, 0.5 * (v + vn)], 1))[:, 1]
            v = vn
            xn = x.copy()
            for _ in range(iters):
                xn = x + dt * sysm.rhs(np.stack([0.5 * (x + xn), v], 1))[:, 0]
            x = xn
        else:
            v = v + dt * sysm.rhs(np.stack([x, v], 1))[:, 1]
            x = x + dt * sysm.rhs(np.stack([x, v], 1))[:, 0]
        out.append(x.copy())
    return np.stack(out, 1)[:, :-1]
res = {}
for key, (dq, dv, tov) in PAIRS.items():
    mq, mv = man(dq), man(dv)
    sq, sv = system_from_manifest(mq), system_from_manifest(mv)
    iq, iv = split_to_array(mq, "test_interp"), split_to_array(mv, "test_interp")
    dt = float(mq["dt_lf"]); steps = se.steps_from_t_end(50.0, dt)
    ref_v = se.cached_reference(sv, mv, "test_interp", iv, dt, steps)[:, :-1, 0]
    r = {}
    for name, sysm, ics, conv in (("qf", sq, iq, tov), ("vi", sv, iv, lambda x: x)):
        for c in (False, True):
            r[f"{name}_{'centered' if c else 'lf'}"] = verr(conv(run(sysm, ics, dt, steps, c)), ref_v)
    res[key] = r
    print(key, {k: (round(v[0], 5), v[1]) for k, v in r.items()}, flush=True)
os.makedirs(os.path.join(WORK_DIR, "crossvar"), exist_ok=True)
json.dump(res, open(os.path.join(WORK_DIR, "crossvar", "classical_centered_20261022.json"), "w"), indent=1)
