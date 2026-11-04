"""Ratios, intervals, and tests of the letter (Sect. 3.3), used for the ratios and daggers of Tables V and VI, Fig. 3, and the text.

Two models A and B are compared on runs matched by seed within strata (size x learning rate; learning rate alone for the
varactor line, whose networks have one size).
  - A stratum is left out when either model has fewer than 5 runs in it, or when the medians of both models' errors are
    0.3 or more (both models failed). With LEAVE_OUT = "any" (letter_tables.py --leave-out any), it is left out when
    either median is 0.3 or more, the rule of the plans of H1-H6 (docs/planned_tests.csv).
  - Each matched pair gives D = log(e_A / e_B). A ratio of ratios, (A/B in condition 1) / (A/B in condition 2), uses
    D = log(e_A1 / e_B1) - log(e_A2 / e_B2) over the pairs present in both conditions and the strata kept in both.
    A ratio of two pairs' ratios in one condition, (A1/B1) / (A2/B2), uses D = log(e_A1 / e_B1) - log(e_A2 / e_B2) over
    the seeds that all four models have and the strata kept for both pairs.
    A measure pooled over drives uses, for each pair, the mean of D over the drives.
  - A/B = exp(mean D) over all kept pairs: the geometric mean of the matched error ratios.
  - Interval: 95 % percentile bootstrap over seeds (10^4 draws of the seeds with replacement; a draw takes the mean of D
    over all pairs of the drawn seeds).
  - Two one-sided sign-flip tests of A/B > 1:
      over runs:       10^5 random sign patterns of D; p = (1 + c) / (1 + 10^5), where c counts the patterns whose mean
                       is at least the observed mean;
      over seed means: D averaged per seed across strata; for up to 20 seeds all 2^n sign patterns are counted
                       (p = fraction of patterns, the observed one included, whose mean is at least the observed mean),
                       for more seeds 10^5 random patterns as above.
  - Holm's step-down correction within each planned family (docs/planned_tests.csv), separately for the two tests.
    A comparison with no stratum left (both models failed in every stratum) enters its family with p = 1.
  - A dagger marks a comparison whose two Holm-adjusted p-values are both below 0.05.
  - An ordering over conditions (one planned test, H4) has no p-value and no correction: it holds when the estimates
    rise strictly from condition to condition and the last condition's interval lies above the first's.

Inputs are nested dicts {stratum: {model: {seed: error}}}; scripts/letter_data.py builds them from the evaluation outputs.
"""
from __future__ import annotations

import math
import zlib
from collections import defaultdict

import numpy as np

FAIL = 0.3  # a median error of 0.3 or more counts as failure
MIN_RUNS = 5  # fewer runs of a model in a stratum leave the stratum out
LEAVE_OUT = "both"  # leave a stratum out when both models failed; "any": when either did (the plans of H1-H6)
N_FLIPS = 100_000
N_BOOT = 10_000
EXACT_MAX_SEEDS = 20  # exact enumeration over 2^n sign patterns up to this many seeds
ALPHA = 0.05


def kept(cell: dict, a: str, b: str) -> bool:
    """Whether a stratum {model: {seed: error}} enters the comparison of a and b."""
    da, db = cell.get(a, {}), cell.get(b, {})
    if len(da) < MIN_RUNS or len(db) < MIN_RUNS:
        return False
    failed = (np.median(list(da.values())) >= FAIL, np.median(list(db.values())) >= FAIL)
    return not (any(failed) if LEAVE_OUT == "any" else all(failed))


def log_ratios(groups: dict, a: str, b: str) -> dict:
    """{stratum: {seed: log(e_a / e_b)}} over the kept strata and the seeds both models have."""
    out = {}
    for key, cell in groups.items():
        if kept(cell, a, b):
            out[key] = {s: math.log(cell[a][s]) - math.log(cell[b][s]) for s in sorted(set(cell[a]) & set(cell[b]))}
    return out


def ratio_of_ratios(groups1: dict, groups2: dict, a: str, b: str) -> dict:
    """(a/b in groups1) / (a/b in groups2): strata kept in both conditions, pairs present in both."""
    d1, d2 = log_ratios(groups1, a, b), log_ratios(groups2, a, b)
    return {k: {s: d1[k][s] - d2[k][s] for s in d1[k] if s in d2[k]} for k in d1 if k in d2}


def interaction(groups: dict, a1: str, b1: str, a2: str, b2: str) -> dict:
    """(a1/b1) / (a2/b2) in one condition: strata kept for both pairs, seeds that both pairs have."""
    d1, d2 = log_ratios(groups, a1, b1), log_ratios(groups, a2, b2)
    return {k: {s: d1[k][s] - d2[k][s] for s in d1[k] if s in d2[k]} for k in d1 if k in d2}


def pooled(parts: list[dict]) -> dict:
    """Mean of D over several measures (drives) for each pair that appears in at least one of them."""
    acc = defaultdict(list)
    for d in parts:
        for key, x in d.items():
            for s, v in x.items():
                acc[(key, s)].append(v)
    out = defaultdict(dict)
    for (key, s), v in acc.items():
        out[key][s] = float(np.mean(v))
    return dict(out)


def _flatten(d: dict):
    """D over all pairs, and D averaged per seed across strata (seeds in increasing order)."""
    D = np.array([v for x in d.values() for v in x.values()], float)
    per = defaultdict(list)
    for x in d.values():
        for s, v in x.items():
            per[s].append(v)
    seeds = sorted(per)
    return D, seeds, per


def p_random_flips(x, rng, n_flips: int = N_FLIPS, batch: int = 1000) -> float:
    """One-sided sign-flip test of mean(x) > 0 with random sign patterns; p = (1 + c) / (1 + n_flips)."""
    x = np.asarray(x, float)
    obs, cnt = x.mean(), 0
    for start in range(0, n_flips, batch):
        m = min(batch, n_flips - start)
        signs = rng.integers(0, 2, size=(m, len(x))) * 2.0 - 1.0
        cnt += int(((signs * x).mean(1) >= obs - 1e-12).sum())
    return (1 + cnt) / (1 + n_flips)


def p_exact_flips(x) -> float:
    """One-sided sign-flip test of mean(x) > 0 over all 2^n sign patterns, counted in blocks of at most 2^16 patterns."""
    x = np.asarray(x, float)
    n, obs, cnt = len(x), x.mean(), 0
    bits = np.arange(n)
    block = 1 << min(n, 16)
    for start in range(0, 1 << n, block):
        idx = np.arange(start, min(start + block, 1 << n))
        signs = 1.0 - 2.0 * ((idx[:, None] >> bits) & 1)
        cnt += int(np.sum(signs @ x / n >= obs - 1e-12))
    return cnt / (1 << n)


def p_seed_means(seed_means, rng, n_flips: int = N_FLIPS, exact_max: int = EXACT_MAX_SEEDS) -> float:
    """The test over seed means: exact for up to exact_max seeds, random sign patterns otherwise."""
    if len(seed_means) <= exact_max:
        return p_exact_flips(seed_means)
    return p_random_flips(seed_means, rng, n_flips)


def seed_interval(d: dict, rng, n_boot: int = N_BOOT) -> tuple[float, float]:
    """95 % percentile bootstrap interval of A/B with seeds resampled."""
    _, seeds, per = _flatten(d)
    S = np.array([sum(per[s]) for s in seeds])
    n = np.array([len(per[s]) for s in seeds], float)
    idx = rng.integers(0, len(seeds), (n_boot, len(seeds)))
    boot = S[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return math.exp(lo), math.exp(hi)


def rng_for(name: str):
    """A fixed random stream for each comparison, so that every value can be reproduced."""
    return np.random.default_rng(zlib.crc32(name.encode()))


def estimate(d: dict, name: str = "") -> dict | None:
    """Estimate and seed interval for {stratum: {seed: D}}, without tests (the same values as compare()); None if no
    pair is left."""
    if not any(d.values()):
        return None
    D, seeds, _ = _flatten(d)
    lo, hi = seed_interval(d, rng_for(name + "/interval"))
    return {"ratio": math.exp(D.mean()), "lo": lo, "hi": hi, "runs": int(len(D)), "seeds": len(seeds)}


def ordered(parts: list) -> bool:
    """An ordering over conditions (H4 of the list): the estimates rise strictly from each condition to the next, and
    the interval of the last condition lies above that of the first."""
    if not parts or any(p is None for p in parts):
        return False
    return all(x["ratio"] < y["ratio"] for x, y in zip(parts, parts[1:])) and parts[-1]["lo"] > parts[0]["hi"]


def compare(d: dict, name: str = "") -> dict | None:
    """Estimate, seed interval, and both raw p-values for {stratum: {seed: D}}; None if no pair is left."""
    if not any(d.values()):
        return None
    D, seeds, per = _flatten(d)
    x = np.array([np.mean(per[s]) for s in seeds])
    lo, hi = seed_interval(d, rng_for(name + "/interval"))
    return {"ratio": math.exp(D.mean()), "lo": lo, "hi": hi, "runs": int(len(D)), "seeds": len(seeds),
            "strata": [str(k) for k in d if d[k]], "b_better": int((D > 0).sum()),
            "p_runs": p_random_flips(D, rng_for(name + "/runs")), "p_seeds": p_seed_means(x, rng_for(name + "/seeds"))}


def holm(ps: list[float]) -> list[float]:
    """Holm's step-down adjustment (running maximum, capped at 1)."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def family(results: dict) -> dict:
    """Holm within one family for {test: compare() or None}; adds the adjusted p-values and the dagger."""
    keys = list(results)
    p_runs = [results[k]["p_runs"] if results[k] else 1.0 for k in keys]
    p_seeds = [results[k]["p_seeds"] if results[k] else 1.0 for k in keys]
    out = {}
    for k, hr, hs in zip(keys, holm(p_runs), holm(p_seeds)):
        r = dict(results[k] or {"ratio": None, "p_runs": 1.0, "p_seeds": 1.0})
        r.update(p_runs_holm=hr, p_seeds_holm=hs, dagger=bool(hr < ALPHA and hs < ALPHA))
        out[k] = r
    return out
