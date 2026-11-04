"""Checks of the letter's statistics (scripts/letter_stats.py) and of the planned-test list, on synthetic data."""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import letter_data as LD  # noqa: E402
import letter_stats as LS  # noqa: E402
import letter_tables as LT  # noqa: E402


def brute_force_p(x):
    """Fraction of all 2^n sign patterns whose mean is at least the observed mean (itertools, no blocks)."""
    x = np.asarray(x, float)
    obs = x.mean()
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=len(x))))
    return float(((signs * x).mean(1) >= obs - 1e-12).mean())


@pytest.mark.parametrize("n,seed", [(1, 0), (5, 1), (8, 2), (10, 3), (10, 4), (12, 5)])
def test_exact_seed_level_p_equals_brute_force(n, seed):
    x = np.random.default_rng(seed).normal(0.2, 1.0, n)
    assert LS.p_exact_flips(x) == pytest.approx(brute_force_p(x), abs=1e-15)


def test_exact_p_with_ties_and_zeros():
    x = np.array([0.5, -0.5, 0.0, 0.5, 0.25, 0.0, -0.25, 0.5])
    assert LS.p_exact_flips(x) == pytest.approx(brute_force_p(x), abs=1e-15)


def test_exact_p_over_several_blocks():
    """17 seeds take two blocks of 2^16 patterns; the count must equal the full enumeration."""
    x = np.random.default_rng(7).normal(0.1, 1.0, 17)
    assert LS.p_exact_flips(x) == pytest.approx(brute_force_p(x), abs=1e-15)


def test_exact_p_twenty_seeds_all_positive():
    """Only the observed pattern reaches the observed mean: p = 2^-20 (16 blocks, never 2^20 patterns at once)."""
    assert LS.p_exact_flips(np.full(20, 0.3)) == 2.0 ** -20


def test_ten_positive_seeds_give_the_minimum_p():
    assert LS.p_exact_flips(np.linspace(0.1, 1.0, 10)) == 1 / 1024


def test_seed_level_switches_to_random_flips_above_twenty_seeds():
    x = np.random.default_rng(3).normal(0.3, 1.0, 30)
    p = LS.p_seed_means(x, np.random.default_rng(11))
    assert p == LS.p_random_flips(x, np.random.default_rng(11))
    assert LS.p_seed_means(x[:20], None) == LS.p_exact_flips(x[:20])


def test_random_flips_p_formula():
    """All D positive and many runs: no random pattern reaches the observed mean, so p = 1 / (1 + N)."""
    p = LS.p_random_flips(np.full(40, 0.7), np.random.default_rng(0), n_flips=10_000)
    assert p == pytest.approx(1 / 10_001)
    # a symmetric sample (mean 0): about half of the patterns reach it
    x = np.concatenate([np.ones(20), -np.ones(20)])
    assert 0.5 < LS.p_random_flips(x, np.random.default_rng(1), n_flips=20_000) < 0.65


def test_random_flips_agree_with_exact_p():
    x = np.random.default_rng(5).normal(0.15, 1.0, 10)
    assert LS.p_random_flips(x, np.random.default_rng(2)) == pytest.approx(LS.p_exact_flips(x), abs=0.005)


def test_holm_known_values():
    assert LS.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert LS.holm([0.5, 0.6]) == [1.0, 1.0]
    assert LS.holm([0.02]) == [0.02]
    # six seed-level p of one family: 5 x 10/1024 and 6 x 2/1024 for the two smallest
    adj = LS.holm([0.33, 0.50, 10 / 1024, 2 / 1024, 0.90, 0.92])
    assert adj[2] == pytest.approx(5 * 10 / 1024) and adj[3] == pytest.approx(6 * 2 / 1024)
    assert adj[0] == 1.0 and adj[4] == 1.0


def test_holm_ties_do_not_depend_on_order():
    assert LS.holm([0.01, 0.01, 0.2]) == pytest.approx([0.03, 0.03, 0.2])


def _groups(errors_a, errors_b, key=("small", "1e2")):
    return {key: {"A": dict(enumerate(errors_a)), "B": dict(enumerate(errors_b))}}


def test_strata_are_left_out_only_when_both_models_failed():
    both_failed = _groups([0.5] * 10, [0.4] * 10)
    one_failed = _groups([0.5] * 10, [0.1] * 10)
    assert LS.log_ratios(both_failed, "A", "B") == {}
    d = LS.log_ratios(one_failed, "A", "B")
    assert d[("small", "1e2")][0] == pytest.approx(math.log(5.0))


def test_any_failed_model_leaves_the_stratum_out_with_the_other_rule(monkeypatch):
    one_failed = _groups([0.5] * 10, [0.1] * 10)
    monkeypatch.setattr(LS, "LEAVE_OUT", "any")
    assert LS.log_ratios(one_failed, "A", "B") == {}
    assert LS.log_ratios(_groups([0.2] * 10, [0.1] * 10), "A", "B") != {}


def test_strata_with_fewer_than_five_runs_are_left_out():
    g = _groups([0.1] * 4, [0.2] * 10)
    assert LS.log_ratios(g, "A", "B") == {}


def test_pairs_are_matched_by_seed():
    g = {("small", "1e2"): {"A": {0: 0.2, 1: 0.4, 2: 0.2, 3: 0.2, 4: 0.2, 9: 1.0},
                            "B": {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1}}}
    d = LS.log_ratios(g, "A", "B")[("small", "1e2")]
    assert sorted(d) == [0, 1, 2, 3, 4]
    assert d[1] == pytest.approx(math.log(4.0))


def test_ratio_of_ratios_keeps_strata_and_pairs_present_in_both():
    k1, k2 = ("small", "1e2"), ("medium", "1e2")
    g1 = {k1: {"A": {s: 0.4 for s in range(10)}, "B": {s: 0.1 for s in range(10)}},
          k2: {"A": {s: 0.2 for s in range(10)}, "B": {s: 0.1 for s in range(10)}}}
    g2 = {k1: {"A": {s: 0.2 for s in range(8)}, "B": {s: 0.1 for s in range(8)}},
          k2: {"A": {s: 0.5 for s in range(10)}, "B": {s: 0.5 for s in range(10)}}}  # k2 failed in condition 2
    d = LS.ratio_of_ratios(g1, g2, "A", "B")
    assert list(d) == [k1] and sorted(d[k1]) == list(range(8))
    assert d[k1][0] == pytest.approx(math.log(4.0) - math.log(2.0))


def test_pooled_averages_over_drives():
    d1 = {"1e2": {0: 0.2, 1: 0.4}}
    d2 = {"1e2": {0: 0.4}, "3e2": {0: 1.0}}
    assert LS.pooled([d1, d2]) == {"1e2": {0: pytest.approx(0.3), 1: pytest.approx(0.4)}, "3e2": {0: 1.0}}


def test_interaction_keeps_strata_kept_for_both_pairs():
    k1, k2 = ("small", "1e2"), ("medium", "1e2")
    g = {k1: {"A1": {s: 0.4 for s in range(10)}, "B1": {s: 0.1 for s in range(10)},
              "A2": {s: 0.2 for s in range(8)}, "B2": {s: 0.1 for s in range(10)}},
         k2: {"A1": {s: 0.2 for s in range(10)}, "B1": {s: 0.1 for s in range(10)},
              "A2": {s: 0.5 for s in range(10)}, "B2": {s: 0.5 for s in range(10)}}}  # pair 2 failed in k2
    d = LS.interaction(g, "A1", "B1", "A2", "B2")
    assert list(d) == [k1] and sorted(d[k1]) == list(range(8))
    assert d[k1][0] == pytest.approx(math.log(4.0) - math.log(2.0))


def test_ordering_needs_rising_estimates_and_apart_intervals():
    part = lambda r, lo, hi: {"ratio": r, "lo": lo, "hi": hi}  # noqa: E731
    assert LS.ordered([part(1.0, 0.9, 1.1), part(1.2, 1.0, 1.4), part(4.0, 3.0, 5.0)])
    assert not LS.ordered([part(1.2, 0.9, 1.4), part(1.1, 1.0, 1.3), part(4.0, 3.0, 5.0)])  # not rising
    assert not LS.ordered([part(1.0, 0.9, 3.5), part(1.2, 1.0, 1.4), part(4.0, 3.0, 5.0)])  # intervals overlap
    assert not LS.ordered([part(1.0, 0.9, 1.1), None])


class _Fixed:
    """Stands in for letter_tables.Data with fixed groups per source."""

    def __init__(self, groups):
        self.groups = groups

    def get(self, source, seeds, sizes=LD.SIZES):
        return self.groups[source]


def test_sources_and_models_of_the_plan_are_parsed():
    k = ("medium", "1e2")
    def g(a, b):  # noqa: E306
        return {k: {"A": {s: a for s in range(6)}, "B": {s: b for s in range(6)}}}
    data = _Fixed({"x": g(0.4, 0.1), "y": g(0.2, 0.1), "z": g(0.3, 0.1), "w": g(0.2, 0.2)})
    seeds = range(6)
    # 'X / Y + Z / W': the mean over the two ratios of ratios, pair by pair
    d = LT.log_ratios_for(data, "x / y + z / w", "A", "B", seeds)
    assert d[k][0] == pytest.approx(0.5 * (math.log(2.0) + math.log(3.0)))
    # the existing forms give what they gave before
    assert LT.log_ratios_for(data, "x / y", "A", "B", seeds) == LS.ratio_of_ratios(data.get("x", seeds), data.get("y", seeds), "A", "B")
    assert LT.log_ratios_for(data, "x + z", "A", "B", seeds) == LS.pooled([LS.log_ratios(data.get(s, seeds), "A", "B") for s in "xz"])
    # models 'A1/B1' and 'A2/B2' in one source
    data = _Fixed({"x": {k: {"A1": {s: 0.4 for s in range(6)}, "B1": {s: 0.1 for s in range(6)},
                             "A2": {s: 0.2 for s in range(6)}, "B2": {s: 0.1 for s in range(6)}}}})
    d = LT.log_ratios_for(data, "x", "A1/B1", "A2/B2", seeds)
    assert d[k][3] == pytest.approx(math.log(2.0))


def test_compare_ratio_is_the_geometric_mean_of_matched_ratios():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.05, 0.2, 10)
    b = rng.uniform(0.05, 0.2, 10)
    r = LS.compare(LS.log_ratios(_groups(a, b), "A", "B"), "unit")
    assert r["ratio"] == pytest.approx(math.exp(np.mean(np.log(a / b))))
    assert r["lo"] <= r["ratio"] <= r["hi"] or abs(math.log(r["ratio"])) < 1e-9
    assert r["runs"] == 10 and r["seeds"] == 10
    assert r["p_seeds"] == pytest.approx(brute_force_p(np.log(a / b)))


def test_seed_interval_resamples_seeds_not_runs():
    """With D constant within each seed, the seed bootstrap equals a bootstrap of the seed means."""
    d = {("small", "1e2"): {0: 0.1, 1: 0.3, 2: 0.5}, ("medium", "1e2"): {0: 0.1, 1: 0.3, 2: 0.5}}
    lo, hi = LS.seed_interval(d, np.random.default_rng(0), n_boot=20_000)
    assert math.exp(0.1) <= lo < hi <= math.exp(0.5)
    const = {("small", "1e2"): {s: 0.2 for s in range(10)}}
    lo, hi = LS.seed_interval(const, np.random.default_rng(0))
    assert lo == pytest.approx(math.exp(0.2)) and hi == pytest.approx(math.exp(0.2))


def test_family_counts_untestable_comparisons_with_p_one():
    good = {"p_runs": 0.01, "p_seeds": 0.015, "ratio": 2.0}
    res = LS.family({"T1": good, "T2": None, "T3": dict(good)})
    assert res["T2"]["ratio"] is None and not res["T2"]["dagger"]
    assert res["T2"]["p_runs_holm"] == 1.0
    # three tests: 3 x 0.01 for the first, max(0.03, 2 x 0.01) for the second
    assert res["T1"]["p_runs_holm"] == pytest.approx(0.03) and res["T3"]["p_runs_holm"] == pytest.approx(0.03)
    assert res["T1"]["p_seeds_holm"] == pytest.approx(0.045)
    assert res["T1"]["dagger"] and res["T3"]["dagger"]
    res = LS.family({"T1": {"p_runs": 0.02, "p_seeds": 0.01}, "T2": None, "T3": {"p_runs": 0.02, "p_seeds": 0.01}})
    assert res["T1"]["p_runs_holm"] == pytest.approx(0.06) and not res["T1"]["dagger"]


def test_dagger_needs_both_tests():
    res = LS.family({"A": {"p_runs": 1e-5, "p_seeds": 0.06, "ratio": 3.0}})
    assert not res["A"]["dagger"]
    res = LS.family({"A": {"p_runs": 1e-5, "p_seeds": 0.049, "ratio": 3.0}})
    assert res["A"]["dagger"]


def test_compare_is_reproducible():
    g = _groups(np.linspace(0.1, 0.3, 10), np.linspace(0.12, 0.2, 10))
    d = LS.log_ratios(g, "A", "B")
    assert LS.compare(d, "x/y") == LS.compare(d, "x/y")


# ---- the planned-test list

PLAN = LT.read_plan()


def _sources(text):
    return [s for part in text.replace(" < ", " + ").split(" + ") for s in part.split(" / ")]


def _models(r):
    return {m for x in (r["model_a"], r["model_b"]) for m in x.split("/")}


def _known(source):
    return source in LD.FREE or source in LD.DRIVES or source in LD.WAVEFORMS or source in LD.OTHER


def test_plan_has_unique_tests_and_known_sources():
    seen = set()
    for fam, rows in PLAN.items():
        for r in rows:
            key = (fam, r["test"])
            assert key not in seen
            seen.add(key)
            for s in _sources(r["source"]):
                assert _known(s), s
            LD.seed_range(r["seeds"])
            assert set(r["sizes"].split()) <= set(LD.SIZES)


def test_plan_models_exist_in_their_sources():
    for rows in PLAN.values():
        for r in rows:
            for s in _sources(r["source"]):
                if s in LD.FREE:
                    assert _models(r) <= set(LD.FREE[s][1]), (r["family"], r["test"])
                elif s in LD.VOLTAGE_PAIR:
                    assert r["model_a"].endswith("(V-I)") and r["model_b"].endswith("(q-phi)")


def test_family_sizes():
    sizes = {fam: len(rows) for fam, rows in PLAN.items()}
    assert len(sizes) == 28 and sum(sizes.values()) == 128
    assert min(sizes.values()) == 1 and max(sizes.values()) == 12
    assert sum(r["letter"] == "repository only" for rows in PLAN.values() for r in rows) == 43


def test_tables_point_to_planned_tests():
    for _, _, fam, test in LT.FIG3:
        assert any(r["test"] == test for r in PLAN[fam])
    for _, _, cells in LT.T4:
        for _, t, source, a in cells:
            for s in _sources(source):
                assert _known(s)
            if t:
                row = next(r for r in PLAN[t[0]] if r["test"] == t[1])
                assert (row["model_a"], row["model_b"], row["source"]) == (a, "LF", source)


# ---- the saved per-run values

def test_saved_runs_read_back_as_loaded(tmp_path):
    rows = [(("small", "1e2"), "LF", 0, 0.1), (("small", "1e2"), "Res", 0, 1e3), (("medium", "3e2"), "LF", 1, 1 / 3),
            (("small", "1e2"), "LF", 1, 0.125)]
    path = tmp_path / "runs.csv"
    LD.write_runs(path, [("circuit", *r) for r in rows] + [("pulses", *r) for r in rows])
    saved = LD.read_runs(path)
    assert LD.from_runs(saved, "circuit", range(2)) == LD.nest(rows)
    assert list(LD.from_runs(saved, "circuit", [1])) == [("medium", "3e2"), ("small", "1e2")]  # order of the file
    assert LD.from_runs(saved, "circuit", range(2), ("medium",)) == LD.nest(rows[2:3])
    assert LD.from_runs(saved, "pulses", range(2), ("medium",)) == LD.nest(rows)  # sizes select free runs only
    assert LD.from_runs(saved, "line", range(2)) == {}
    with pytest.raises(KeyError):
        LD.from_runs(saved, "no such source", range(2))


def test_saved_runs_cover_the_planned_tests():
    saved = LD.read_runs()
    for rows in PLAN.values():
        for r in rows:
            for s in _sources(r["source"]):
                g = LD.from_runs(saved, s, LD.seed_range(r["seeds"]), tuple(r["sizes"].split()))
                models = {m for cell in g.values() for m in cell}
                assert _models(r) <= models, (r["family"], r["test"], s)
    assert set(LD.read_classical()) == set(LT.T3_SOURCES)


def test_saved_table4_matches_the_saved_output(capsys):
    LT.print_table4(LT.table4(LT.SavedData()))
    text = (LD.REPO / "results" / "letter_tables.md").read_text(encoding="utf-8")
    assert capsys.readouterr().out.strip() in text
