"""Skill is measured against the BEST baseline (by mean loss, never per row)."""
import numpy as np
import pandas as pd

from engine.evaluation import BASE_PREFIX, has_edge, skill_stats, summarize
from engine.training import accuracy_verdict, history_status, model_status


def test_skill_uses_best_baseline_by_mean():
    g = pd.DataFrame({"pinball_model": [1.0, 1.0], BASE_PREFIX + "a": [0.5, 2.0], BASE_PREFIX + "b": [1.2, 1.2]})
    s = skill_stats(g)
    # means: a=1.25, b=1.20 -> b is the best baseline (a per-row minimum would be an unfair oracle)
    assert s["best_baseline"] == "b" and np.isclose(s["skill"], 1 - 1.0 / 1.2)


def test_skill_tstat_uses_sessions_as_units():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(10):
        for _ in range(50):
            rows.append({"session": f"d{d}", "pinball_model": 1.0, BASE_PREFIX + "b": 1.1 + rng.normal(0, 0.01)})
    assert skill_stats(pd.DataFrame(rows))["skill_tstat"] > 5
    one = pd.DataFrame(rows[:50])
    assert np.isnan(skill_stats(one)["skill_tstat"])  # one session is not evidence


def test_has_edge_rules():
    cc = {"symbol_min_sessions": 5, "min_skill": 0.0, "medium_coverage_tolerance": 0.1, "edge_min_skill_tstat": 2.0}
    ok, _ = has_edge({"n": 10, "sessions": 10, "skill_vs_baseline": 0.03, "skill_tstat": 2.5, "coverage_probable": 0.78}, 0.8, cc)
    assert ok
    noisy, why = has_edge({"n": 10, "sessions": 10, "skill_vs_baseline": 0.03, "skill_tstat": 1.1,
                           "coverage_probable": 0.78}, 0.8, cc)
    assert not noisy and "not statistically distinguishable" in why
    no, why = has_edge({"n": 10, "sessions": 10, "skill_vs_baseline": -0.01, "coverage_probable": 0.8}, 0.8, cc)
    assert not no and "NO DEMONSTRATED EDGE" in why
    few, why = has_edge({"n": 10, "sessions": 2, "skill_vs_baseline": 0.2, "coverage_probable": 0.8}, 0.8, cc)
    assert not few and "sessions" in why


def test_backtest_reports_every_baseline_and_edge_lists(trained_home):
    _, res = trained_home
    s = res.summary
    assert set(s["overall"]["pinball_baselines"]) == {"empirical_tod", "volatility_only", "recent_realized_vol"}
    assert set(s["symbols_with_edge"]) | set(s["symbols_without_edge"]) == set(res.records["symbol"].unique())
    assert s["standard_60m"]["n"] < s["overall"]["n"]
    s2 = summarize(res.records, [0.1, 0.25, 0.5, 0.75, 0.9])
    assert "symbols_with_edge" not in s2


def test_status_never_claims_accuracy_for_synthetic(trained_home):
    from conftest import SMALL

    from engine.config import load_config

    root, _ = trained_home
    cfg = load_config(SMALL, root=root)
    st = history_status(cfg)
    assert st["provenance"]["synthetic"] and st["total_candles"] > 0 and st["stocks_available"] == 8
    ms = model_status(cfg)
    assert ms["exists"] and ms["synthetic"]
    assert accuracy_verdict(ms).startswith("NOT ESTABLISHED")
    assert accuracy_verdict({"exists": False}).startswith("NOT ESTABLISHED")
    real = {**ms, "synthetic": False, "provenance": {"unknown": False},
            "overall": {"n": 100, "sessions": 20, "skill_vs_baseline": -0.02, "skill_tstat": -1.0}}
    assert accuracy_verdict(real).startswith("NO DEMONSTRATED EDGE")
    noisy = {**real, "overall": {"n": 100, "sessions": 20, "skill_vs_baseline": 0.01, "skill_tstat": 0.8}}
    assert accuracy_verdict(noisy).startswith("NO DEMONSTRATED EDGE")
    good = {**real, "overall": {"n": 100, "sessions": 20, "skill_vs_baseline": 0.06, "skill_tstat": 3.1}}
    assert accuracy_verdict(good).startswith("EVALUATED") and "not a guarantee" in accuracy_verdict(good)
