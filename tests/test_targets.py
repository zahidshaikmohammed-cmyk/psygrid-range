"""PHASE 4 - targets: brute-force equality, session boundaries, coverage."""
import numpy as np
import pandas as pd

from engine.features import to_minute_grid
from engine.targets import compute_targets


def _two_days(market):
    df = market["CIPLA"]
    days = df.index.normalize().unique()[:2]
    return df[df.index.normalize().isin(days)]


def test_matches_brute_force(cfg, market):
    df = _two_days(market)
    g = to_minute_grid(df, cfg)
    sigma = pd.Series(0.001, index=g.index)
    t = compute_targets(g, sigma, cfg)
    rng = np.random.default_rng(0)
    checked = 0
    for i in rng.choice(np.where(t["target_valid"])[0], 50, replace=False):
        ts = g.index[i]
        h = int(t["h_eff"].iloc[i])
        win = df[(df.index > ts) & (df.index <= ts + pd.Timedelta(minutes=h))]
        price = g["close"].iloc[i]
        assert np.isclose(t["future_high"].iloc[i], win["high"].max())
        assert np.isclose(t["future_low"].iloc[i], win["low"].min())
        assert np.isclose(t["upper_excursion"].iloc[i], (win["high"].max() - price) / price)
        assert np.isclose(t["lower_excursion"].iloc[i], (price - win["low"].min()) / price)
        assert np.isclose(t["y_up"].iloc[i], t["upper_excursion"].iloc[i] / (0.001 * np.sqrt(h)))
        checked += 1
    assert checked == 50


def test_window_never_crosses_the_close(cfg, market):
    g = to_minute_grid(_two_days(market), cfg)
    t = compute_targets(g, pd.Series(0.001, index=g.index), cfg)
    assert (t["target_end"].dt.strftime("%H:%M") <= "15:29").all()
    at_1500 = t[g.index.strftime("%H:%M") == "15:00"].iloc[0]
    assert at_1500["h_eff"] == 29
    late = t[g.index.strftime("%H:%M") == "15:20"]
    assert not late["target_valid"].any()  # < min_effective_minutes left
    early = t[g.index.strftime("%H:%M") == "10:00"].iloc[0]
    assert early["h_eff"] == 60 and early["target_valid"]


def test_insufficient_window_coverage_invalidates(cfg, market):
    df = _two_days(market)
    day0 = df[df.index.normalize() == df.index[0].normalize()]
    holed = df.drop(day0.index[31:61])  # 30-minute hole after 09:45
    g = to_minute_grid(holed, cfg)
    t = compute_targets(g, pd.Series(0.001, index=g.index), cfg)
    assert not t.loc[day0.index[30], "target_valid"]


def test_partial_session_labels_only_when_observable(cfg, market):
    df = _two_days(market)
    live = df[df.index <= df.index[0] + pd.Timedelta(minutes=100)]  # session in progress
    g = to_minute_grid(live, cfg)
    t = compute_targets(g, pd.Series(0.001, index=g.index), cfg)
    assert t["target_valid"].iloc[:40].all() and not t["target_valid"].iloc[41:].any()
