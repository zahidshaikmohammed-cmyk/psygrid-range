"""Regime layer: multi-feature inference on constructed states."""
import numpy as np
import pandas as pd

from engine.features import compute_features, to_minute_grid
from engine.regime import REGIMES, classify


def _series(returns, vol_noise=0.0, seed=0, day="2026-03-02"):
    rng = np.random.default_rng(seed)
    n = len(returns)
    r = np.asarray(returns) + rng.standard_normal(n) * vol_noise
    ts = pd.date_range(pd.Timestamp(f"{day} 09:15", tz="Asia/Kolkata"), periods=n, freq="min")
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100, c[:-1]]
    df = pd.DataFrame({"open": o, "close": c, "volume": 1000 + 20000 * np.abs(r) / (vol_noise + 1e-4)}, index=ts)
    df["high"] = np.maximum(o, c) * (1 + 0.0002)
    df["low"] = np.minimum(o, c) * (1 - 0.0002)
    return df


def _regimes(cfg, df):
    return classify(compute_features(to_minute_grid(df, cfg), cfg, symbol="CIPLA"))


def test_trend_up_and_down(cfg):
    up = _regimes(cfg, _series(np.full(200, 0.0006), vol_noise=0.0004))
    assert (up["regime"].iloc[-60:].isin(["TREND_UP", "BREAKOUT"])).mean() > 0.6
    assert (up["structure"].iloc[-60:] == "HIGHER_HIGHS_HIGHER_LOWS").mean() > 0.5
    dn = _regimes(cfg, _series(np.full(200, -0.0006), vol_noise=0.0004))
    assert (dn["regime"].iloc[-60:].isin(["TREND_DOWN", "BREAKDOWN"])).mean() > 0.6


def test_range_when_no_drift(cfg):
    rg = _regimes(cfg, _series(np.zeros(250), vol_noise=0.0005, seed=3))
    assert (rg["regime"].iloc[-120:] == "RANGE").mean() > 0.5


def test_volatility_expansion_detected(cfg):
    rng = np.random.default_rng(5)
    r = np.r_[rng.standard_normal(200) * 0.0003, rng.standard_normal(40) * 0.003]
    reg = _regimes(cfg, _series(r))
    assert reg["regime"].iloc[-30:].isin(["VOLATILITY_EXPANSION", "BREAKOUT", "BREAKDOWN", "REVERSAL"]).mean() > 0.5


def test_codes_are_consistent(cfg):
    reg = _regimes(cfg, _series(np.zeros(120), vol_noise=0.0005))
    assert set(reg["regime"]).issubset(REGIMES)
    assert (reg["regime_code"] == reg["regime"].map(REGIMES.index)).all()
    assert (reg["relative_strength"] == "UNAVAILABLE").all()  # no market context supplied
