"""PHASE 3 - feature engine: causality, gaps, known values, context."""
import numpy as np
import pandas as pd
import pytest

from engine.dataset import build_symbol_frame
from engine.features import (
    CONTEXT_FEATURES,
    build_market_context,
    compute_features,
    intraday_log_returns,
    tod_bucket_code,
    to_minute_grid,
)


def _hist(market, sym, n_days):
    df = market[sym]
    days = df.index.normalize().unique()[:n_days]
    return df[df.index.normalize().isin(days)]


def test_grid_keeps_gaps_as_nan(cfg, market):
    df = _hist(market, "CIPLA", 2)
    holed = df.drop(df.index[[5, 6, 7]])
    g = to_minute_grid(holed, cfg)
    assert (~g["present"]).sum() >= 3
    assert g.loc[df.index[6], "close"] != g.loc[df.index[6], "close"]  # NaN, not filled
    assert g["mos"].min() == 0


def test_no_lookahead(cfg, market):
    """Features at time t must not change when data after t is removed or altered."""
    df = _hist(market, "CIPLA", 6)
    full = compute_features(to_minute_grid(df, cfg), cfg, symbol="CIPLA")
    cut_ts = df.index[-200]
    trunc = df[df.index <= cut_ts]
    part = compute_features(to_minute_grid(trunc, cfg), cfg, symbol="CIPLA")
    a = full.loc[:cut_ts].tail(300)
    b = part.loc[:cut_ts].tail(300)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)

    # scramble the future instead of truncating
    shocked = df.copy()
    shocked.loc[shocked.index > cut_ts, ["open", "high", "low", "close"]] *= 1.5
    shocked.loc[shocked.index > cut_ts, "volume"] *= 10
    c = compute_features(to_minute_grid(shocked, cfg), cfg, symbol="CIPLA").loc[:cut_ts].tail(300)
    pd.testing.assert_frame_equal(a, c, check_exact=False, rtol=1e-9, atol=1e-12)


def test_known_values(cfg):
    ts = pd.date_range(pd.Timestamp("2026-03-02 09:15", tz="Asia/Kolkata"), periods=90, freq="min")
    close = 100 * np.exp(np.cumsum(np.full(90, 0.001)))
    df = pd.DataFrame({"open": np.r_[100, close[:-1]], "close": close, "volume": 100.0}, index=ts)
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    g = to_minute_grid(df, cfg)
    lr = intraday_log_returns(g)
    assert np.allclose(lr.to_numpy(), 0.001)
    f = compute_features(g, cfg, symbol="CIPLA")
    # z_ret_k = log return over k / (sigma * sqrt(k))
    z5 = f["z_ret_5"].iloc[-1]
    assert np.isclose(z5, 0.005 / (f["sigma"].iloc[-1] * np.sqrt(5)))
    assert f["run_length"].iloc[-1] == 90  # 90 consecutive bullish candles
    assert f["rsi_14"].iloc[-1] > 99
    assert f["struct_score"].iloc[-1] == 2
    # VWAP with constant volume: typical-price average below the latest close
    assert f["d_vwap"].iloc[-1] > 0
    assert list(tod_bucket_code(np.array([0, 29, 30, 104, 105, 224, 225, 314, 315, 374]))) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_time_of_day_distinguishes_states(cfg, market):
    f = compute_features(to_minute_grid(_hist(market, "CIPLA", 3), cfg), cfg, symbol="CIPLA")
    a = f.loc[f.index.strftime("%H:%M") == "09:20"].iloc[-1]
    b = f.loc[f.index.strftime("%H:%M") == "13:45"].iloc[-1]
    assert a["tod_bucket"] != b["tod_bucket"] and a["mos"] != b["mos"] and a["mtc"] != b["mtc"]


def test_context_relative_strength(cfg, market):
    syms = list(market)
    grids = {s: to_minute_grid(_hist(market, s, 4), cfg) for s in syms}
    ctx = build_market_context({s: intraday_log_returns(g) for s, g in grids.items()}, cfg)
    f = compute_features(grids["CIPLA"], cfg, symbol="CIPLA", context=ctx)
    assert f["ctx_frac"].iloc[-1] == pytest.approx(len(syms) / 50)
    assert f["rs_30"].notna().mean() > 0.8
    # a stock holding up while the market falls must have positive relative strength
    lrs = {s: intraday_log_returns(g) for s, g in grids.items()}
    falling = {s: (v - 0.002 if s != "CIPLA" else v) for s, v in lrs.items()}
    ctx2 = build_market_context(falling, cfg)
    f2 = compute_features(grids["CIPLA"], cfg, symbol="CIPLA", context=ctx2)
    assert (f2["rs_30"] - f["rs_30"]).dropna().mean() > 0
    assert (f2["m_z_ret_30"]).dropna().mean() < f["m_z_ret_30"].dropna().mean()


def test_no_context_gives_nan_context_features(cfg, market):
    f = compute_features(to_minute_grid(_hist(market, "TCS", 2), cfg), cfg, symbol="TCS")
    assert f[CONTEXT_FEATURES].isna().all().all()


def test_symbol_frame_has_regimes_and_targets(cfg, market):
    fr = build_symbol_frame(_hist(market, "INFY", 4), cfg, "INFY")
    for col in ("regime", "volatility_state", "structure", "relative_strength", "y_up", "y_lo", "target_valid"):
        assert col in fr.columns
