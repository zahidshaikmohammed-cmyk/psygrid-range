"""PHASE 4 - Target generation.

For every minute *t* (current price = close of candle *t*):

    future_high = max(high[t+1 .. t+H_eff])
    future_low  = min(low [t+1 .. t+H_eff])
    upper_excursion = (future_high - price) / price
    lower_excursion = (price - future_low) / price

``H_eff = min(H, minutes left in the session)``: the window never crosses the
close, so no overnight information leaks into a label.  A label is only
produced when at least ``min_window_coverage`` of the window's minutes have
genuine candles and ``H_eff >= min_effective_minutes``.

For pooled learning the excursions are also expressed in volatility units::

    y = excursion / (sigma_t * sqrt(H_eff))

where ``sigma_t`` is the causal per-minute volatility estimate from the
feature engine.  Quantiles are learned directly on ``y``; no point forecast is
ever widened into a range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import EngineConfig


def _forward_window(arr: np.ndarray, H: int, fn) -> tuple[np.ndarray, np.ndarray]:
    """For each i, aggregate arr[i+1 : i+1+H] (NaN padded) -> (agg, count)."""
    n = len(arr)
    padded = np.concatenate([arr[1:], np.full(H, np.nan)])
    win = sliding_window_view(padded, H)[:n]
    cnt = np.sum(~np.isnan(win), axis=1)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            agg = fn(win, axis=1)
    return agg, cnt


def compute_targets(grid: pd.DataFrame, sigma: pd.Series, cfg: EngineConfig) -> pd.DataFrame:
    H = cfg.horizon
    hcfg = cfg["horizon"]
    n_min = cfg.session_minutes
    out = pd.DataFrame(index=grid.index)
    fut_hi = np.full(len(grid), np.nan)
    fut_lo = np.full(len(grid), np.nan)
    count = np.zeros(len(grid))
    for _, pos in grid.groupby("session", sort=False).indices.items():
        pos = np.sort(pos)
        a, cnt = _forward_window(grid["high"].to_numpy(float)[pos], H, np.nanmax)
        b, _ = _forward_window(grid["low"].to_numpy(float)[pos], H, np.nanmin)
        fut_hi[pos], fut_lo[pos], count[pos] = a, b, cnt
    mos = grid["mos"].to_numpy()
    h_eff = np.minimum(H, n_min - 1 - mos)
    # the window must be fully observable: the session grid must extend to t+H_eff
    last_mos = grid.groupby("session")["mos"].transform("max").to_numpy()
    observable = (mos + h_eff) <= last_mos
    coverage = np.where(h_eff > 0, count / np.maximum(h_eff, 1), 0)
    valid = (
        grid["present"].to_numpy()
        & observable
        & (h_eff >= int(hcfg["min_effective_minutes"]))
        & (coverage >= float(hcfg["min_window_coverage"]))
    )
    price = grid["close"].to_numpy(float)
    up = (fut_hi - price) / price
    dn = (price - fut_lo) / price
    if cfg["target"].get("scaling", "volatility") == "none":
        scale = np.ones(len(grid))
    else:
        scale = sigma.to_numpy(float) * np.sqrt(np.maximum(h_eff, 1))
    out["h_eff"] = h_eff
    out["window_coverage"] = coverage
    out["future_high"] = np.where(valid, fut_hi, np.nan)
    out["future_low"] = np.where(valid, fut_lo, np.nan)
    out["upper_excursion"] = np.where(valid, up, np.nan)
    out["lower_excursion"] = np.where(valid, dn, np.nan)
    out["target_scale"] = scale
    out["y_up"] = np.where(valid, up / scale, np.nan)
    out["y_lo"] = np.where(valid, dn / scale, np.nan)
    out["target_end"] = grid.index + pd.to_timedelta(h_eff, unit="min")
    out["target_valid"] = valid
    return out
