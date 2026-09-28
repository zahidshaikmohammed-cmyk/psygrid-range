"""Synthetic 1-minute market generator - FOR TESTS ONLY.

Produces NIFTY-like sessions with a common market factor, sector factors,
U-shaped intraday volatility, volatility clustering, trend episodes and
volume that co-moves with activity.  It exists to exercise the pipeline
mechanically; results on this data say nothing about real-market accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trading_days(n: int, start: str = "2026-01-05") -> list[pd.Timestamp]:
    return list(pd.bdate_range(start, periods=n))


def generate_market(
    symbols: list[str],
    sectors: dict[str, str],
    n_sessions: int = 40,
    seed: int = 7,
    tz: str = "Asia/Kolkata",
    start: str = "2026-01-05",
    drop_fraction: float = 0.003,
    base_prices: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = trading_days(n_sessions, start)
    n_min = 375
    mos = np.arange(n_min)
    season = 1.0 + 1.6 * np.exp(-mos / 25.0) + 0.5 * np.exp(-(n_min - 1 - mos) / 30.0)
    vol_season = season / season.mean()
    sector_names = sorted(set(sectors[s] for s in symbols))
    params = {
        s: dict(
            beta=rng.uniform(0.6, 1.4),
            vol=rng.uniform(0.0006, 0.0012),
            price=(base_prices or {}).get(s, rng.uniform(300, 4000)),
            vol_base=rng.uniform(2e3, 5e4),
        )
        for s in symbols
    }
    out: dict[str, list[pd.DataFrame]] = {s: [] for s in symbols}
    m_state = 1.0
    for d in days:
        m_state = float(np.clip(0.7 * m_state + 0.3 * rng.lognormal(0, 0.35), 0.5, 2.5))
        m_drift = np.zeros(n_min)
        start_ep = rng.integers(0, n_min - 60)
        m_drift[start_ep : start_ep + rng.integers(30, 120)] = rng.normal(0, 0.00012)
        m_ret = m_drift + rng.standard_normal(n_min) * 0.0004 * m_state * vol_season
        sec_ret = {k: rng.standard_normal(n_min) * 0.0003 * vol_season for k in sector_names}
        idx = pd.date_range(pd.Timestamp(d.date()).tz_localize(tz) + pd.Timedelta(hours=9, minutes=15), periods=n_min, freq="min")
        for s in symbols:
            p = params[s]
            st = float(np.clip(rng.lognormal(0, 0.3), 0.5, 2.5))
            drift = np.zeros(n_min)
            for _ in range(rng.integers(0, 3)):
                a = rng.integers(0, n_min - 30)
                drift[a : a + rng.integers(20, 90)] += rng.normal(0, 0.00025)
            idio = rng.standard_normal(n_min) * p["vol"] * st * vol_season
            r = p["beta"] * m_ret + 0.6 * sec_ret[sectors[s]] + idio + drift
            gap = rng.normal(0, 0.004)
            o = np.empty(n_min)
            c = np.empty(n_min)
            price = p["price"] * np.exp(gap)
            for i in range(n_min):
                o[i] = price
                price = price * np.exp(r[i])
                c[i] = price
            spread = np.abs(rng.standard_normal((2, n_min))) * p["vol"] * st * vol_season * 0.6
            h = np.maximum(o, c) * np.exp(spread[0])
            lo = np.minimum(o, c) * np.exp(-spread[1])
            v = p["vol_base"] * season * (1 + 3 * np.abs(r) / (p["vol"] * vol_season)) * rng.lognormal(0, 0.4, n_min)
            p["price"] = price
            df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": np.round(v)}, index=idx)
            keep = rng.random(n_min) > drop_fraction
            keep[0] = True
            out[s].append(df[keep])
    result = {s: pd.concat(v) for s, v in out.items()}
    for df in result.values():
        df.index.name = "timestamp"
    return result


def to_payload(symbol: str, candles: pd.DataFrame, previous_close: float | None = None, fmt: str = "dict") -> dict:
    """Render candles as a PSYGRID-style endpoint document."""
    recs = []
    for ts, row in candles.iterrows():
        vals = [ts.isoformat(), row["open"], row["high"], row["low"], row["close"], row["volume"]]
        if fmt == "dict":
            recs.append(dict(zip(["timestamp", "open", "high", "low", "close", "volume"], vals)))
        else:
            recs.append(vals)
    return {
        "service": "psygrid",
        "schema_version": "1",
        "status": "ok",
        "symbol": symbol,
        "security_id": "0",
        "previous_close": previous_close,
        "today_open": float(candles["open"].iloc[0]) if len(candles) else None,
        "candles_1m": recs,
    }
