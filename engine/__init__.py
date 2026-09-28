"""PSYGRID RANGE ENGINE - probabilistic next-60-minute high/low ranges from
genuine 1-minute OHLCV.

    >>> import engine
    >>> fc = engine.predict("CIPLA")        # -> engine.forecasting.Forecast
    >>> fc.forecast_status, fc.range
"""
from __future__ import annotations

__version__ = "1.0.0"


def predict(symbol: str, **kwargs):
    """Forecast the next-60-minute range for ``symbol`` (see RangeEngine.predict)."""
    from .forecasting import RangeEngine

    return RangeEngine().predict(symbol, **kwargs)
