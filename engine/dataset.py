"""Assembles minute grids, market context, features, regimes and targets into
one frame per symbol.  The *same* code path is used for training and for live
inference, so the model always sees identically-constructed features."""
from __future__ import annotations

import warnings

import pandas as pd

from .config import EngineConfig
from .features import (
    MarketContext,
    build_market_context,
    compute_features,
    intraday_log_returns,
    to_minute_grid,
)
from .regime import classify
from .storage import HistoryStore
from .targets import compute_targets
from .validation import clean_candles


def build_symbol_frame(
    candles: pd.DataFrame,
    cfg: EngineConfig,
    symbol: str,
    *,
    context: MarketContext | None = None,
    previous_close: float | None = None,
    with_targets: bool = True,
    grid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    grid = to_minute_grid(candles, cfg) if grid is None else grid
    if grid.empty:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats = compute_features(grid, cfg, symbol=symbol, context=context, previous_close=previous_close)
    reg = classify(feats)
    parts = [grid[["session", "present", "mos"]].rename(columns={"mos": "_mos"}), feats, reg]
    frame = pd.concat(parts, axis=1)
    frame["symbol"] = symbol
    frame["price"] = grid["close"]
    if with_targets:
        frame = pd.concat([frame, compute_targets(grid, feats["sigma"], cfg)], axis=1)
    return frame


def load_clean_history(store: HistoryStore, cfg: EngineConfig, symbol: str, **kw) -> pd.DataFrame:
    raw = store.load(symbol, **kw)
    if raw.empty:
        return raw
    clean, _ = clean_candles(raw.reset_index(), cfg, symbol)
    return clean


def build_training_frame(
    cfg: EngineConfig,
    symbols: list[str] | None = None,
    *,
    store: HistoryStore | None = None,
    use_context: bool | None = None,
    progress=None,
) -> tuple[pd.DataFrame, MarketContext | None]:
    """Build the pooled dataset from local history for ``symbols``
    (default: every universe symbol that has stored history)."""
    store = store or HistoryStore(cfg)
    available = [s for s in cfg.universe.constituents if store.sessions(s)]
    symbols = [s for s in (symbols or available) if s in available]
    use_context = cfg["market_context"]["enabled"] if use_context is None else use_context

    grids: dict[str, pd.DataFrame] = {}
    for s in available if use_context else symbols:
        candles = load_clean_history(store, cfg, s)
        if not candles.empty:
            grids[s] = to_minute_grid(candles, cfg)
    context = None
    if use_context:
        context = build_market_context({s: intraday_log_returns(g) for s, g in grids.items()}, cfg)

    frames = []
    for i, s in enumerate(symbols):
        if s not in grids:
            continue
        if progress:
            progress(f"features {s} ({i + 1}/{len(symbols)})")
        fr = build_symbol_frame(pd.DataFrame(), cfg, s, context=context, grid=grids[s])
        frames.append(fr[fr["present"]])
    if not frames:
        return pd.DataFrame(), context
    data = pd.concat(frames)
    data.index.name = "timestamp"
    return data.reset_index(), context
