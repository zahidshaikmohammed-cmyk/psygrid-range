"""PHASE 3 - Feature engine, built exclusively from 1-minute OHLCV.

Design rules
  * **Strictly causal.**  Every feature at minute *t* uses candles <= *t* only
    (profiles such as time-of-day volume use *previous sessions* only).
    ``tests/test_features.py`` checks this by truncating the future.
  * **Gaps stay gaps.**  Candles are placed on a per-session minute grid; a
    missing minute is NaN and propagates to features that need it.  Nothing
    is forward-filled into price data.
  * **Scale-free for pooling.**  Most features are normalised by the stock's
    own volatility (``sigma``) so one global model can learn across stocks,
    while level features (``log_sigma``, ``log_turnover_ref``, sector) keep
    stock-specific behaviour visible to the model.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EngineConfig
from .validation import minute_of_session, session_bounds

TOD_BUCKETS = ["OPEN", "MORNING", "MIDDAY", "AFTERNOON", "CLOSE"]
_TOD_EDGES = [0, 30, 105, 225, 315]  # minutes since open: 09:15, 09:45, 11:00, 13:00, 14:30
SIGMA_FLOOR = 1e-5


def tod_bucket_code(mos: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(_TOD_EDGES, mos, side="right") - 1, 0, len(TOD_BUCKETS) - 1)


# --------------------------------------------------------------------------- #
# Minute grid
# --------------------------------------------------------------------------- #
def to_minute_grid(candles: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    """Place candles on a complete per-session minute grid.

    Each session spans from the open to the last observed candle of that
    session.  Missing minutes are NaN rows with ``present == False``.
    """
    cols = ["open", "high", "low", "close", "volume"]
    if candles.empty:
        g = pd.DataFrame(columns=cols + ["present", "session", "mos"])
        g.index = pd.DatetimeIndex([], tz=cfg.tz, name="timestamp")
        return g
    pieces = []
    for day, part in candles.groupby(candles.index.normalize()):
        first, _ = session_bounds(day, cfg)
        idx = pd.date_range(first, part.index.max(), freq="min", name="timestamp")
        pieces.append(part[cols].reindex(idx))
    grid = pd.concat(pieces)
    grid["present"] = grid["close"].notna()
    grid["session"] = grid.index.strftime("%Y-%m-%d")
    grid["mos"] = minute_of_session(grid.index, cfg)
    return grid


def intraday_log_returns(grid: pd.DataFrame) -> pd.Series:
    """1-minute log return inside a session.  The first minute of each session
    uses close/open (the overnight gap is a separate feature).  A return next
    to a missing minute is NaN."""
    c = grid["close"]
    same = grid["session"].eq(grid["session"].shift(1))
    lr = np.log(c / c.shift(1)).where(same)
    first = ~same
    lr = lr.where(~first, np.log(grid["close"] / grid["open"]))
    return lr


# --------------------------------------------------------------------------- #
# Market / sector context
# --------------------------------------------------------------------------- #
PROXY_SOURCE = "NIFTY50_PROXY"
INDEX_SOURCE = "NIFTY50_INDEX"
CONTEXT_SOURCES = (PROXY_SOURCE, INDEX_SOURCE)


@dataclass
class MarketContext:
    """Per-minute market and sector returns.

    ``source`` is either ``NIFTY50_PROXY`` - an *equal-weight* average of the
    constituents' 1-minute log returns built by this engine, which is **not**
    the official cap-weighted NIFTY 50 index - or ``NIFTY50_INDEX`` when a
    genuine index OHLCV series is configured (``index_symbol`` in the
    universe file).  The two are never mixed: a model records the source it
    was trained with and refuses a different one at inference.

    ``frame`` columns: ``m_lr`` (market 1m log return), ``m_count``,
    ``m_frac`` (coverage), and per-sector ``sec_sum::<SECTOR>`` /
    ``sec_cnt::<SECTOR>`` (constituent sums, for leave-one-out sector returns).
    """

    frame: pd.DataFrame
    universe_size: int
    source: str = PROXY_SOURCE


def build_market_context(
    lrs: dict[str, pd.Series],
    cfg: EngineConfig,
    *,
    index_lr: pd.Series | None = None,
) -> MarketContext | None:
    """Build market context.  With ``market_context.source == NIFTY50_PROXY``
    the market series is the equal-weight constituent average (only where at
    least ``min_constituent_fraction`` of the universe is present).  With
    ``NIFTY50_INDEX`` it is the supplied genuine index series and nothing else."""
    if not lrs:
        return None
    source = cfg["market_context"].get("source", PROXY_SOURCE)
    if source not in CONTEXT_SOURCES:
        raise ValueError(f"unknown market_context.source {source!r}")
    panel = pd.DataFrame(lrs).sort_index()
    n_univ = len(cfg.universe.constituents)
    cnt = panel.notna().sum(axis=1)
    frac = cnt / n_univ
    if source == INDEX_SOURCE:
        if index_lr is None or index_lr.dropna().empty:
            return None
        idx = panel.index.union(index_lr.index)
        panel = panel.reindex(idx)
        cnt = panel.notna().sum(axis=1)
        m_lr = index_lr.reindex(idx)
        frac = m_lr.notna().astype(float).where(m_lr.notna())
    else:
        min_frac = float(cfg["market_context"]["min_constituent_fraction"])
        m_lr = panel.mean(axis=1, skipna=True).where(frac >= min_frac)
    out = {"m_lr": m_lr, "m_count": cnt, "m_frac": frac}
    for sector in sorted(set(cfg.universe.sectors.values())):
        members = [s for s in panel.columns if cfg.universe.sector_of(s) == sector]
        if not members:
            continue
        out[f"sec_sum::{sector}"] = panel[members].sum(axis=1, skipna=True)
        out[f"sec_cnt::{sector}"] = panel[members].notna().sum(axis=1)
    return MarketContext(frame=pd.DataFrame(out), universe_size=n_univ, source=source)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sshift(s: pd.Series, k: int, session: pd.Series) -> pd.Series:
    """Shift by k minutes without crossing a session boundary."""
    return s.shift(k).where(session.eq(session.shift(k)))


def _sroll(s: pd.Series, w: int, session: pd.Series, how: str, minp: int | None = None) -> pd.Series:
    """Session-bounded rolling aggregate."""
    minp = max(1, int(0.6 * w)) if minp is None else minp
    r = s.groupby(session.to_numpy(), sort=False).rolling(w, min_periods=minp)
    r = getattr(r, how)()
    return r.reset_index(level=0, drop=True).reindex(s.index)


def _roll(s: pd.Series, w: int, how: str, minp: int | None = None) -> pd.Series:
    minp = max(1, int(0.6 * w)) if minp is None else minp
    return getattr(s.rolling(w, min_periods=minp), how)()


def _session_profile(values: pd.Series, grid: pd.DataFrame, n_prior: int, n_minutes: int, how: str) -> pd.DataFrame:
    """Per (session, minute-of-session) profile computed from the *previous*
    ``n_prior`` sessions only (median or mean), smoothed over 5 minutes."""
    piv = pd.DataFrame({"v": values.to_numpy(), "s": grid["session"].to_numpy(), "m": grid["mos"].to_numpy()})
    piv = piv.pivot_table(index="s", columns="m", values="v", aggfunc="first", dropna=False)
    piv = piv.reindex(columns=range(n_minutes)).sort_index()
    agg = getattr(piv.rolling(n_prior, min_periods=3), how)().shift(1)
    return agg.T.rolling(5, center=True, min_periods=1).mean().T


def _lookup_profile(profile: pd.DataFrame, grid: pd.DataFrame) -> np.ndarray:
    s_idx = profile.index.get_indexer(grid["session"])
    m = grid["mos"].to_numpy().astype(int)
    arr = profile.to_numpy()
    out = np.full(len(grid), np.nan)
    ok = (s_idx >= 0) & (m >= 0) & (m < arr.shape[1])
    out[ok] = arr[s_idx[ok], m[ok]]
    return out


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #
def compute_features(
    grid: pd.DataFrame,
    cfg: EngineConfig,
    *,
    symbol: str,
    context: MarketContext | None = None,
    previous_close: float | None = None,
) -> pd.DataFrame:
    """Compute the full feature frame on a minute grid (see module doc)."""
    f = pd.DataFrame(index=grid.index)
    ses = grid["session"]
    o, h, lo, c, v = (grid[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    n_min = cfg.session_minutes
    H = cfg.horizon
    lr = intraday_log_returns(grid)
    lc = np.log(c)

    # ---------------- volatility scale (causal) --------------------------------
    sess_var = (lr**2).groupby(ses.to_numpy()).mean()
    ref_var = sess_var.rolling(5, min_periods=2).mean().shift(1)  # previous sessions only
    ref_var_row = ses.map(ref_var).astype(float)
    ewm_var30 = (lr**2).ewm(span=30, min_periods=10, ignore_na=True).mean()
    ewm_var120 = (lr**2).ewm(span=120, min_periods=30, ignore_na=True).mean()
    sess_so_far = (lr**2).groupby(ses.to_numpy()).expanding(min_periods=5).mean().reset_index(level=0, drop=True)
    sess_so_far = sess_so_far.reindex(grid.index)
    base_var = ewm_var30.fillna(sess_so_far)
    sigma2 = np.where(ref_var_row.notna(), 0.5 * base_var + 0.5 * ref_var_row, base_var)
    sigma = pd.Series(np.sqrt(sigma2), index=grid.index).clip(lower=SIGMA_FLOOR)
    f["sigma"] = sigma
    f["log_sigma"] = np.log(sigma)
    f["log_ref_vol"] = np.log(np.sqrt(ref_var_row))

    # ---------------- price / returns ------------------------------------------
    for k in (1, 3, 5, 10, 15, 30, 60):
        r = lc - _sshift(lc, k, ses)
        f[f"z_ret_{k}"] = r / (sigma * np.sqrt(k))
    f["acc_5"] = f["z_ret_5"] - _sshift(f["z_ret_5"], 5, ses)
    f["acc_15"] = f["z_ret_15"] - _sshift(f["z_ret_15"], 15, ses)
    first_present_open = o.where(grid["mos"].eq(0))
    sess_open = first_present_open.groupby(ses.to_numpy()).transform("first")
    f["ret_open"] = (lc - np.log(sess_open)) / (sigma * np.sqrt(grid["mos"] + 1))
    # overnight gap: previous session's last close
    last_close = c.groupby(ses.to_numpy()).last()
    prev_close_map = last_close.shift(1)
    prev_close = ses.map(prev_close_map).astype(float)
    if previous_close is not None:
        last_s = ses.iloc[-1]
        prev_close = prev_close.where(~(ses.eq(last_s) & prev_close.isna()), previous_close)
    f["gap"] = np.log(sess_open / prev_close) / (np.sqrt(ref_var_row).clip(lower=SIGMA_FLOOR) * np.sqrt(n_min))
    for w in (30, 60):
        hi = _sroll(h, w, ses, "max", minp=1)
        lw = _sroll(lo, w, ses, "min", minp=1)
        f[f"dist_high_{w}"] = np.log(c / hi) / sigma
        f[f"dist_low_{w}"] = np.log(c / lw) / sigma
    s_hi = h.groupby(ses.to_numpy()).cummax()
    s_lo = lo.groupby(ses.to_numpy()).cummin()
    elapsed = sigma * np.sqrt(grid["mos"] + 1)
    f["dist_sess_high"] = np.log(c / s_hi) / elapsed
    f["dist_sess_low"] = np.log(c / s_lo) / elapsed
    rng = (s_hi - s_lo).replace(0, np.nan)
    f["range_pos_sess"] = ((c - s_lo) / rng).clip(0, 1)
    f["sess_range"] = np.log(s_hi / s_lo) / (sigma * np.sqrt(grid["mos"] + 1))

    # ---------------- volatility -----------------------------------------------
    for w in (15, 30, 60):
        f[f"rv_{w}"] = np.sqrt(_roll(lr**2, w, "mean")) / sigma
    f["ewm_vol_30"] = np.sqrt(ewm_var30) / sigma
    f["ewm_vol_120"] = np.sqrt(ewm_var120) / sigma
    f["vol_ratio_ref"] = np.sqrt(ewm_var30 / ref_var_row)
    f["vol_acc"] = np.sqrt(_roll(lr**2, 15, "mean") / _roll(lr**2, 60, "mean"))
    f["ewm_acc"] = np.sqrt(ewm_var30 / ewm_var120)
    prev_c = _sshift(c, 1, ses)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1, skipna=True) / c
    tr = tr.where(grid["present"])
    atr14 = tr.ewm(alpha=1 / 14, min_periods=5, ignore_na=True).mean()
    f["atr_14"] = atr14 / sigma
    hl = (np.log(h / lo)).where(grid["present"])
    r5, r15, r60 = _roll(hl, 5, "mean"), _roll(hl, 15, "mean"), _roll(hl, 60, "mean")
    f["range_5"] = r5 / sigma
    f["compress_5_60"] = r5 / r60
    f["compress_15_60"] = r15 / r60
    f["vol_of_vol"] = _roll(hl, 30, "std") / r60

    # ---------------- candle structure -----------------------------------------
    body = (c - o) / c
    upper = (h - np.maximum(o, c)) / c
    lower = (np.minimum(o, c) - lo) / c
    span = (h - lo) / c
    f["body"] = body / sigma
    f["upper_wick"] = upper / sigma
    f["lower_wick"] = lower / sigma
    br = (body.abs() / span.replace(0, np.nan)).fillna(0).where(grid["present"])
    f["body_ratio"] = br
    f["body_ratio_5"] = _roll(br, 5, "mean")
    bull = np.sign(c - o).where(grid["present"])
    f["bull_frac_10"] = _roll((bull > 0).astype(float).where(grid["present"]), 10, "mean")
    # signed run length of consecutive same-direction candles (session-bounded)
    sgn = bull.fillna(0)
    sess_arr = ses.to_numpy()
    new_run = sgn.ne(sgn.shift()) | ses.ne(ses.shift())
    run = sgn * (sgn.groupby(new_run.cumsum().to_numpy()).cumcount() + 1)
    f["run_length"] = np.where(grid["present"], run, np.nan)
    wick_lo_share = (lower / span.replace(0, np.nan)).fillna(0)
    wick_hi_share = (upper / span.replace(0, np.nan)).fillna(0)
    prior_down = _sshift(f["z_ret_15"], 1, ses) < -1.0
    prior_up = _sshift(f["z_ret_15"], 1, ses) > 1.0
    prev_body = _sshift(body, 1, ses)
    engulf_bull = (body > 0) & (prev_body < 0) & (body.abs() > prev_body.abs())
    engulf_bear = (body < 0) & (prev_body > 0) & (body.abs() > prev_body.abs())
    rev_bull = ((wick_lo_share > 0.6) | engulf_bull) & prior_down
    rev_bear = ((wick_hi_share > 0.6) | engulf_bear) & prior_up
    f["reversal_signal"] = (rev_bull.astype(float) - rev_bear.astype(float)).where(grid["present"])
    f["reversal_signal_5"] = _roll(f["reversal_signal"], 5, "sum", minp=1)

    # ---------------- volume ---------------------------------------------------
    vp = v.where(grid["present"])
    m5, m15, m60 = _roll(vp, 5, "mean"), _roll(vp, 15, "mean"), _roll(vp, 60, "mean")
    s60 = _roll(vp, 60, "std")
    f["log_vol_ratio_5_60"] = np.log((m5 + 1) / (m60 + 1))
    f["log_vol_ratio_15_60"] = np.log((m15 + 1) / (m60 + 1))
    f["vol_z"] = (vp - m60) / s60.replace(0, np.nan)
    f["volume_acc"] = np.log((m5 + 1) / (m5.shift(5) + 1))
    vprof = _session_profile(vp, grid, 10, n_min, "median")
    exp_v = pd.Series(_lookup_profile(vprof, grid), index=grid.index)
    sum_v15 = _sroll(vp, 15, ses, "sum", minp=1)
    sum_e15 = _sroll(exp_v, 15, ses, "sum", minp=1)
    f["rvol_15"] = np.log((sum_v15 + 1) / (sum_e15 + 1))
    cum_v = vp.fillna(0).groupby(sess_arr).cumsum()
    cum_e = exp_v.fillna(0).groupby(sess_arr).cumsum()
    f["rvol_session"] = np.log((cum_v + 1) / (cum_e + 1)).where(exp_v.notna())
    f["signed_vol_5"] = _roll((np.sign(c - o) * vp).where(grid["present"]), 5, "sum") / (m60 * 5 + 1)
    turnover = (c * vp).groupby(sess_arr).mean()
    f["log_turnover_ref"] = ses.map(np.log1p(turnover.rolling(5, min_periods=1).mean().shift(1))).astype(float)

    # ---------------- trend / structure ----------------------------------------
    ema = {}
    for span_ in (9, 20, 50):
        ema[span_] = c.ewm(span=span_, min_periods=span_ // 2, ignore_na=True).mean()
        f[f"d_ema{span_}"] = np.log(c / ema[span_]) / sigma
    f["ema_spread_9_20"] = np.log(ema[9] / ema[20]) / sigma
    f["ema_spread_20_50"] = np.log(ema[20] / ema[50]) / sigma
    f["ema20_slope"] = np.log(ema[20] / ema[20].shift(5)) / (sigma * np.sqrt(5))
    f["ema9_slope"] = np.log(ema[9] / ema[9].shift(3)) / (sigma * np.sqrt(3))
    tp = (h + lo + c) / 3
    pv = (tp * vp).fillna(0).groupby(sess_arr).cumsum()
    vv = vp.fillna(0).groupby(sess_arr).cumsum()
    vwap = (pv / vv.replace(0, np.nan)).where(grid["present"])
    f["d_vwap"] = np.log(c / vwap) / elapsed
    above = (c > ema[20]).astype(float).where(grid["present"])
    f["persistence_30"] = _sroll(above, 30, ses, "mean") - 0.5
    mx15 = _sroll(h, 15, ses, "max", minp=5)
    mn15 = _sroll(lo, 15, ses, "min", minp=5)
    pmx15 = _sshift(mx15, 15, ses)
    pmn15 = _sshift(mn15, 15, ses)
    f["struct_score"] = np.sign(mx15 - pmx15) + np.sign(mn15 - pmn15)
    prior_hi60 = _sshift(_sroll(h, 60, ses, "max", minp=10), 1, ses)
    prior_lo60 = _sshift(_sroll(lo, 60, ses, "min", minp=10), 1, ses)
    f["breakout_60"] = np.log(c / prior_hi60) / sigma
    f["breakdown_60"] = np.log(c / prior_lo60) / sigma
    rej = ((h > prior_hi60) & (c < prior_hi60)).astype(float) - ((lo < prior_lo60) & (c > prior_lo60)).astype(float)
    f["rejection_5"] = _roll(rej.where(grid["present"]), 5, "sum", minp=1)

    # ---------------- momentum -------------------------------------------------
    up = lr.clip(lower=0)
    dn = (-lr).clip(lower=0)
    au = up.ewm(alpha=1 / 14, min_periods=10, ignore_na=True).mean()
    ad = dn.ewm(alpha=1 / 14, min_periods=10, ignore_na=True).mean()
    f["rsi_14"] = 100 * au / (au + ad).replace(0, np.nan)
    f["mom_acc"] = f["z_ret_10"] - _sshift(f["z_ret_10"], 10, ses)
    sma20 = _roll(c, 20, "mean")
    sd20 = _roll(c, 20, "std")
    f["eq_z"] = (c - sma20) / sd20.replace(0, np.nan)

    # ---------------- time -----------------------------------------------------
    mos = grid["mos"].astype(float)
    f["mos"] = mos
    f["mtc"] = n_min - 1 - mos
    f["tod_bucket"] = tod_bucket_code(grid["mos"].to_numpy())
    f["dow"] = grid.index.dayofweek
    aprof = _session_profile(lr.abs(), grid, 20, n_min, "median")
    a_now = pd.Series(_lookup_profile(aprof, grid), index=grid.index)
    arr = aprof.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof_mean = np.nanmean(arr, axis=1, keepdims=True)
        fut = np.full_like(arr, np.nan)
        for j in range(arr.shape[1]):
            seg = arr[:, j + 1 : j + 1 + H]
            if seg.shape[1]:
                fut[:, j] = np.nanmean(seg, axis=1)
    fut_df = pd.DataFrame(fut / prof_mean, index=aprof.index, columns=aprof.columns)
    prof_mean_row = ses.map(pd.Series(prof_mean[:, 0], index=aprof.index)).astype(float)
    f["seas_now"] = a_now / prof_mean_row
    f["seas_future"] = _lookup_profile(fut_df, grid)

    # ---------------- market / sector context ----------------------------------
    sector = cfg.universe.sector_of(symbol)
    f = _context_features(f, grid, lr, sigma, context, sector)
    f["sector_code"] = sorted(set(cfg.universe.sectors.values())).index(sector) if sector in cfg.universe.sectors.values() else -1
    if cfg["model"].get("use_symbol_feature"):
        cons = list(cfg.universe.constituents)
        f["symbol_code"] = cons.index(symbol) if symbol in cons else -1
    return f.replace([np.inf, -np.inf], np.nan)


CONTEXT_FEATURES = [
    "ctx_frac", "m_z_ret_5", "m_z_ret_15", "m_z_ret_30", "m_z_ret_60", "m_vol_ratio",
    "rs_5", "rs_15", "rs_30", "rs_60", "beta_60", "corr_60", "sec_rs_15", "sec_rs_60",
]


def _context_features(f, grid, lr, sigma, context, sector):
    if context is None:
        for k in CONTEXT_FEATURES:
            f[k] = np.nan
        return f
    ses = grid["session"]
    cf = context.frame.reindex(grid.index)
    m_lr = cf["m_lr"]
    f["ctx_frac"] = cf["m_frac"]
    m_var30 = (m_lr**2).ewm(span=30, min_periods=10, ignore_na=True).mean()
    m_sess_var = (m_lr**2).groupby(ses.to_numpy()).mean()
    m_ref = ses.map(m_sess_var.rolling(5, min_periods=2).mean().shift(1)).astype(float)
    m_sigma = np.sqrt(np.where(m_ref.notna(), 0.5 * m_var30 + 0.5 * m_ref, m_var30))
    m_sigma = pd.Series(m_sigma, index=grid.index).clip(lower=SIGMA_FLOOR)
    f["m_vol_ratio"] = np.sqrt(m_var30 / m_ref)
    for k in (5, 15, 30, 60):
        m_ret = _sroll(m_lr, k, ses, "sum", minp=int(0.8 * k))
        s_ret = _sroll(lr, k, ses, "sum", minp=int(0.8 * k))
        if k in (5, 15, 30, 60):
            f[f"m_z_ret_{k}"] = m_ret / (m_sigma * np.sqrt(k))
        f[f"rs_{k}"] = (s_ret - m_ret) / (sigma * np.sqrt(k))
    cov = (lr * m_lr).rolling(60, min_periods=30).mean() - lr.rolling(60, min_periods=30).mean() * m_lr.rolling(60, min_periods=30).mean()
    mvar = m_lr.rolling(60, min_periods=30).var(ddof=0)
    svar = lr.rolling(60, min_periods=30).var(ddof=0)
    f["beta_60"] = cov / mvar.replace(0, np.nan)
    f["corr_60"] = cov / np.sqrt(mvar * svar).replace(0, np.nan)
    ssum, scnt = cf.get(f"sec_sum::{sector}"), cf.get(f"sec_cnt::{sector}")
    if ssum is not None:
        own_present = lr.notna().astype(float)
        loo_cnt = scnt - own_present
        sec_lr = ((ssum - lr.fillna(0)) / loo_cnt.replace(0, np.nan)).where(loo_cnt > 0)
        for k in (15, 60):
            sec_ret = _sroll(sec_lr, k, ses, "sum", minp=int(0.8 * k))
            s_ret = _sroll(lr, k, ses, "sum", minp=int(0.8 * k))
            f[f"sec_rs_{k}"] = (s_ret - sec_ret) / (sigma * np.sqrt(k))
    else:
        f["sec_rs_15"] = np.nan
        f["sec_rs_60"] = np.nan
    return f


CATEGORICAL_FEATURES = ["tod_bucket", "sector_code", "regime_code", "vol_state_code", "symbol_code"]
