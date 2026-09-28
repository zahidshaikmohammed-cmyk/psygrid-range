"""PHASE 1 - Candle validation.

Validates timestamps, ordering, duplicates, OHLC consistency, session bounds and
gaps.  The output is a clean frame *containing only genuine candles* plus a
:class:`ValidationReport` describing everything that was dropped or missing.

Policy
  * Missing minutes are reported, never filled.
  * Identical duplicates are collapsed; conflicting duplicates keep the last
    record received (the source's latest revision) and are flagged.
  * Records violating OHLC invariants or with non-positive prices are dropped.
  * A still-forming last candle (``ts + 1 min > now``) is dropped.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import EngineConfig

GOOD, DEGRADED, BAD = "GOOD", "DEGRADED", "BAD"


@dataclass
class ValidationReport:
    symbol: str
    n_input: int = 0
    n_valid: int = 0
    n_malformed: int = 0
    n_duplicates_identical: int = 0
    n_duplicates_conflicting: int = 0
    n_out_of_order: int = 0
    n_misaligned: int = 0
    n_ohlc_violations: int = 0
    n_outside_session: int = 0
    n_forming_dropped: int = 0
    n_sessions: int = 0
    expected_minutes: int = 0
    missing_minutes: int = 0
    gap_fraction: float = 0.0
    largest_gap_minutes: int = 0
    first_candle: str | None = None
    last_candle: str | None = None
    issues: list[str] = field(default_factory=list)
    quality: str = GOOD

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Freshness:
    now: str
    market_open: bool
    last_candle: str | None
    age_minutes: float | None
    is_stale: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def session_bounds(day, cfg: EngineConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last candle *start* timestamps of a regular session."""
    tz = cfg.tz
    o, c = cfg.session_open, cfg.session_close
    d = pd.Timestamp(day).date()
    first = pd.Timestamp(datetime.combine(d, o), tz=tz)
    last = pd.Timestamp(datetime.combine(d, c), tz=tz) - pd.Timedelta(minutes=1)
    return first, last


def minute_of_session(ts: pd.DatetimeIndex | pd.Series, cfg: EngineConfig) -> np.ndarray:
    o = cfg.session_open
    idx = pd.DatetimeIndex(ts)
    return (idx.hour * 60 + idx.minute - (o.hour * 60 + o.minute)).to_numpy()


def clean_candles(
    frame: pd.DataFrame,
    cfg: EngineConfig,
    symbol: str,
    *,
    n_malformed: int = 0,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Return (clean candles indexed by IST timestamp, report)."""
    rep = ValidationReport(symbol=symbol.upper(), n_input=len(frame) + n_malformed, n_malformed=n_malformed)
    cols = ["open", "high", "low", "close", "volume"]
    if frame.empty:
        rep.issues.append("no parseable candles")
        rep.quality = BAD
        empty = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz=cfg.tz, name="timestamp"), dtype=float)
        return empty, rep

    df = frame.copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(cfg.tz)
    ts = ts.dt.tz_convert(cfg.tz)

    # Minute alignment (candle timestamps must be minute starts).
    floored = ts.dt.floor("min")
    rep.n_misaligned = int((floored != ts).sum())
    if rep.n_misaligned:
        rep.issues.append(f"{rep.n_misaligned} timestamps not aligned to the minute (floored)")
    df["timestamp"] = floored

    # Ordering (as received).
    diffs = df["timestamp"].diff().dt.total_seconds().to_numpy()[1:]
    rep.n_out_of_order = int((diffs < 0).sum())
    if rep.n_out_of_order:
        rep.issues.append(f"{rep.n_out_of_order} out-of-order records (re-sorted)")

    # Duplicates.
    df["_seq"] = np.arange(len(df))
    dup_mask = df.duplicated("timestamp", keep=False)
    if dup_mask.any():
        dups = df[dup_mask]
        n_identical = 0
        n_conflict = 0
        for _, g in dups.groupby("timestamp"):
            vals = g[cols].to_numpy()
            if np.allclose(vals, vals[0], equal_nan=True):
                n_identical += len(g) - 1
            else:
                n_conflict += len(g) - 1
        rep.n_duplicates_identical = n_identical
        rep.n_duplicates_conflicting = n_conflict
        if n_identical:
            rep.issues.append(f"{n_identical} identical duplicate candles removed")
        if n_conflict:
            rep.issues.append(f"{n_conflict} conflicting duplicate candles (kept latest record)")
        df = df.sort_values("_seq").drop_duplicates("timestamp", keep="last")
    df = df.drop(columns="_seq").sort_values("timestamp")

    # OHLC invariants.
    o, h, lo, c, v = (df[k].to_numpy(dtype=float) for k in cols)
    bad = (
        (o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0) | (v < 0)
        | (h + 1e-9 < np.maximum(o, c)) | (lo - 1e-9 > np.minimum(o, c)) | (h + 1e-9 < lo)
    )
    rep.n_ohlc_violations = int(bad.sum())
    if rep.n_ohlc_violations:
        rep.issues.append(f"{rep.n_ohlc_violations} candles violate OHLC invariants (dropped)")
    df = df[~bad]

    # Session bounds and weekends.
    mos = minute_of_session(df["timestamp"], cfg)
    in_session = (mos >= 0) & (mos < cfg.session_minutes) & (df["timestamp"].dt.dayofweek.to_numpy() < 5)
    rep.n_outside_session = int((~in_session).sum())
    if rep.n_outside_session:
        rep.issues.append(f"{rep.n_outside_session} candles outside regular session (dropped)")
    df = df[in_session]

    # A candle whose minute has not finished yet is still forming.
    if now is not None and not df.empty:
        now_ts = pd.Timestamp(now).tz_convert(cfg.tz)
        forming = (df["timestamp"] + pd.Timedelta(minutes=1)) > now_ts
        rep.n_forming_dropped = int(forming.sum())
        df = df[~forming]

    df = df.set_index("timestamp")[cols].astype(float)
    df.index.name = "timestamp"
    rep.n_valid = len(df)

    # Gaps, measured inside each session from the open to the last candle seen.
    if not df.empty:
        days = df.index.normalize().unique()
        rep.n_sessions = len(days)
        expected = 0
        missing = 0
        largest = 0
        for d in days:
            day_idx = df.index[df.index.normalize() == d]
            first, _ = session_bounds(d, cfg)
            span = int((day_idx.max() - first) / pd.Timedelta(minutes=1)) + 1
            expected += span
            missing += span - len(day_idx)
            present = np.sort(minute_of_session(day_idx, cfg))
            edges = np.concatenate([[-1], present])
            if len(edges) > 1:
                largest = max(largest, int(np.max(np.diff(edges)) - 1))
        rep.expected_minutes = expected
        rep.missing_minutes = missing
        rep.gap_fraction = missing / expected if expected else 0.0
        rep.largest_gap_minutes = largest
        rep.first_candle = df.index[0].isoformat()
        rep.last_candle = df.index[-1].isoformat()
        if missing:
            rep.issues.append(
                f"{missing} missing minutes ({rep.gap_fraction:.1%}), largest gap {largest} min (not filled)"
            )

    # Quality grade.
    vcfg = cfg["validation"]
    bad_frac = (rep.n_malformed + rep.n_ohlc_violations) / max(rep.n_input, 1)
    if rep.n_valid == 0:
        rep.quality = BAD
        rep.issues.append("no valid candles")
    elif rep.gap_fraction > vcfg["max_gap_fraction"] or bad_frac > vcfg["max_bad_record_fraction"]:
        rep.quality = BAD
    elif (
        rep.gap_fraction > vcfg.get("degraded_gap_fraction", 0.02)
        or rep.n_duplicates_conflicting or rep.n_ohlc_violations or rep.n_malformed
    ):
        rep.quality = DEGRADED
    else:
        rep.quality = GOOD
    return df, rep


def is_market_open(now: datetime, cfg: EngineConfig) -> bool:
    now = pd.Timestamp(now).tz_convert(cfg.tz)
    if now.dayofweek >= 5:
        return False
    first, last = session_bounds(now, cfg)
    return first <= now < last + pd.Timedelta(minutes=1)


def assess_freshness(candles: pd.DataFrame, now: datetime, cfg: EngineConfig) -> Freshness:
    """Data is *fresh* only while the market is open and the last complete
    candle closed no more than ``max_staleness_minutes`` ago."""
    now_ts = pd.Timestamp(now).tz_convert(cfg.tz)
    open_ = is_market_open(now_ts, cfg)
    if candles.empty:
        return Freshness(now_ts.isoformat(), open_, None, None, True, "no candles")
    last = candles.index[-1]
    age = (now_ts - (last + pd.Timedelta(minutes=1))) / timedelta(minutes=1)
    limit = float(cfg["validation"]["max_staleness_minutes"])
    if not open_:
        return Freshness(now_ts.isoformat(), False, last.isoformat(), round(age, 1), True, "market closed")
    if last.normalize() != now_ts.normalize():
        return Freshness(now_ts.isoformat(), True, last.isoformat(), round(age, 1), True, "no candles for today")
    if age > limit:
        return Freshness(
            now_ts.isoformat(), True, last.isoformat(), round(age, 1), True,
            f"last candle closed {age:.1f} min ago (limit {limit:.0f})",
        )
    return Freshness(now_ts.isoformat(), True, last.isoformat(), round(age, 1), False, "fresh")
