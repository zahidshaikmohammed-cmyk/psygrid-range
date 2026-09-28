"""PHASE 1 - validation: ordering, duplicates, OHLC, gaps, session, freshness."""
import pandas as pd
from conftest import ist

from engine.validation import assess_freshness, clean_candles, is_market_open


def _frame(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = [ist(t) for t in df["timestamp"]]
    return df


def test_clean_series_is_good(cfg):
    ts = pd.date_range(ist("2026-03-02 09:15"), periods=30, freq="min")
    df = pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 5.0})
    clean, rep = clean_candles(df, cfg, "CIPLA")
    assert rep.quality == "GOOD" and rep.n_valid == 30 and rep.missing_minutes == 0
    assert clean.index.is_monotonic_increasing and str(clean.index.tz) == "Asia/Kolkata"


def test_duplicates_order_ohlc_and_gaps(cfg):
    rows = [
        ("2026-03-02 09:15", 100, 101, 99, 100, 1),
        ("2026-03-02 09:17", 100, 101, 99, 100.5, 1),   # out of order below
        ("2026-03-02 09:16", 100, 101, 99, 100, 1),
        ("2026-03-02 09:16", 100, 101, 99, 100, 1),     # identical dup
        ("2026-03-02 09:18", 100, 101, 99, 100, 1),
        ("2026-03-02 09:18", 100, 102, 99, 101, 2),     # conflicting dup -> keep last
        ("2026-03-02 09:19", 100, 99, 101, 100, 1),     # high < low -> dropped
        ("2026-03-02 09:23", 100, 101, 99, 100, 1),     # gap 09:20-09:22 (+ dropped 09:19)
        ("2026-03-02 16:00", 100, 101, 99, 100, 1),     # outside session
    ]
    clean, rep = clean_candles(_frame(rows), cfg, "CIPLA")
    assert rep.n_out_of_order == 1
    assert rep.n_duplicates_identical == 1 and rep.n_duplicates_conflicting == 1
    assert clean.loc[ist("2026-03-02 09:18"), "close"] == 101
    assert rep.n_ohlc_violations == 1 and rep.n_outside_session == 1
    assert rep.missing_minutes == 4 and rep.largest_gap_minutes == 4
    assert ist("2026-03-02 09:20") not in clean.index  # gaps are never filled
    assert rep.quality == "BAD"  # 4/9 minutes missing > 10%


def test_forming_candle_dropped(cfg):
    ts = pd.date_range(ist("2026-03-02 09:15"), periods=10, freq="min")
    df = pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0})
    now = ist("2026-03-02 09:24:30")  # 09:24 candle still forming
    clean, rep = clean_candles(df, cfg, "X", now=now.to_pydatetime())
    assert rep.n_forming_dropped == 1 and clean.index[-1] == ist("2026-03-02 09:23")


def test_misaligned_timestamps_floored(cfg):
    clean, rep = clean_candles(_frame([("2026-03-02 09:15:30", 1, 1, 1, 1, 1)]), cfg, "X")
    assert rep.n_misaligned == 1 and clean.index[0] == ist("2026-03-02 09:15")


def test_empty_is_bad(cfg):
    clean, rep = clean_candles(_frame([]), cfg, "X")
    assert clean.empty and rep.quality == "BAD"


def test_freshness(cfg):
    ts = pd.date_range(ist("2026-03-02 09:15"), periods=60, freq="min")  # last candle 10:14
    c = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=ts)
    f = assess_freshness(c, ist("2026-03-02 10:16").to_pydatetime(), cfg)
    assert f.market_open and not f.is_stale and f.age_minutes == 1.0
    f = assess_freshness(c, ist("2026-03-02 10:30").to_pydatetime(), cfg)
    assert f.is_stale and "limit" in f.reason
    f = assess_freshness(c, ist("2026-03-03 10:30").to_pydatetime(), cfg)
    assert f.is_stale and f.reason == "no candles for today"
    f = assess_freshness(c, ist("2026-03-02 18:00").to_pydatetime(), cfg)
    assert not f.market_open and f.is_stale
    assert not is_market_open(ist("2026-03-07 11:00"), cfg)  # Saturday
    assert is_market_open(ist("2026-03-02 15:29:30"), cfg)
    assert not is_market_open(ist("2026-03-02 15:30"), cfg)


def test_future_candles_flagged(cfg):
    rows = [("2026-03-02 09:15", 1, 1, 1, 1, 1), ("2026-03-02 09:16", 1, 1, 1, 1, 1), ("2026-03-02 11:00", 1, 1, 1, 1, 1)]
    clean, rep = clean_candles(_frame(rows), cfg, "X", now=ist("2026-03-02 09:20").to_pydatetime())
    assert rep.n_future == 1 and len(clean) == 2 and rep.quality != "GOOD"


def test_multi_date_live_payload_flagged(cfg):
    rows = [("2026-03-02 15:29", 1, 1, 1, 1, 1), ("2026-03-03 09:15", 1, 1, 1, 1, 1)]
    _, rep = clean_candles(_frame(rows), cfg, "X", single_session=True)
    assert any("spans 2 dates" in i for i in rep.issues)


def test_timezone_conversion_to_ist(cfg):
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-03-02 03:45", tz="UTC")], "open": [1.0], "high": [1.0],
                       "low": [1.0], "close": [1.0], "volume": [1.0]})
    clean, rep = clean_candles(df, cfg, "X")
    assert clean.index[0] == ist("2026-03-02 09:15") and str(clean.index.tz) == "Asia/Kolkata"


def test_negative_volume_and_impossible_ohlc_dropped(cfg):
    rows = [("2026-03-02 09:15", 10, 11, 9, 10, -5),      # negative volume
            ("2026-03-02 09:16", 10, 9.5, 9, 10, 1),      # high < open
            ("2026-03-02 09:17", 10, 11, 10.5, 10, 1),    # low > close
            ("2026-03-02 09:18", 10, 11, 9, 10, 1)]
    clean, rep = clean_candles(_frame(rows), cfg, "X")
    assert rep.n_ohlc_violations == 3 and len(clean) == 1
