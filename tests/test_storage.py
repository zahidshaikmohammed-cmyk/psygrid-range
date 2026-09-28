"""PHASE 2 - historical storage."""
import pandas as pd

from engine.storage import HistoryStore


def test_roundtrip_merge_and_dedupe(cfg, market):
    store = HistoryStore(cfg)
    df = market["CIPLA"]
    days = df.index.normalize().unique()
    first_day = df[df.index.normalize() == days[0]]
    store.save("CIPLA", first_day.iloc[:100])
    store.save("CIPLA", first_day.iloc[50:])  # overlapping fetch later in the day
    loaded = store.load("CIPLA")
    assert len(loaded) == len(first_day)
    pd.testing.assert_frame_equal(loaded, first_day, check_freq=False, check_names=False, rtol=1e-12)
    assert str(loaded.index.tz) == "Asia/Kolkata"

    # newer record for the same minute replaces the older one
    revised = first_day.iloc[[10]].copy()
    revised["close"] = revised["high"]
    store.save("CIPLA", revised)
    assert store.load("CIPLA").iloc[10]["close"] == revised["close"].iloc[0]


def test_sessions_and_filters(cfg, market):
    store = HistoryStore(cfg)
    store.save("TCS", market["TCS"])
    sessions = store.sessions("TCS")
    assert len(sessions) == 24 and sessions == sorted(sessions)
    last3 = store.load("TCS", last_n_sessions=3)
    assert last3.index.normalize().nunique() == 3
    excl = store.load("TCS", exclude=sessions[-1])
    assert excl.index.normalize().nunique() == 23
    assert store.symbols() == ["TCS"]
    assert store.load("NOPE").empty


def test_special_characters_in_symbol(cfg, market):
    store = HistoryStore(cfg)
    store.save("M&M", market["CIPLA"].iloc[:20])
    assert store.symbols() == ["M&M"] and len(store.load("M&M")) == 20
