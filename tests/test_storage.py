"""PHASE 2 - historical storage."""
import pandas as pd

from engine.storage import HistoryStore


def test_roundtrip_merge_and_dedupe(cfg, market):
    store = HistoryStore(cfg)
    df = market["CIPLA"]
    days = df.index.normalize().unique()
    first_day = df[df.index.normalize() == days[0]]
    store.save("CIPLA", first_day.iloc[:100], source="synthetic")
    store.save("CIPLA", first_day.iloc[50:], source="synthetic")  # overlapping fetch later in the day
    loaded = store.load("CIPLA")
    assert len(loaded) == len(first_day)
    pd.testing.assert_frame_equal(loaded, first_day, check_freq=False, check_names=False, rtol=1e-12)
    assert str(loaded.index.tz) == "Asia/Kolkata"

    # newer record for the same minute replaces the older one
    revised = first_day.iloc[[10]].copy()
    revised["close"] = revised["high"]
    store.save("CIPLA", revised, source="synthetic")
    assert store.load("CIPLA").iloc[10]["close"] == revised["close"].iloc[0]


def test_sessions_and_filters(cfg, market):
    store = HistoryStore(cfg)
    store.save("TCS", market["TCS"], source="synthetic")
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
    store.save("M&M", market["CIPLA"].iloc[:20], source="synthetic")
    assert store.symbols() == ["M&M"] and len(store.load("M&M")) == 20


def test_provenance_is_mandatory_and_recorded(cfg, market):
    import pytest

    store = HistoryStore(cfg)
    with pytest.raises(TypeError):
        store.save("CIPLA", market["CIPLA"].iloc[:5])  # no source -> refused
    with pytest.raises(ValueError):
        store.save("CIPLA", market["CIPLA"].iloc[:5], source="")
    store.save("CIPLA", market["CIPLA"].iloc[:5], source="psygrid_live")
    store.save("CIPLA", market["CIPLA"].iloc[3:8], source="import:x.csv")
    day = store.sessions("CIPLA")[0]
    assert store.provenance("CIPLA")[day] == ["import:x.csv", "psygrid_live"]
    assert "live" not in store.symbols()


def test_raw_live_folder_is_not_a_symbol(cfg):
    (cfg.paths.raw / "live").mkdir(parents=True)
    (cfg.paths.raw / "live" / "CIPLA_20260928.json").write_text("{}")
    assert HistoryStore(cfg).symbols() == []
