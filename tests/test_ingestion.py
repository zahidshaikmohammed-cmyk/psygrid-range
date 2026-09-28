"""PHASE 1 - ingestion: parsing, envelope validation, network and cache."""
import json
from datetime import datetime

import pandas as pd
import pytest
import requests
from conftest import FakeHTTP, ist
from synthetic import to_payload

from engine.ingestion import (
    MalformedPayloadError,
    NetworkError,
    endpoint_url,
    get_snapshot,
    parse_candles,
    parse_payload,
    parse_timestamp,
)


def test_parse_timestamp_formats(cfg):
    tz = cfg.tz
    ref = ist("2026-03-02 09:15")
    assert parse_timestamp("2026-03-02T09:15:00+05:30", tz) == ref
    assert parse_timestamp("2026-03-02 09:15:00", tz) == ref  # naive -> IST
    assert parse_timestamp("2026-03-02T03:45:00Z", tz) == ref  # UTC converted
    epoch = int(ref.timestamp())
    assert parse_timestamp(epoch, tz) == ref
    assert parse_timestamp(epoch * 1000, tz) == ref
    assert parse_timestamp(str(epoch), tz) == ref
    assert parse_timestamp("garbage", tz) is None
    assert parse_timestamp(None, tz) is None
    assert parse_timestamp(-5, tz) is None
    assert parse_timestamp(True, tz) is None


def test_parse_candles_dict_and_list_and_malformed(cfg):
    recs = [
        {"timestamp": "2026-03-02 09:15", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10},
        ["2026-03-02 09:16", "100.5", "102", "100", "101", "20"],
        {"timestamp": "2026-03-02 09:17", "open": "x", "high": 1, "low": 1, "close": 1, "volume": 1},
        {"open": 1},
        "junk",
        {"timestamp": "2026-03-02 09:18", "open": float("nan"), "high": 1, "low": 1, "close": 1, "volume": 1},
    ]
    df, bad, examples = parse_candles(recs, cfg.tz)
    assert len(df) == 2 and bad == 4 and examples
    assert str(df["timestamp"].dt.tz) == "Asia/Kolkata"
    assert df["close"].tolist() == [100.5, 101.0]


def test_parse_payload_envelope(cfg):
    now = datetime.now(cfg.tz)
    good = {"service": "psygrid", "status": "ok", "symbol": "CIPLA", "previous_close": 1500,
            "today_open": 1502, "schema_version": 2, "candles_1m": []}
    snap = parse_payload(good, "CIPLA", cfg, now)
    assert snap.previous_close == 1500 and snap.schema_version == "2" and snap.candles.empty
    with pytest.raises(MalformedPayloadError):
        parse_payload([], "CIPLA", cfg, now)
    with pytest.raises(MalformedPayloadError):
        parse_payload({**good, "status": "error"}, "CIPLA", cfg, now)
    with pytest.raises(MalformedPayloadError):
        parse_payload({**good, "symbol": "TCS"}, "CIPLA", cfg, now)
    bad = dict(good)
    del bad["candles_1m"]
    with pytest.raises(MalformedPayloadError):
        parse_payload(bad, "CIPLA", cfg, now)
    with pytest.raises(MalformedPayloadError):
        parse_payload({**good, "candles_1m": "nope"}, "CIPLA", cfg, now)


def test_endpoint_url_escapes_symbols(cfg):
    assert endpoint_url("CIPLA", cfg).endswith("/public/stock/CIPLA.json")
    assert endpoint_url("M&M", cfg).endswith("/M%26M.json")


def test_network_timeout_and_http_errors(cfg):
    cfg.raw["data_source"].update(retries=1, backoff_seconds=0)
    with pytest.raises(NetworkError, match="timed out"):
        get_snapshot("CIPLA", cfg, http=FakeHTTP(exc=requests.Timeout()), use_cache=False)
    with pytest.raises(NetworkError, match="network error"):
        get_snapshot("CIPLA", cfg, http=FakeHTTP(exc=requests.ConnectionError("down")), use_cache=False)
    with pytest.raises(NetworkError, match="404"):
        get_snapshot("CIPLA", cfg, http=FakeHTTP(), use_cache=False)
    with pytest.raises(NetworkError, match="HTTP 503"):
        get_snapshot("CIPLA", cfg, http=FakeHTTP({"CIPLA": {}}, status=503), use_cache=False)

    class NotJSON(FakeHTTP):
        def get(self, url, timeout=None, headers=None):
            from conftest import FakeResponse

            return FakeResponse(text="<html>")

    with pytest.raises(MalformedPayloadError):
        get_snapshot("CIPLA", cfg, http=NotJSON(), use_cache=False)


def test_cache_is_used_only_while_fresh(cfg, market):
    day = market["CIPLA"][market["CIPLA"].index.normalize() == market["CIPLA"].index[0].normalize()]
    http = FakeHTTP({"CIPLA": to_payload("CIPLA", day.iloc[:30])})
    t0 = ist("2026-01-05 09:46")
    s1 = get_snapshot("CIPLA", cfg, now=t0.to_pydatetime(), http=http)
    s2 = get_snapshot("CIPLA", cfg, now=(t0 + pd.Timedelta(seconds=10)).to_pydatetime(), http=http)
    assert not s1.from_cache and s2.from_cache and len(http.calls) == 1
    s3 = get_snapshot("CIPLA", cfg, now=(t0 + pd.Timedelta(seconds=60)).to_pydatetime(), http=http)
    assert not s3.from_cache and len(http.calls) == 2
    blob = json.loads((cfg.paths.cache / "CIPLA.json").read_text())
    assert blob["payload"]["symbol"] == "CIPLA"
