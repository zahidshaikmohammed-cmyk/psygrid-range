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


# ---------------------------------------------------------------------------
# Real-endpoint hardening
# ---------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

from engine.config import load_config  # noqa: E402
from engine.ingestion import classify_timestamp, fetch, save_raw_live  # noqa: E402
from engine.validation import clean_candles  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _doc():
    return json.loads((FIXTURES / "documented_schema_CIPLA.json").read_text())


def test_base_url_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PSYGRID_BASE_URL", "http://127.0.0.1:9999/")
    cfg = load_config(root=tmp_path)
    assert endpoint_url("CIPLA", cfg) == "http://127.0.0.1:9999/public/stock/CIPLA.json"
    monkeypatch.delenv("PSYGRID_BASE_URL")
    assert load_config(root=tmp_path)["data_source"]["base_url"] == "http://140.245.226.102:10000"


def test_documented_schema_fixture_parses_cleanly(cfg):
    snap = parse_payload(_doc(), "CIPLA", cfg, datetime.now(cfg.tz))
    assert snap.schema_issues == [] and snap.n_malformed == 0 and len(snap.candles) == 30
    assert snap.service == "psygrid" and snap.security_id == "EXAMPLE" and snap.previous_close == 1498.1
    assert snap.timestamp_format == "iso8601_with_offset"
    assert str(snap.candles["timestamp"].dt.tz) == "Asia/Kolkata"
    clean, rep = clean_candles(snap.candles, cfg, "CIPLA", schema_issues=snap.schema_issues, single_session=True)
    assert rep.quality == "GOOD" and rep.missing_minutes == 0


def test_schema_issues_are_flagged_not_repaired(cfg):
    doc = _doc()
    del doc["security_id"]
    doc["previous_close"] = "abc"
    snap = parse_payload(doc, "CIPLA", cfg, datetime.now(cfg.tz))
    assert any("security_id" in i for i in snap.schema_issues)
    assert any("previous_close" in i for i in snap.schema_issues) and snap.previous_close is None
    _, rep = clean_candles(snap.candles, cfg, "CIPLA", schema_issues=snap.schema_issues)
    assert rep.quality == "DEGRADED"
    for missing in ("status", "symbol", "candles_1m"):
        d = _doc()
        del d[missing]
        with pytest.raises(MalformedPayloadError, match="required"):
            parse_payload(d, "CIPLA", cfg, datetime.now(cfg.tz))


def test_timestamp_format_classification():
    assert classify_timestamp("2026-09-28T09:15:00+05:30") == "iso8601_with_offset"
    assert classify_timestamp("2026-09-28 09:15:00") == "iso8601_naive(assumed IST)"
    assert classify_timestamp("2026-09-28T03:45:00Z") == "iso8601_utc"
    assert classify_timestamp(1790000000) == "epoch_seconds"
    assert classify_timestamp(1790000000000) == "epoch_milliseconds"


def test_mixed_timestamp_formats_are_reported(cfg):
    doc = _doc()
    doc["candles_1m"][0]["timestamp"] = int(ist("2026-09-28 09:15").timestamp())
    snap = parse_payload(doc, "CIPLA", cfg, datetime.now(cfg.tz))
    assert snap.timestamp_format.startswith("MIXED")


def test_retry_after_429_then_success(cfg):
    from conftest import FakeResponse

    cfg.raw["data_source"].update(retries=2, backoff_seconds=0, max_retry_after_seconds=0)

    class RateLimited(FakeHTTP):
        n = 0

        def get(self, url, timeout=None, headers=None):
            self.calls.append((url, timeout))
            RateLimited.n += 1
            if RateLimited.n == 1:
                r = FakeResponse(status=429)
                r.headers = {"Retry-After": "5"}
                return r
            return FakeResponse(_doc())

    http = RateLimited()
    fr = fetch("CIPLA", cfg, http=http)
    assert fr.http_status == 200 and fr.attempts == 2
    assert http.calls[0][1] == (5.0, 10.0)  # separate connect / read timeouts


def test_http_500_exhausts_retries(cfg):
    cfg.raw["data_source"].update(retries=1, backoff_seconds=0)
    http = FakeHTTP({"CIPLA": _doc()}, status=500)
    with pytest.raises(NetworkError, match="HTTP 500"):
        fetch("CIPLA", cfg, http=http)
    assert len(http.calls) == 2


def test_oversized_response_rejected(cfg):
    from conftest import FakeResponse

    cfg.raw["data_source"]["max_response_bytes"] = 10

    class Big(FakeHTTP):
        def get(self, url, timeout=None, headers=None):
            r = FakeResponse(_doc())
            r.content = b"x" * 100
            return r

    with pytest.raises(MalformedPayloadError, match="limit"):
        fetch("CIPLA", cfg, http=Big())


def test_connect_timeout_message(cfg):
    cfg.raw["data_source"].update(retries=0)
    with pytest.raises(NetworkError, match="connection timed out"):
        fetch("CIPLA", cfg, http=FakeHTTP(exc=requests.ConnectTimeout()))


def test_raw_live_archive_never_shrinks(cfg):
    from engine.ingestion import FetchResult

    now = ist("2026-09-28 11:00").to_pydatetime()
    full = _doc()
    part = _doc()
    part["candles_1m"] = part["candles_1m"][:5]
    p1 = save_raw_live("CIPLA", FetchResult("u", 200, 1.0, None, full, 1), now, cfg)
    assert p1.name == "CIPLA_20260928.json" and p1.parent == cfg.paths.raw / "live"
    p2 = save_raw_live("CIPLA", FetchResult("u", 200, 1.0, None, part, 1), now, cfg)
    assert p2 != p1 and p2.name.startswith("CIPLA_20260928_")
    assert len(json.loads(p1.read_text())["payload"]["candles_1m"]) == 30  # the fuller capture survives
    p3 = save_raw_live("CIPLA", FetchResult("u", 200, 1.0, None, full, 1), now, cfg)
    assert p3 == p1


def test_get_snapshot_archives_raw_response(cfg):
    snap = get_snapshot("CIPLA", cfg, now=ist("2026-09-28 09:46").to_pydatetime(),
                        http=FakeHTTP({"CIPLA": _doc()}), use_cache=False)
    assert snap.raw_path and Path(snap.raw_path).exists()
    assert snap.fetch.http_status == 200
