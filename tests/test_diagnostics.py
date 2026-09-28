"""diagnose_endpoint: measured values on success, nothing fabricated on failure."""
import json
from pathlib import Path

import requests
from conftest import FakeHTTP, ist

from engine.diagnostics import render_diagnostic, run_diagnostic

DOC = json.loads((Path(__file__).parent / "fixtures" / "documented_schema_CIPLA.json").read_text())


def test_diagnostic_success(cfg):
    d = run_diagnostic("cipla", cfg, http=FakeHTTP({"CIPLA": DOC}), now=ist("2026-09-28 09:46:10").to_pydatetime())
    assert d["http_status"] == 200 and d["candles"] == 30 and d["data_quality"] == "GOOD"
    assert d["first_candle"].startswith("2026-09-28T09:15") and d["last_candle"].startswith("2026-09-28T09:44")
    assert d["latest_candle_age_minutes"] == 1.2 and d["verdict"] == "OK"
    assert d["connection"].startswith("VERIFIED") and d["candle_schema"].startswith("VALIDATED")
    assert d["missing_minutes"] == 0 and d["duplicates_identical"] == 0
    assert d["timestamp_format"] == "iso8601_with_offset" and d["raw_saved"]
    text = render_diagnostic(d)
    for label in ("HTTP Status", "Response Time", "Schema", "Endpoint Status", "Security ID", "Candles",
                  "First Candle", "Last Candle", "Latest Price", "Latest Candle Age", "Duplicate Candles",
                  "Missing Minutes", "Timestamp Format", "Timezone", "Data Quality"):
        assert label in text


def test_diagnostic_stale_data(cfg):
    d = run_diagnostic("CIPLA", cfg, http=FakeHTTP({"CIPLA": DOC}), now=ist("2026-09-28 11:00").to_pydatetime())
    assert d["verdict"] == "STALE_DATA" and d["data_quality"] == "GOOD"


def test_diagnostic_failure_fabricates_nothing(cfg):
    cfg.raw["data_source"].update(retries=0)
    d = run_diagnostic("CIPLA", cfg, http=FakeHTTP(exc=requests.ConnectTimeout()))
    assert d["verdict"] == "BAD_DATA" and d["errors"] and d["connection"] == "FAILED"
    for k in ("http_status", "candles", "latest_close", "schema_version", "data_quality"):
        assert k not in d
    assert "n/a" in render_diagnostic(d)


def test_diagnostic_malformed(cfg):
    d = run_diagnostic("CIPLA", cfg, http=FakeHTTP({"CIPLA": {"status": "error", "symbol": "CIPLA", "candles_1m": []}}))
    assert d["verdict"] == "BAD_DATA" and "status" in d["errors"][0]
