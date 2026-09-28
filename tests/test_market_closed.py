"""Closed-market responses (HTTP 200, healthy status, no candles_1m).

Observed on the real endpoint outside market hours: a ~60-byte document with
only ``service``, ``status`` and ``symbol``.  The *value* of ``status`` in that
response has not been recorded here, so the payloads below are CONSTRUCTED to
that shape and cover each plausible case.  They are not captured responses.

States: MARKET_CLOSED (healthy, no session candles), BAD_DATA (malformed or
inconsistent), STALE_DATA (valid but old candles while open), OK.
"""
import json
from pathlib import Path

import pytest
from conftest import FakeHTTP, ist
from synthetic import to_payload

from engine import confidence as C
from engine.config import load_config
from engine.diagnostics import render_diagnostic, run_diagnostic
from engine.forecasting import RangeEngine
from engine.ingestion import MalformedPayloadError, parse_payload
from engine.output import render_text
from engine.storage import HistoryStore
from engine.validation import classify_data_state, clean_candles

EVENING = ist("2026-09-28 20:05")          # Monday, after the close
SUNDAY = ist("2026-09-27 11:00")
SESSION = ist("2026-09-28 11:00")          # Monday, during regular hours
DOC = json.loads((Path(__file__).parent / "fixtures" / "documented_schema_CIPLA.json").read_text())


def closed_doc(status="closed", symbol="CIPLA", **extra):
    return {"service": "psygrid", "status": status, "symbol": symbol, **extra}


def _state(cfg, doc, now):
    snap = parse_payload(doc, "CIPLA", cfg, now.to_pydatetime())
    clean, rep = clean_candles(snap.candles, cfg, "CIPLA", n_malformed=snap.n_malformed, now=now.to_pydatetime(),
                               schema_issues=snap.schema_issues, single_session=True)
    return snap, classify_data_state(endpoint_state=snap.endpoint_state, has_candles=snap.has_candles, clean=clean,
                                     report=rep if snap.has_candles else None, now=now.to_pydatetime(), cfg=cfg)


# ---------------------------------------------------------------- MARKET_CLOSED
@pytest.mark.parametrize("status", ["closed", "CLOSED", "market_closed", "Market Closed", "holiday"])
def test_explicit_closed_status_is_market_closed(cfg, status):
    snap, st = _state(cfg, closed_doc(status), EVENING)
    assert snap.endpoint_state == "CLOSED" and not snap.has_candles and snap.candles.empty
    assert st.state == "MARKET_CLOSED" and st.data_quality == "NOT_APPLICABLE"
    assert snap.schema_issues == []  # the minimal closed document is complete for what it is


@pytest.mark.parametrize("status", ["ok", "success"])
def test_healthy_status_without_candles_outside_hours_is_market_closed(cfg, status):
    for now in (EVENING, SUNDAY):
        _, st = _state(cfg, closed_doc(status), now)
        assert st.state == "MARKET_CLOSED"


def test_empty_candle_list_is_treated_like_absent(cfg):
    _, st = _state(cfg, closed_doc("closed", candles_1m=[]), EVENING)
    assert st.state == "MARKET_CLOSED"


def test_closed_status_during_session_hours_is_closed_with_note(cfg):
    _, st = _state(cfg, closed_doc("closed"), SESSION)
    assert st.state == "MARKET_CLOSED" and "holiday" in st.notes[0]


# ---------------------------------------------------------------- BAD_DATA
def test_live_status_but_no_candles_during_session_is_bad_data(cfg):
    _, st = _state(cfg, closed_doc("ok"), SESSION)
    assert st.state == "BAD_DATA"


@pytest.mark.parametrize("doc,match", [
    (closed_doc("error"), "not a recognised"),
    (closed_doc("idle"), "closed_statuses"),
    (closed_doc("closed", symbol="TCS"), "does not match"),
    ({"service": "psygrid", "symbol": "CIPLA"}, "required"),
    ({"service": "psygrid", "status": "closed"}, "required"),
    (closed_doc("closed", candles_1m="none"), "not a list"),
    (["closed"], "not a JSON object"),
])
def test_malformed_or_unexpected_is_bad_data(cfg, doc, match):
    with pytest.raises(MalformedPayloadError, match=match):
        parse_payload(doc, "CIPLA", cfg, EVENING.to_pydatetime())


def test_impossible_ohlc_is_bad_data(cfg):
    doc = json.loads(json.dumps(DOC))
    for c in doc["candles_1m"][:10]:
        c["high"] = c["low"] - 1
    _, st = _state(cfg, doc, ist("2026-09-28 09:46"))
    assert st.state == "BAD_DATA"


# ---------------------------------------------------------------- STALE / OK
def test_stale_and_ok(cfg):
    _, st = _state(cfg, DOC, ist("2026-09-28 09:46:10"))
    assert st.state == "OK" and st.data_quality == "GOOD"
    _, st = _state(cfg, DOC, ist("2026-09-28 10:30"))
    assert st.state == "STALE_DATA"
    _, st = _state(cfg, DOC, EVENING)  # valid candles after the close
    assert st.state == "MARKET_CLOSED" and st.data_quality == "GOOD"


# ---------------------------------------------------------------- diagnostic
def test_diagnostic_prints_market_closed(cfg):
    d = run_diagnostic("CIPLA", cfg, http=FakeHTTP({"CIPLA": closed_doc("closed")}), now=EVENING.to_pydatetime())
    assert d["verdict"] == "MARKET_CLOSED" and d["connection"].startswith("VERIFIED")
    assert d["endpoint_status"] == "CLOSED" and d["raw_status"] == "closed"
    assert d["candles"] is None and d["data_quality"] == "NOT_APPLICABLE"
    assert d["candle_schema"].startswith("NOT VALIDATED") and d["fields_present"] == ["service", "status", "symbol"]
    assert not d["errors"]
    text = render_diagnostic(d)
    for line in ("Endpoint Status     : CLOSED", "Candles             : n/a",
                 "Data Quality        : NOT_APPLICABLE", "VERDICT             : MARKET_CLOSED",
                 "Candle Schema       : NOT VALIDATED"):
        assert line in text


def test_diagnostic_unrecognised_status_explains_fix(cfg):
    d = run_diagnostic("CIPLA", cfg, http=FakeHTTP({"CIPLA": closed_doc("idle")}), now=EVENING.to_pydatetime())
    assert d["verdict"] == "BAD_DATA" and d["raw_status"] == "idle" and "closed_statuses" in d["errors"][0]
    assert d["connection"].startswith("VERIFIED")  # the connection itself worked


# ---------------------------------------------------------------- forecasting + collect
def test_forecast_on_closed_response_is_market_closed_and_stores_nothing(tmp_path):
    from conftest import SMALL

    cfg = load_config(SMALL, root=tmp_path)
    fc = RangeEngine(cfg, http=FakeHTTP({"CIPLA": closed_doc("closed")})).predict("CIPLA", now=EVENING.to_pydatetime())
    assert fc.forecast_status == C.MARKET_CLOSED and fc.data_state == "MARKET_CLOSED"
    assert fc.data_quality == "NOT_APPLICABLE" and not fc.has_range and fc.current_price is None
    assert HistoryStore(cfg).sessions("CIPLA") == []  # no candles inferred or stored
    assert "MARKET_CLOSED" in render_text(fc, currency="Rs.")
    fc2 = RangeEngine(cfg, http=FakeHTTP({"CIPLA": closed_doc("ok")})).predict(
        "CIPLA", now=SESSION.to_pydatetime(), use_context=False)
    assert fc2.forecast_status == C.BAD_DATA and fc2.data_state == "BAD_DATA"


def test_forecast_on_stale_data_is_refused_with_stale_state(tmp_path, market):
    from conftest import SMALL

    cfg = load_config(SMALL, root=tmp_path)
    day = market["CIPLA"][market["CIPLA"].index.normalize() == market["CIPLA"].index[0].normalize()]
    http = FakeHTTP({"CIPLA": to_payload("CIPLA", day.iloc[:30], previous_close=100.0)})
    fc = RangeEngine(cfg, http=http).predict("CIPLA", now=(day.index[29] + __import__("pandas").Timedelta(minutes=30))
                                             .to_pydatetime(), use_context=False)
    assert fc.forecast_status == C.BAD_DATA and fc.data_state == "STALE_DATA" and "STALE_DATA" in fc.status_reasons[0]
