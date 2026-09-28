"""PHASES 9-10 - engine end to end: live (mocked HTTP), replay, refusals, CLI."""
import json
import subprocess
import sys

import pytest
import requests
from conftest import ROOT, SMALL, FakeHTTP, ist
from synthetic import generate_market, to_payload

from engine import confidence as C
from engine.config import load_config
from engine.forecasting import RangeEngine
from engine.output import render_json, render_text
from engine.storage import HistoryStore


@pytest.fixture(scope="module")
def live_day(market):
    """A fresh 'today' (not in the stored history) for every test symbol."""
    cfg = load_config(SMALL)
    last_close = {s: float(df["close"].iloc[-1]) for s, df in market.items()}
    day = generate_market(list(market), cfg.universe.sectors, n_sessions=1, seed=99, start="2026-02-09",
                          base_prices=last_close, drop_fraction=0.0)
    return day, last_close


def _engine(root, overrides=None, http=None):
    cfg = load_config({**SMALL, **(overrides or {})}, root=root)
    return RangeEngine(cfg, http=http)


def _payloads(day, last_close, upto):
    return {s: to_payload(s, df[df.index <= upto], previous_close=last_close[s]) for s, df in day.items()}


def test_live_forecast_end_to_end(trained_home, live_day):
    root, _ = trained_home
    day, last_close = live_day
    upto = ist("2026-02-09 11:29")
    http = FakeHTTP(_payloads(day, last_close, upto))
    eng = _engine(root, http=http)
    fc = eng.predict("cipla", now=ist("2026-02-09 11:30:20").to_pydatetime())
    assert fc.symbol == "CIPLA" and fc.mode == "LIVE"
    assert fc.timestamp.startswith("2026-02-09T11:30") and fc.horizon_minutes == 60
    assert fc.market_context["available"] and len(http.calls) == 50  # whole universe queried
    assert fc.market_context["constituents_fetched_failed"] == 50 - len(day)
    assert fc.forecast_status in (C.HIGH, C.MEDIUM, C.NO_EDGE)
    if fc.has_range:
        p = fc.current_price
        assert fc.range["low"] < fc.central_range["low"] < p < fc.central_range["high"] < fc.range["high"]
        assert fc.upside_pct > 0 > fc.downside_pct
        q = fc.quantiles["upper_high"]
        assert q["p10"] <= q["p50"] <= q["p90"]
    else:
        assert fc.status_reasons
    # live candles were persisted into history
    assert "2026-02-09" in HistoryStore(eng.cfg).sessions("CIPLA")
    js = json.loads(render_json(fc))
    for k in ("symbol", "timestamp", "current_price", "horizon_minutes", "range", "central_range", "upside_pct",
              "downside_pct", "regime", "volatility_state", "forecast_status", "data_quality", "model_version"):
        assert k in js
    text = render_text(fc, currency="Rs.")
    assert "PSYGRID 1-HOUR RANGE ENGINE" in text and "FORECAST STATUS" in text


def test_forced_forecast_path_is_consistent(trained_home, live_day):
    """With permissive confidence thresholds a range must be produced and be coherent."""
    root, _ = trained_home
    day, last_close = live_day
    upto = ist("2026-02-09 12:14")
    permissive = {"confidence": {"min_cell_sessions": 0, "medium_coverage_tolerance": 1.0, "min_skill": -10,
                                 "symbol_min_sessions": 0, "domain_quantile": 0.0}}
    eng = _engine(root, permissive, http=FakeHTTP(_payloads(day, last_close, upto)))
    eng.pipeline()[0].domain = {}
    fc = eng.predict("TCS", now=ist("2026-02-09 12:15:05").to_pydatetime())
    assert fc.forecast_status in (C.HIGH, C.MEDIUM), fc.status_reasons
    assert fc.range["low"] < fc.current_price < fc.range["high"]


def test_stale_data_is_refused(trained_home, live_day):
    root, _ = trained_home
    day, last_close = live_day
    eng = _engine(root, http=FakeHTTP(_payloads(day, last_close, ist("2026-02-09 10:00"))))
    fc = eng.predict("CIPLA", now=ist("2026-02-09 11:00").to_pydatetime(), use_context=False)
    assert fc.forecast_status == C.BAD_DATA and not fc.has_range and "stale" in fc.status_reasons[0]


def test_market_closed(trained_home, live_day):
    root, _ = trained_home
    day, last_close = live_day
    eng = _engine(root, http=FakeHTTP(_payloads(day, last_close, ist("2026-02-09 15:29"))))
    fc = eng.predict("CIPLA", now=ist("2026-02-09 19:00").to_pydatetime(), use_context=False)
    assert fc.forecast_status == C.MARKET_CLOSED and not fc.has_range and fc.regime


def test_network_failure_is_bad_data(trained_home):
    root, _ = trained_home
    eng = _engine(root, {"data_source": {"retries": 0}}, http=FakeHTTP(exc=requests.Timeout()))
    fc = eng.predict("CIPLA", now=ist("2026-02-10 11:00").to_pydatetime())
    assert fc.forecast_status == C.BAD_DATA and "timed out" in fc.status_reasons[0]


def test_malformed_candles_are_bad_data(trained_home, live_day):
    root, _ = trained_home
    day, last_close = live_day
    p = _payloads(day, last_close, ist("2026-02-09 11:29"))
    recs = p["CIPLA"]["candles_1m"]
    for r in recs[::3]:
        r["high"] = "corrupt"
    eng = _engine(root, http=FakeHTTP(p))
    fc = eng.predict("CIPLA", now=ist("2026-02-09 11:30:10").to_pydatetime(), use_context=False)
    assert fc.forecast_status == C.BAD_DATA and fc.data_quality == "BAD"


def test_warming_up_without_model(tmp_path, market, live_day):
    day, last_close = live_day
    cfg = load_config(SMALL, root=tmp_path)
    eng = RangeEngine(cfg, http=FakeHTTP(_payloads(day, last_close, ist("2026-02-09 11:29"))))
    fc = eng.predict("CIPLA", now=ist("2026-02-09 11:30:10").to_pydatetime(), use_context=False)
    assert fc.forecast_status == C.WARMING_UP and not fc.has_range
    assert fc.regime is not None  # state is still described


def test_late_session_refuses(trained_home, live_day):
    root, _ = trained_home
    day, last_close = live_day
    eng = _engine(root, http=FakeHTTP(_payloads(day, last_close, ist("2026-02-09 15:22"))))
    fc = eng.predict("CIPLA", now=ist("2026-02-09 15:23:05").to_pydatetime(), use_context=False)
    assert fc.forecast_status == C.NO_EDGE and "minutes left" in fc.status_reasons[0]


def test_replay_mode_reports_actual_and_in_sample(trained_home):
    root, _ = trained_home
    eng = _engine(root, {"confidence": {"min_cell_sessions": 0, "medium_coverage_tolerance": 1.0, "min_skill": -10,
                                        "symbol_min_sessions": 0}})
    eng.pipeline()[0].domain = {}
    sessions = HistoryStore(eng.cfg).sessions("INFY")
    fc = eng.predict("INFY", as_of=f"{sessions[-2]} 11:00")
    assert fc.mode == "REPLAY" and fc.actual is not None and "inside_range" in fc.actual
    assert any("IN-SAMPLE" in r for r in fc.status_reasons)


def test_unknown_symbol(trained_home):
    root, _ = trained_home
    with pytest.raises(KeyError):
        _engine(root).predict("NOTASTOCK")


def test_decide_rules():
    table = {"levels": {"probable": 0.8}, "cells": {"RANGE|MIDDAY": {"sessions": 30, "coverage_probable": 0.81, "skill": 0.08}},
             "regimes": {}, "symbols": {"CIPLA": {"sessions": 30, "coverage_probable": 0.79}}}
    cc = load_config()["confidence"]
    kw = dict(table=table, regime="RANGE", tod="MIDDAY", symbol="CIPLA", cfg_conf=cc, data_quality="GOOD",
              out_of_domain=[], context_ok=True)
    assert C.decide(**kw).status == C.HIGH
    assert C.decide(**{**kw, "data_quality": "DEGRADED"}).status == C.MEDIUM
    assert C.decide(**{**kw, "out_of_domain": ["vol_z"]}).status == C.NO_EDGE
    assert C.decide(**{**kw, "context_ok": False}).status == C.NO_EDGE
    assert C.decide(**{**kw, "regime": "BREAKOUT"}).status == C.NO_EDGE
    bad = {**table, "cells": {"RANGE|MIDDAY": {"sessions": 30, "coverage_probable": 0.6, "skill": 0.08}}}
    assert C.decide(**{**kw, "table": bad}).status == C.NO_EDGE
    noskill = {**table, "cells": {"RANGE|MIDDAY": {"sessions": 30, "coverage_probable": 0.8, "skill": -0.01}}}
    assert C.decide(**{**kw, "table": noskill}).status == C.NO_EDGE


def test_wrapper_scripts_are_thin():
    for name in ("cipla.py", "tcs.py", "infy.py", "reliance.py", "range.py"):
        src = (ROOT / name).read_text()
        assert "engine.cli" in src and len(src.splitlines()) < 12


def test_cli_unknown_symbol_exit_code(tmp_path):
    env = {"PSYGRID_HOME": str(tmp_path), "PATH": "", "SYSTEMROOT": ""}
    out = subprocess.run([sys.executable, str(ROOT / "range.py"), "XYZ", "--no-context"],
                         capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    assert out.returncode == 2 and "not in the NIFTY50 universe" in out.stderr
