"""Real-endpoint diagnostic (``python diagnose_endpoint.py CIPLA``).

Performs one genuine HTTP request from the local machine and reports exactly
what came back.  Nothing is filled in when the request fails: every value is
either measured from the response or shown as n/a, and the verdict is
BAD_DATA.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import EngineConfig
from .ingestion import DataSourceError, MalformedPayloadError, endpoint_url, fetch, parse_payload, save_raw_live
from .validation import assess_freshness, clean_candles


def run_diagnostic(symbol: str, cfg: EngineConfig, *, http=None, now: datetime | None = None,
                   save_fixture: Path | None = None) -> dict:
    now = now or datetime.now(cfg.tz)
    sym = cfg.universe.normalize(symbol)
    d: dict = {"symbol": sym, "url": endpoint_url(sym, cfg), "system_time": pd.Timestamp(now).isoformat(),
               "timezone": str(cfg.tz), "errors": [], "verdict": "BAD_DATA"}
    t0 = time.perf_counter()
    try:
        fr = fetch(sym, cfg, http=http)
    except DataSourceError as exc:
        d["response_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        d["errors"].append(f"{exc.__class__.__name__}: {exc}")
        return d
    d.update(http_status=fr.http_status, response_time_ms=round(fr.elapsed_ms, 1), response_bytes=fr.n_bytes,
             attempts=fr.attempts)
    if save_fixture:
        save_fixture.parent.mkdir(parents=True, exist_ok=True)
        save_fixture.write_text(json.dumps(fr.payload, indent=1), encoding="utf-8")
        d["fixture_saved"] = str(save_fixture)
    try:
        d["raw_saved"] = str(save_raw_live(sym, fr, now, cfg))
    except OSError as exc:
        d["errors"].append(f"could not archive raw response: {exc}")
    try:
        snap = parse_payload(fr.payload, sym, cfg, now)
    except MalformedPayloadError as exc:
        d["errors"].append(f"MalformedPayloadError: {exc}")
        if isinstance(fr.payload, dict):
            d["fields_present"] = sorted(map(str, fr.payload.keys()))[:40]
        return d
    d.update(service=snap.service, schema_version=snap.schema_version, endpoint_status=snap.endpoint_status,
             security_id=snap.security_id, previous_close=snap.previous_close, today_open=snap.today_open,
             records=snap.n_records, malformed_records=snap.n_malformed, malformed_examples=snap.malformed_examples,
             schema_issues=snap.schema_issues, timestamp_format=snap.timestamp_format)
    if isinstance(fr.payload, dict):
        d["fields_present"] = sorted(map(str, fr.payload.keys()))[:40]
    clean, rep = clean_candles(snap.candles, cfg, sym, n_malformed=snap.n_malformed, now=now,
                               schema_issues=snap.schema_issues, single_session=True)
    fresh = assess_freshness(clean, now, cfg)
    d.update(
        candles=rep.n_valid,
        first_candle=rep.first_candle, last_candle=rep.last_candle,
        latest_close=float(clean["close"].iloc[-1]) if not clean.empty else None,
        latest_candle_age_minutes=fresh.age_minutes,
        market_open=fresh.market_open, freshness=fresh.reason,
        duplicates_identical=rep.n_duplicates_identical, duplicates_conflicting=rep.n_duplicates_conflicting,
        out_of_order=rep.n_out_of_order, ohlc_violations=rep.n_ohlc_violations,
        outside_session=rep.n_outside_session, future_candles=rep.n_future, forming_dropped=rep.n_forming_dropped,
        missing_minutes=rep.missing_minutes, largest_gap_minutes=rep.largest_gap_minutes,
        data_quality=rep.quality, issues=rep.issues,
    )
    if rep.quality == "BAD":
        d["verdict"] = "BAD_DATA"
    elif fresh.market_open and fresh.is_stale:
        d["verdict"] = "BAD_DATA (stale)"
    else:
        d["verdict"] = rep.quality + ("" if fresh.market_open else " (market closed: freshness not applicable)")
    return d


def render_diagnostic(d: dict) -> str:
    def v(k, fmt=None):
        x = d.get(k)
        if x is None:
            return "n/a"
        return fmt.format(x) if fmt else str(x)

    dup = "n/a" if "duplicates_identical" not in d else \
        f"{d['duplicates_identical'] + d['duplicates_conflicting']} ({d['duplicates_conflicting']} conflicting)"
    rows = [
        ("Symbol", v("symbol")), ("URL", v("url")), ("HTTP Status", v("http_status")),
        ("Response Time", v("response_time_ms", "{:.0f} ms")), ("Response Size", v("response_bytes", "{:,} bytes")),
        ("Service", v("service")), ("Schema", v("schema_version")), ("Endpoint Status", v("endpoint_status")),
        ("Security ID", v("security_id")), ("Previous Close", v("previous_close")), ("Today Open", v("today_open")),
        ("Records", v("records")), ("Candles (valid)", v("candles")), ("Malformed Records", v("malformed_records")),
        ("First Candle", v("first_candle")), ("Last Candle", v("last_candle")), ("Latest Price", v("latest_close")),
        ("System Time", v("system_time")),
        ("Latest Candle Age", v("latest_candle_age_minutes", "{:.1f} min after candle close")),
        ("Freshness", v("freshness")), ("Duplicate Candles", dup), ("Out-of-order", v("out_of_order")),
        ("Impossible OHLC", v("ohlc_violations")), ("Outside Session", v("outside_session")),
        ("Future-dated", v("future_candles")), ("Missing Minutes", v("missing_minutes")),
        ("Largest Gap", v("largest_gap_minutes", "{} min")), ("Timestamp Format", v("timestamp_format")),
        ("Timezone", v("timezone") + " (all timestamps converted to IST)"),
        ("Data Quality", v("data_quality")), ("VERDICT", v("verdict")),
    ]
    out = ["PSYGRID ENDPOINT DIAGNOSTIC", "=" * 28, ""] + [f"{k:<20}: {val}" for k, val in rows]
    for label, key in (("Schema issues", "schema_issues"), ("Validation issues", "issues"), ("ERRORS", "errors")):
        if d.get(key):
            out.append(f"{label}:")
            out += [f"  - {x}" for x in d[key]]
    if d.get("malformed_examples"):
        out.append("Malformed examples:")
        out += [f"  - {x}" for x in d["malformed_examples"]]
    if d.get("fields_present"):
        out.append(f"Fields present: {', '.join(d['fields_present'])}")
    for k in ("raw_saved", "fixture_saved"):
        if d.get(k):
            out.append(f"{k.replace('_', ' ').title():<20}: {d[k]}")
    return "\n".join(out)
