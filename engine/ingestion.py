"""PHASE 1 - Data ingestion from the PSYGRID public stock endpoints.

Responsibilities
  * HTTP fetch with separate connect/read timeouts, bounded retries with
    exponential back-off (and bounded ``Retry-After`` on HTTP 429), and a
    response-size cap.  Nothing ever waits indefinitely.
  * Schema validation of the documented envelope
    (service / schema_version / status / symbol / security_id /
    previous_close / today_open / candles_1m).  Hard violations raise;
    softer ones are recorded as ``schema_issues``.
  * Parsing of ``candles_1m`` into a typed frame with explicit IST timestamps.
    Records that cannot be parsed are *dropped and counted*, never repaired.
  * Raw-response archiving (``data/raw/live/<SYMBOL>_<YYYYMMDD>.json``).
  * A short-lived on-disk cache; data older than the TTL is never reused.

Endpoint content is untrusted input: it is only ever parsed as JSON data and
converted to numbers/timestamps; it is never executed or passed to a shell.
Nothing here invents candles or fills gaps; see :mod:`engine.validation`.
"""
from __future__ import annotations

import json
import math
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from .config import EngineConfig

CANDLE_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")
OK_STATUSES = ("ok", "success", "live", "200", "true")


class DataSourceError(RuntimeError):
    """Base class for all ingestion failures."""


class NetworkError(DataSourceError):
    """Timeout, connection failure or non-200 response."""


class MalformedPayloadError(DataSourceError):
    """The endpoint answered, but not with a usable PSYGRID stock document."""


@dataclass
class FetchResult:
    url: str
    http_status: int
    elapsed_ms: float
    n_bytes: int | None
    payload: Any
    attempts: int


@dataclass
class Snapshot:
    """One parsed PSYGRID stock document."""

    symbol: str
    candles: pd.DataFrame  # columns: timestamp(IST tz-aware), open, high, low, close, volume
    fetched_at: datetime
    previous_close: float | None = None
    today_open: float | None = None
    security_id: str | None = None
    schema_version: str | None = None
    service: str | None = None
    endpoint_status: str | None = None
    n_records: int = 0
    n_malformed: int = 0
    malformed_examples: list[str] = field(default_factory=list)
    schema_issues: list[str] = field(default_factory=list)
    timestamp_format: str = "unknown"
    from_cache: bool = False
    fetch: FetchResult | None = None
    raw_path: str | None = None


# --------------------------------------------------------------------------- #
# Timestamp parsing
# --------------------------------------------------------------------------- #
def classify_timestamp(value: Any) -> str:
    """Describe the wire format of one timestamp (for diagnostics)."""
    if isinstance(value, bool) or value is None:
        return "invalid"
    if isinstance(value, (int, float, np.integer, np.floating)) or (
        isinstance(value, str) and value.strip().lstrip("-").replace(".", "", 1).isdigit()
    ):
        try:
            x = float(value)
        except ValueError:
            return "invalid"
        return "epoch_milliseconds" if x > 1e11 else "epoch_seconds"
    if isinstance(value, str):
        v = value.strip()
        if v.endswith("Z") or v.endswith("z"):
            return "iso8601_utc"
        tail = v[10:]
        if "+" in tail or (tail.count("-") >= 1 and ":" in tail.split("-")[-1]):
            return "iso8601_with_offset"
        return "iso8601_naive(assumed IST)"
    return "invalid"


def parse_timestamp(value: Any, tz) -> pd.Timestamp | None:
    """Parse one candle timestamp into a tz-aware IST ``Timestamp``.

    Accepts epoch seconds, epoch milliseconds (numbers or numeric strings) and
    ISO-8601-like strings.  Naive strings are interpreted as IST, which is the
    exchange's timezone.  Returns ``None`` when the value is unusable.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)) or (
            isinstance(value, str) and value.strip().lstrip("-").replace(".", "", 1).isdigit()
        ):
            x = float(value)
            if not math.isfinite(x) or x <= 0:
                return None
            if x > 1e14:  # microseconds / nanoseconds are not expected
                return None
            unit = "ms" if x > 1e11 else "s"
            if x < 1e8:
                return None
            return pd.Timestamp(x, unit=unit, tz="UTC").tz_convert(tz)
        if isinstance(value, str):
            if len(value) > 64:
                return None
            ts = pd.Timestamp(value.strip())
            if ts is pd.NaT:
                return None
            if ts.tzinfo is None:
                return ts.tz_localize(tz)
            return ts.tz_convert(tz)
    except (ValueError, TypeError, OverflowError):
        return None
    return None


def _to_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_candles(records: Any, tz) -> tuple[pd.DataFrame, int, list[str]]:
    """Parse ``candles_1m``.  Supports a list of objects
    (``{"timestamp":..,"open":..}``) or a list of arrays
    (``[timestamp, open, high, low, close, volume]``).

    Returns (frame, n_malformed, examples_of_malformed).
    """
    frame, bad, examples, _ = _parse_candles(records, tz)
    return frame, bad, examples


def _parse_candles(records: Any, tz) -> tuple[pd.DataFrame, int, list[str], str]:
    if not isinstance(records, list):
        raise MalformedPayloadError("candles_1m is not a list")
    rows: list[tuple] = []
    bad = 0
    examples: list[str] = []
    formats: Counter = Counter()
    for rec in records:
        vals: list[Any] | None = None
        if isinstance(rec, dict):
            keymap = {str(k).lower(): v for k, v in rec.items()}
            if "timestamp" not in keymap:
                for alt in ("ts", "time", "datetime", "date"):
                    if alt in keymap:
                        keymap["timestamp"] = keymap[alt]
                        break
            if "volume" not in keymap and "vol" in keymap:
                keymap["volume"] = keymap["vol"]
            vals = [keymap.get(f) for f in CANDLE_FIELDS]
        elif isinstance(rec, (list, tuple)) and len(rec) >= 6:
            vals = list(rec[:6])
        if vals is None:
            bad += 1
            if len(examples) < 3:
                examples.append(repr(rec)[:120])
            continue
        formats[classify_timestamp(vals[0])] += 1
        ts = parse_timestamp(vals[0], tz)
        nums = [_to_float(v) for v in vals[1:]]
        if ts is None or any(n is None for n in nums):
            bad += 1
            if len(examples) < 3:
                examples.append(repr(rec)[:120])
            continue
        rows.append((ts, *nums))
    frame = pd.DataFrame(rows, columns=list(CANDLE_FIELDS))
    if frame.empty:
        frame = pd.DataFrame(
            {
                "timestamp": pd.Series([], dtype=f"datetime64[ns, {tz}]"),
                **{c: pd.Series([], dtype=float) for c in CANDLE_FIELDS[1:]},
            }
        )
    else:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_convert(tz)
        for c in CANDLE_FIELDS[1:]:
            frame[c] = frame[c].astype(float)
    valid_formats = Counter({k: v for k, v in formats.items() if k != "invalid"})
    if not valid_formats:
        fmt = "unknown"
    elif len(valid_formats) == 1:
        fmt = next(iter(valid_formats))
    else:
        fmt = "MIXED: " + ", ".join(f"{k}={v}" for k, v in valid_formats.most_common())
    return frame, bad, examples, fmt


def parse_payload(payload: Any, symbol: str, cfg: EngineConfig, fetched_at: datetime) -> Snapshot:
    """Validate the envelope and parse a PSYGRID stock document."""
    ds = cfg["data_source"]
    if not isinstance(payload, dict):
        raise MalformedPayloadError("payload is not a JSON object")
    missing_req = [f for f in ds.get("required_fields", ["candles_1m"]) if f not in payload]
    if missing_req:
        raise MalformedPayloadError(f"payload is missing required field(s): {', '.join(missing_req)}")
    issues: list[str] = []
    missing = [f for f in ds.get("expected_fields", []) if f not in payload]
    if missing:
        issues.append(f"missing documented field(s): {', '.join(missing)}")
    status = payload.get("status")
    if status is not None and str(status).strip().lower() not in OK_STATUSES:
        raise MalformedPayloadError(f"endpoint status is {status!r}")
    psym = payload.get("symbol")
    if psym is not None and str(psym).strip().upper() != symbol.upper():
        raise MalformedPayloadError(f"payload symbol {psym!r} does not match requested {symbol!r}")
    service = payload.get("service")
    exp_service = ds.get("expected_service")
    if exp_service and service != exp_service:
        issues.append(f"service is {service!r}, expected {exp_service!r}")
    if "schema_version" in payload and payload.get("schema_version") in (None, ""):
        issues.append("schema_version is empty")
    for k in ("previous_close", "today_open"):
        if payload.get(k) is not None and _to_float(payload.get(k)) is None:
            issues.append(f"{k} is not a finite number: {str(payload.get(k))[:40]!r}")
        elif _to_float(payload.get(k)) is not None and _to_float(payload.get(k)) <= 0:
            issues.append(f"{k} is not positive")
    candles, bad, examples, fmt = _parse_candles(payload["candles_1m"], cfg.tz)
    return Snapshot(
        symbol=symbol.upper(),
        candles=candles,
        fetched_at=fetched_at,
        previous_close=_to_float(payload.get("previous_close")),
        today_open=_to_float(payload.get("today_open")),
        security_id=None if payload.get("security_id") is None else str(payload.get("security_id"))[:64],
        schema_version=None if payload.get("schema_version") is None else str(payload.get("schema_version"))[:32],
        service=None if service is None else str(service)[:64],
        endpoint_status=None if status is None else str(status)[:32],
        n_records=len(payload["candles_1m"]) if isinstance(payload["candles_1m"], list) else 0,
        n_malformed=bad,
        malformed_examples=examples,
        schema_issues=issues,
        timestamp_format=fmt,
    )


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def endpoint_url(symbol: str, cfg: EngineConfig) -> str:
    ds = cfg["data_source"]
    base = str(ds["base_url"]).rstrip("/")
    return base + ds["stock_path"].format(symbol=quote(symbol.upper(), safe=""))


def fetch(
    symbol: str,
    cfg: EngineConfig,
    http=None,
    *,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    retries: int | None = None,
) -> FetchResult:
    """One HTTP GET with bounded retries.  Raises NetworkError /
    MalformedPayloadError; never blocks longer than
    ``(retries+1) * (connect+read) + back-off``."""
    ds = cfg["data_source"]
    url = endpoint_url(symbol, cfg)
    retries = int(ds.get("retries", 1)) if retries is None else int(retries)
    backoff = float(ds.get("backoff_seconds", 1.0))
    ct = float(ds.get("connect_timeout_seconds", 5)) if connect_timeout is None else connect_timeout
    rt = float(ds.get("read_timeout_seconds", 10)) if read_timeout is None else read_timeout
    max_bytes = int(ds.get("max_response_bytes", 20_000_000))
    getter = http.get if http is not None else requests.get
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        delay = backoff * (2**attempt)
        t0 = _time.perf_counter()
        try:
            resp = getter(url, timeout=(ct, rt), headers={"User-Agent": ds.get("user_agent", "psygrid-range"),
                                                          "Accept": "application/json"})
            elapsed = (_time.perf_counter() - t0) * 1000
            code = int(resp.status_code)
            if code == 404:
                raise NetworkError(f"{symbol}: endpoint returned HTTP 404 (unknown symbol?) {url}")
            if code == 429 or code >= 500:
                ra = _to_float(getattr(resp, "headers", {}).get("Retry-After")) if hasattr(resp, "headers") else None
                if ra is not None:
                    delay = min(max(ra, 0.0), float(ds.get("max_retry_after_seconds", 10)))
                raise _Retryable(f"{symbol}: HTTP {code} from {url}")
            if code != 200:
                raise NetworkError(f"{symbol}: HTTP {code} from {url}")
            content = getattr(resp, "content", None)
            n_bytes = len(content) if isinstance(content, (bytes, bytearray)) else None
            if n_bytes is not None and n_bytes > max_bytes:
                raise MalformedPayloadError(f"{symbol}: response is {n_bytes:,} bytes (limit {max_bytes:,})")
            try:
                payload = resp.json()
            except ValueError as exc:
                raise MalformedPayloadError(f"{symbol}: response is not valid JSON") from exc
            return FetchResult(url, code, elapsed, n_bytes, payload, attempt + 1)
        except (MalformedPayloadError, NetworkError):
            raise
        except _Retryable as exc:
            last_err = NetworkError(str(exc))
        except requests.ConnectTimeout as exc:
            last_err = NetworkError(f"{symbol}: connection timed out after {ct:.0f}s ({url})")
            last_err.__cause__ = exc
        except requests.Timeout as exc:
            last_err = NetworkError(f"{symbol}: timed out (connect {ct:.0f}s / read {rt:.0f}s) ({url})")
            last_err.__cause__ = exc
        except requests.RequestException as exc:
            last_err = NetworkError(f"{symbol}: network error: {exc.__class__.__name__}: {exc}")
        if attempt < retries:
            _time.sleep(delay)
    assert last_err is not None
    raise last_err


class _Retryable(Exception):
    pass


# --------------------------------------------------------------------------- #
# Raw archive + cache
# --------------------------------------------------------------------------- #
def save_raw_live(snap_symbol: str, fr: FetchResult, fetched_at: datetime, cfg: EngineConfig) -> Path:
    """Archive the raw endpoint document for auditing.

    ``data/raw/live/<SYMBOL>_<YYYYMMDD>.json`` holds the most complete capture
    of the day.  A later capture with *fewer* candles (e.g. an endpoint reset)
    never overwrites it; it is written next to it with a time suffix instead.
    """
    d = cfg.paths.raw_live
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{quote(snap_symbol.upper(), safe='')}_{fetched_at.strftime('%Y%m%d')}"
    path = d / f"{stem}.json"
    n_new = len(fr.payload.get("candles_1m", [])) if isinstance(fr.payload, dict) and isinstance(fr.payload.get("candles_1m"), list) else 0
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            n_old = len(old.get("payload", {}).get("candles_1m", []))
        except (ValueError, OSError, AttributeError):
            n_old = -1
        if n_old > n_new:
            path = d / f"{stem}_{fetched_at.strftime('%H%M%S')}.json"
    blob = {"fetched_at": fetched_at.isoformat(), "url": fr.url, "http_status": fr.http_status,
            "elapsed_ms": round(fr.elapsed_ms, 1), "payload": fr.payload}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob), encoding="utf-8")
    tmp.replace(path)
    return path


def _cache_file(symbol: str, cfg: EngineConfig) -> Path:
    safe = quote(symbol.upper(), safe="")
    return cfg.paths.cache / f"{safe}.json"


def get_snapshot(
    symbol: str,
    cfg: EngineConfig,
    *,
    now: datetime | None = None,
    use_cache: bool = True,
    http=None,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    retries: int | None = None,
    save_raw: bool | None = None,
) -> Snapshot:
    """Fetch (or re-use a fresh cached copy of) the live PSYGRID document."""
    now = now or datetime.now(cfg.tz)
    ttl = float(cfg["data_source"].get("cache_ttl_seconds", 20))
    cache = _cache_file(symbol, cfg)
    if use_cache and cache.exists():
        try:
            blob = json.loads(cache.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(blob["fetched_at"])
            age = (now - fetched_at).total_seconds()
            if 0 <= age <= ttl:
                snap = parse_payload(blob["payload"], symbol, cfg, fetched_at)
                snap.from_cache = True
                return snap
        except (ValueError, KeyError, OSError, MalformedPayloadError):
            pass  # unreadable cache -> refetch
    fr = fetch(symbol, cfg, http=http, connect_timeout=connect_timeout, read_timeout=read_timeout, retries=retries)
    fetched_at = now
    raw_path = None
    if cfg["data_source"].get("save_raw_live", True) if save_raw is None else save_raw:
        try:
            raw_path = str(save_raw_live(symbol, fr, fetched_at, cfg))
        except OSError:
            raw_path = None
    snap = parse_payload(fr.payload, symbol, cfg, fetched_at)
    snap.fetch, snap.raw_path = fr, raw_path
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"fetched_at": fetched_at.isoformat(), "payload": fr.payload}), encoding="utf-8")
    except OSError:
        pass
    return snap
