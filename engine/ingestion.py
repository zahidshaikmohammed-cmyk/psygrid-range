"""PHASE 1 - Data ingestion from the PSYGRID public stock endpoints.

Responsibilities
  * HTTP fetch with timeout, bounded retries and exponential back-off.
  * Structural validation of the payload envelope (service/status/symbol/...).
  * Parsing of ``candles_1m`` into a typed frame with explicit IST timestamps.
    Records that cannot be parsed are *dropped and counted*, never repaired.
  * A short-lived on-disk cache so repeated invocations inside the same minute
    do not hammer the server.  Cached data older than the TTL is never used.

Nothing here invents candles or fills gaps; see :mod:`engine.validation`.
"""
from __future__ import annotations

import json
import math
import time as _time
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


class DataSourceError(RuntimeError):
    """Base class for all ingestion failures."""


class NetworkError(DataSourceError):
    """Timeout, connection failure or non-200 response."""


class MalformedPayloadError(DataSourceError):
    """The endpoint answered, but not with a usable PSYGRID stock document."""


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
    n_records: int = 0
    n_malformed: int = 0
    malformed_examples: list[str] = field(default_factory=list)
    from_cache: bool = False


# --------------------------------------------------------------------------- #
# Timestamp parsing
# --------------------------------------------------------------------------- #
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
    if not isinstance(records, list):
        raise MalformedPayloadError("candles_1m is not a list")
    rows: list[tuple] = []
    bad = 0
    examples: list[str] = []
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
    return frame, bad, examples


def parse_payload(payload: Any, symbol: str, cfg: EngineConfig, fetched_at: datetime) -> Snapshot:
    """Validate the envelope and parse a PSYGRID stock document."""
    if not isinstance(payload, dict):
        raise MalformedPayloadError("payload is not a JSON object")
    status = payload.get("status")
    if status is not None and str(status).lower() not in ("ok", "success", "live", "200", "true"):
        raise MalformedPayloadError(f"endpoint status is {status!r}")
    psym = payload.get("symbol")
    if psym is not None and str(psym).strip().upper() != symbol.upper():
        raise MalformedPayloadError(f"payload symbol {psym!r} does not match requested {symbol!r}")
    if "candles_1m" not in payload:
        raise MalformedPayloadError("payload has no candles_1m field")
    candles, bad, examples = parse_candles(payload["candles_1m"], cfg.tz)
    return Snapshot(
        symbol=symbol.upper(),
        candles=candles,
        fetched_at=fetched_at,
        previous_close=_to_float(payload.get("previous_close")),
        today_open=_to_float(payload.get("today_open")),
        security_id=None if payload.get("security_id") is None else str(payload.get("security_id")),
        schema_version=None if payload.get("schema_version") is None else str(payload.get("schema_version")),
        service=payload.get("service"),
        n_records=len(payload["candles_1m"]) if isinstance(payload["candles_1m"], list) else 0,
        n_malformed=bad,
        malformed_examples=examples,
    )


# --------------------------------------------------------------------------- #
# HTTP + cache
# --------------------------------------------------------------------------- #
def endpoint_url(symbol: str, cfg: EngineConfig) -> str:
    return cfg["data_source"]["endpoint_template"].format(symbol=quote(symbol.upper(), safe=""))


def fetch_payload(
    symbol: str,
    cfg: EngineConfig,
    http: requests.Session | None = None,
    *,
    timeout: float | None = None,
    retries: int | None = None,
) -> Any:
    ds = cfg["data_source"]
    url = endpoint_url(symbol, cfg)
    retries = int(ds.get("retries", 2)) if retries is None else retries
    backoff = float(ds.get("backoff_seconds", 1.0))
    timeout = float(ds.get("timeout_seconds", 10)) if timeout is None else timeout
    getter = http.get if http is not None else requests.get
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = getter(url, timeout=timeout, headers={"User-Agent": ds.get("user_agent", "psygrid-range")})
            if resp.status_code == 404:
                raise NetworkError(f"{symbol}: endpoint returned 404 (unknown symbol?) {url}")
            if resp.status_code != 200:
                raise NetworkError(f"{symbol}: HTTP {resp.status_code} from {url}")
            try:
                return resp.json()
            except ValueError as exc:
                raise MalformedPayloadError(f"{symbol}: response is not valid JSON") from exc
        except MalformedPayloadError:
            raise
        except NetworkError as exc:
            last_err = exc
            if "404" in str(exc):
                raise
        except requests.Timeout as exc:
            last_err = NetworkError(f"{symbol}: timed out after {timeout:.0f}s ({url})")
            last_err.__cause__ = exc
        except requests.RequestException as exc:
            last_err = NetworkError(f"{symbol}: network error: {exc.__class__.__name__}: {exc}")
        if attempt < retries:
            _time.sleep(backoff * (2**attempt))
    assert last_err is not None
    raise last_err


def _cache_file(symbol: str, cfg: EngineConfig) -> Path:
    safe = quote(symbol.upper(), safe="")
    return cfg.paths.cache / f"{safe}.json"


def get_snapshot(
    symbol: str,
    cfg: EngineConfig,
    *,
    now: datetime | None = None,
    use_cache: bool = True,
    http: requests.Session | None = None,
    timeout: float | None = None,
    retries: int | None = None,
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
    payload = fetch_payload(symbol, cfg, http=http, timeout=timeout, retries=retries)
    fetched_at = now
    snap = parse_payload(payload, symbol, cfg, fetched_at)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"fetched_at": fetched_at.isoformat(), "payload": payload}), encoding="utf-8")
    except OSError:
        pass
    return snap
