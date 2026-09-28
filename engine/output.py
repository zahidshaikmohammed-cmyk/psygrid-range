"""Human-readable and machine-readable rendering of a :class:`Forecast`."""
from __future__ import annotations

import json
import sys

import pandas as pd

from .confidence import FORECAST_STATUSES
from .forecasting import REMAINING, Forecast

W = 60
RULE = "=" * W
THIN = "-" * W


def _money(x: float | None, sym: str) -> str:
    return "n/a" if x is None else f"{sym}{x:,.2f}"


def _clock(iso: str | None, fmt: str = "%Y-%m-%d %H:%M:%S IST") -> str:
    if not iso:
        return "n/a"
    return pd.Timestamp(iso).strftime(fmt)


def currency_symbol() -> str:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    try:
        "₹".encode(enc or "ascii")
        return "₹"
    except (LookupError, UnicodeEncodeError):
        return "Rs."


def _kv(label: str, value) -> str:
    return f"{label:<20}: {value}"


def render_text(fc: Forecast, *, currency: str | None = None, verbose: bool = False) -> str:
    cur = currency or currency_symbol()
    L = [RULE, "PSYGRID 1-HOUR RANGE ENGINE".center(W).rstrip(), RULE, ""]
    L.append(_kv("SYMBOL", fc.symbol))
    L.append(_kv("CURRENT PRICE", _money(fc.current_price, cur)))
    L.append(_kv("DATA TIME", _clock(fc.timestamp) + (f"  (last complete candle {_clock(fc.last_candle, '%H:%M')})"
                                                       if fc.last_candle else "")))
    fr = fc.freshness or {}
    if fr.get("now"):
        L.append(_kv("SYSTEM TIME", _clock(fr["now"])))
    if fr.get("age_minutes") is not None:
        src = " (cached)" if fr.get("from_cache") else ""
        L.append(_kv("DATA AGE", f"{fr['age_minutes']:.1f} min after candle close{src}"))
    if fc.mode != "LIVE":
        tag = {"REPLAY": f"REPLAY - {fc.replay_sample or 'NO MODEL'} (not a live forecast)",
               "ANALYSIS": "ANALYSIS (market closed; not a live forecast)"}.get(fc.mode, fc.mode)
        L.append(_kv("MODE", tag))
    L += ["", "FORECAST HORIZON"]
    if fc.forecast_mode == REMAINING and fc.horizon_minutes:
        L.append(f"REMAINING SESSION ONLY: NEXT {fc.horizon_minutes} MINUTES (to the close) - not a 60-minute forecast")
    elif fc.horizon_minutes:
        L.append(f"NEXT {fc.horizon_minutes} MINUTES" if fc.horizon_minutes == 60
                 else f"NEXT 60 MINUTES REQUESTED - only {fc.horizon_minutes} remain in the session")
    else:
        L.append("n/a")
    L += ["", THIN]
    lv = fc.coverage_levels or {}
    if fc.has_range and fc.forecast_status in FORECAST_STATUSES:
        L.append(f"PROBABLE RANGE  (calibration target: {lv.get('probable', 0.8):.0%} of held-out hours stayed fully inside)")
        L.append(THIN)
        L.append(_kv("LOW", _money(fc.range["low"], cur)))
        L.append(_kv("HIGH", _money(fc.range["high"], cur)))
        L.append("")
        L.append(f"CENTRAL RANGE   (calibration target: {lv.get('central', 0.5):.0%})")
        L.append(_kv("LOW", _money(fc.central_range["low"], cur)))
        L.append(_kv("HIGH", _money(fc.central_range["high"], cur)))
        L.append("")
        L.append(_kv("UPSIDE EXTENSION", f"{fc.upside_pct:+.2f}%"))
        L.append(_kv("DOWNSIDE EXTENSION", f"{fc.downside_pct:+.2f}%"))
        L.append("(coverage targets are calibration properties, not accuracy or hit-rate claims)")
    else:
        L.append("NO RELIABLE FORECAST / INSUFFICIENT EDGE")
    L += [THIN, ""]
    L.append(_kv("MARKET REGIME", fc.regime or "n/a"))
    L.append(_kv("VOLATILITY STATE", fc.volatility_state or "n/a"))
    L.append(_kv("STRUCTURE", fc.structure or "n/a"))
    L.append(_kv("RELATIVE STRENGTH", (fc.relative_strength or "n/a")
                 + (" (vs NIFTY50_PROXY)" if fc.relative_strength not in (None, "UNAVAILABLE")
                    and fc.market_context.get("source") == "NIFTY50_PROXY" else "")))
    L.append(_kv("FORECAST STATUS", fc.forecast_status))
    for r in fc.status_reasons:
        L.append(f"  - {r}")
    if fc.explanation:
        e = fc.explanation
        L += ["", THIN, "STATE (inputs the model saw; descriptive, not trading rules)"]
        L.append(_kv("Volatility", e.get("volatility")))
        L.append(_kv("Relative volume", e.get("relative_volume")))
        L.append(_kv("Trend structure", e.get("trend_structure")))
        L.append(_kv("Relative strength", e.get("relative_strength")))
        L.append(_kv("Market", e.get("market")))
        L.append(_kv("Time regime", e.get("time_regime")))
    L += ["", THIN, "MODEL EVIDENCE"]
    me = fc.model_evidence or {}
    cs = (fc.evidence or {}).get("cell_stats")
    if me.get("training_rows"):
        L.append(_kv("Historical samples", f"{me['training_rows']:,} training rows, {me['training_sessions']} sessions"))
    if cs:
        L.append(_kv("Regime match", f"{fc.evidence.get('cell')}: {cs['n']:,} out-of-sample forecasts over "
                                    f"{cs['sessions']} sessions"))
        L.append(_kv("Out-of-sample", f"coverage {cs['coverage_probable']:.0%} (target {lv.get('probable', 0.8):.0%}), "
                                     f"skill {cs['skill']:+.1%} vs best baseline ({cs.get('best_baseline')})"))
    mc = fc.market_context or {}
    if mc.get("source") and mc.get("source") != "disabled":
        s = mc["source"]
        if "stocks_fetched" in mc:
            s += (f": {mc['stocks_fetched']} / {mc['universe_size']} stocks fetched, {mc['stocks_failed']} failed, "
                  f"{mc['stocks_stale']} stale")
        if mc.get("stocks_in_proxy_at_forecast_minute") is not None:
            s += f"; {mc['stocks_in_proxy_at_forecast_minute']} in proxy at forecast minute"
        L.append(_kv("Market context", s))
    L.append(_kv("Data quality", fc.data_quality or "n/a"))
    if me.get("data_provenance"):
        L.append(_kv("Training data", ", ".join(f"{k} ({v} sessions)" for k, v in me["data_provenance"].items())))
    L.append(_kv("Model version", (fc.model_version or "n/a")
                 + (f" (history to {fc.model_trained_through})" if fc.model_trained_through else "")))
    if fc.actual:
        a = fc.actual
        inside = a.get("inside_range")
        tag = "" if inside is None else ("  -> INSIDE probable range" if inside else "  -> OUTSIDE probable range")
        L.append(_kv("ACTUAL (replay)", f"low {_money(a['low'], cur)}  high {_money(a['high'], cur)}{tag}"))
    if verbose:
        for i in fc.data_issues:
            L.append(f"  data: {i}")
        if fc.quantiles:
            L.append("QUANTILES (next-hour high / low):")
            for k in fc.quantiles["upper_high"]:
                L.append(f"  {k:>4}: high {_money(fc.quantiles['upper_high'][k], cur)}   "
                         f"low {_money(fc.quantiles['lower_low'][k], cur)}")
        if mc:
            L.append(f"MARKET CONTEXT DETAIL: {mc}")
        if fr.get("raw_archive"):
            L.append(f"RAW RESPONSE SAVED  : {fr['raw_archive']}")
    L += ["", RULE]
    return "\n".join(L)


def _round(o, key=""):
    if isinstance(o, dict):
        return {k: _round(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, key) for v in o]
    if isinstance(o, float):
        return round(o, 3) if key.endswith("_pct") else round(o, 4)
    return o


def to_json_dict(fc: Forecast) -> dict:
    d = _round(fc.to_dict())
    ordered = {
        "symbol": d["symbol"], "timestamp": d["timestamp"], "current_price": d["current_price"],
        "horizon_minutes": d["horizon_minutes"], "forecast_mode": d["forecast_mode"],
        "range": d["range"], "central_range": d["central_range"],
        "upside_pct": d["upside_pct"], "downside_pct": d["downside_pct"], "regime": d["regime"],
        "volatility_state": d["volatility_state"], "forecast_status": d["forecast_status"],
        "data_quality": d["data_quality"], "model_version": d["model_version"],
    }
    ordered.update({k: v for k, v in d.items() if k not in ordered})
    return ordered


def render_json(fc: Forecast) -> str:
    return json.dumps(to_json_dict(fc), indent=2, ensure_ascii=False)
