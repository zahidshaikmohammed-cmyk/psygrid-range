"""Human-readable and machine-readable rendering of a :class:`Forecast`."""
from __future__ import annotations

import json
import sys

import pandas as pd

from .confidence import FORECAST_STATUSES
from .forecasting import Forecast

RULE = "=" * 60


def _money(x: float | None, sym: str) -> str:
    return "n/a" if x is None else f"{sym}{x:,.2f}"


def _ts(iso: str | None) -> str:
    if not iso:
        return "n/a"
    return pd.Timestamp(iso).strftime("%Y-%m-%d %H:%M IST")


def currency_symbol() -> str:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    try:
        "₹".encode(enc or "ascii")
        return "₹"
    except (LookupError, UnicodeEncodeError):
        return "Rs."


def render_text(fc: Forecast, *, currency: str | None = None, verbose: bool = False) -> str:
    cur = currency or currency_symbol()
    L = [RULE, "PSYGRID 1-HOUR RANGE ENGINE", RULE, ""]
    L.append(f"SYMBOL       : {fc.symbol}")
    L.append(f"PRICE        : {_money(fc.current_price, cur)}")
    L.append(f"TIMESTAMP    : {_ts(fc.timestamp)}")
    if fc.mode != "LIVE":
        L.append(f"MODE         : {fc.mode}" + ("  (not a live forecast)" if fc.mode in ("ANALYSIS", "REPLAY") else ""))
    fr = fc.freshness or {}
    if fr.get("age_minutes") is not None:
        src = " (cached)" if fr.get("from_cache") else ""
        L.append(f"DATA AGE     : last candle closed {fr['age_minutes']:.1f} min ago{src}")
    L.append("")
    L.append("FORECAST HORIZON")
    if fc.horizon_minutes:
        extra = "" if fc.horizon_minutes == 60 else f"  (truncated at the close: {fc.horizon_minutes} min)"
        L.append(f"NEXT {fc.horizon_minutes} MINUTES{extra}")
    else:
        L.append("n/a")
    L.append("")
    lv = fc.coverage_levels or {}
    if fc.has_range and fc.forecast_status in FORECAST_STATUSES:
        L.append(f"PROBABLE RANGE  (calibrated for {lv.get('probable', 0.8):.0%} of hours to stay inside)")
        L.append(f"LOW          : {_money(fc.range['low'], cur)}")
        L.append(f"HIGH         : {_money(fc.range['high'], cur)}")
        L.append("")
        L.append(f"CENTRAL RANGE   (calibrated for {lv.get('central', 0.5):.0%})")
        L.append(f"LOW          : {_money(fc.central_range['low'], cur)}")
        L.append(f"HIGH         : {_money(fc.central_range['high'], cur)}")
        L.append("")
        L.append(f"UPSIDE EXTENSION   : {fc.upside_pct:+.2f}%")
        L.append(f"DOWNSIDE EXTENSION : {fc.downside_pct:+.2f}%")
    else:
        L.append("NO RELIABLE FORECAST / INSUFFICIENT EDGE")
    L.append("")
    L.append(f"MARKET REGIME      : {fc.regime or 'n/a'}")
    L.append(f"VOLATILITY STATE   : {fc.volatility_state or 'n/a'}")
    L.append(f"STRUCTURE          : {fc.structure or 'n/a'}")
    L.append(f"RELATIVE STRENGTH  : {fc.relative_strength or 'n/a'}")
    L.append("")
    L.append(f"FORECAST STATUS    : {fc.forecast_status}")
    for r in fc.status_reasons:
        L.append(f"  - {r}")
    cs = (fc.evidence or {}).get("cell_stats")
    if cs:
        L.append(
            f"EVIDENCE           : {cs['sessions']} out-of-sample sessions in this state, "
            f"{cs['coverage_probable']:.0%} realised coverage, skill {cs['skill']:+.1%} vs baseline"
        )
    L.append(f"DATA QUALITY       : {fc.data_quality or 'n/a'}")
    if fc.actual:
        a = fc.actual
        inside = a.get("inside_range")
        tag = "" if inside is None else ("  -> INSIDE probable range" if inside else "  -> OUTSIDE probable range")
        L.append(f"ACTUAL (replay)    : low {_money(a['low'], cur)}  high {_money(a['high'], cur)}{tag}")
    if verbose:
        for i in fc.data_issues:
            L.append(f"  data: {i}")
        if fc.quantiles:
            L.append("QUANTILES (next-hour high / low):")
            for k in fc.quantiles["upper_high"]:
                L.append(f"  {k:>4}: high {_money(fc.quantiles['upper_high'][k], cur)}   low {_money(fc.quantiles['lower_low'][k], cur)}")
        if fc.market_context:
            L.append(f"MARKET CONTEXT     : {fc.market_context}")
    L.append(f"MODEL VERSION      : {fc.model_version}" + (f" (history to {fc.model_trained_through})" if fc.model_trained_through else ""))
    L.append("")
    L.append(RULE)
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
        "horizon_minutes": d["horizon_minutes"], "range": d["range"], "central_range": d["central_range"],
        "upside_pct": d["upside_pct"], "downside_pct": d["downside_pct"], "regime": d["regime"],
        "volatility_state": d["volatility_state"], "forecast_status": d["forecast_status"],
        "data_quality": d["data_quality"], "model_version": d["model_version"],
    }
    ordered.update({k: v for k, v in d.items() if k not in ordered})
    return ordered


def render_json(fc: Forecast) -> str:
    return json.dumps(to_json_dict(fc), indent=2, ensure_ascii=False)
