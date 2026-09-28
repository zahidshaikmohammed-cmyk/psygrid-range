"""The engine: live data -> current state -> calibrated 60-minute range.

``RangeEngine().predict("CIPLA")`` is the single entry point used by every
CLI script.  Normal flow never trains: fetch -> validate -> features ->
load trained model -> predict -> calibrate -> gate -> output.

Modes
  LIVE      fetch PSYGRID now (default).
  ANALYSIS  market closed, ``allow_closed=True``: evaluate as of the last
            stored candle, clearly labelled as not a live forecast.
  REPLAY    ``as_of=...``: rebuild the state at a past minute from local
            history only (no network, nothing after ``as_of`` is used for
            features) and show what actually happened.  Labelled IN-SAMPLE
            when that session was used to train/calibrate the model.

Horizon
  The standard forecast is a true NEXT-60-MINUTE forecast.  When fewer than
  60 session minutes remain it is refused; ``remaining_session=True`` gives
  an explicitly labelled REMAINING_SESSION forecast instead.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from . import confidence as C
from .config import EngineConfig, load_config
from .dataset import build_symbol_frame, load_clean_history
from .features import INDEX_SOURCE, PROXY_SOURCE, TOD_BUCKETS, build_market_context, intraday_log_returns, to_minute_grid
from .ingestion import DataSourceError, get_snapshot
from .pipeline import FittedPipeline
from .regime import describe
from .storage import LIVE, HistoryStore
from .training import calibration_file, model_dir
from .validation import assess_freshness, clean_candles

STANDARD, REMAINING = "STANDARD_60M", "REMAINING_SESSION"
PROXY_DESCRIPTION = ("equal-weight average of NIFTY-50 constituent 1-minute returns built by this engine; "
                     "NOT the official (free-float cap-weighted) NIFTY 50 index")


@dataclass
class Forecast:
    symbol: str
    forecast_status: str
    mode: str = "LIVE"
    forecast_mode: str = STANDARD
    timestamp: str | None = None          # close time of the last complete candle (forecast origin)
    last_candle: str | None = None
    current_price: float | None = None
    horizon_minutes: int | None = None
    window_end: str | None = None
    range: dict | None = None             # probable range (nominal `probable` joint coverage)
    central_range: dict | None = None     # central range (nominal `central` joint coverage)
    upside_pct: float | None = None
    downside_pct: float | None = None
    quantiles: dict | None = None
    regime: str | None = None
    volatility_state: str | None = None
    structure: str | None = None
    relative_strength: str | None = None
    explanation: dict = field(default_factory=dict)
    status_reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    model_evidence: dict = field(default_factory=dict)
    data_quality: str | None = None
    data_issues: list[str] = field(default_factory=list)
    freshness: dict = field(default_factory=dict)
    market_context: dict = field(default_factory=dict)
    coverage_levels: dict = field(default_factory=dict)
    model_version: str | None = None
    model_trained_through: str | None = None
    replay_sample: str | None = None      # OUT-OF-SAMPLE / IN-SAMPLE (replay only)
    actual: dict | None = None
    history_sessions: int = 0

    @property
    def has_range(self) -> bool:
        return self.range is not None

    def to_dict(self) -> dict:
        return _clean(asdict(self))


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class RangeEngine:
    def __init__(self, cfg: EngineConfig | None = None, *, http=None):
        self.cfg = cfg or load_config()
        self.store = HistoryStore(self.cfg)
        self.http = http
        self._pipe: tuple[FittedPipeline, dict] | None = None
        self._pipe_loaded = False

    # ------------------------------------------------------------------ model
    def pipeline(self) -> tuple[FittedPipeline, dict] | None:
        if not self._pipe_loaded:
            self._pipe_loaded = True
            md, cf = model_dir(self.cfg), calibration_file(self.cfg)
            if (md / "bundle.json").exists() and cf.exists():
                self._pipe = FittedPipeline.load(md, cf)
        return self._pipe

    # ------------------------------------------------------------------ data
    def _live_today(self, symbol: str, now: datetime, **fetch_kw) -> tuple[pd.DataFrame, dict]:
        snap = get_snapshot(symbol, self.cfg, now=now, http=self.http, **fetch_kw)
        clean, rep = clean_candles(snap.candles, self.cfg, symbol, n_malformed=snap.n_malformed, now=now,
                                   schema_issues=snap.schema_issues, single_session=True)
        # individually-validated genuine candles are kept even when the batch is
        # graded BAD (e.g. too many gaps); the grade still blocks forecasting.
        self.store.save(symbol, clean, source=LIVE)
        meta = {"report": rep, "previous_close": snap.previous_close, "fetched_at": snap.fetched_at.isoformat(),
                "from_cache": snap.from_cache, "schema_version": snap.schema_version, "raw_path": snap.raw_path}
        return clean, meta

    def _context(self, symbol: str, now: datetime, as_of: pd.Timestamp | None, live: bool,
                 today_candles: dict[str, pd.DataFrame]) -> tuple[object, dict]:
        """Market context with bounded concurrency, per-request timeouts and a
        hard overall deadline.  One slow or failed constituent cannot stall
        the engine; failures and stale constituents are counted and reported."""
        cfg = self.cfg
        mc = cfg["market_context"]
        source = mc.get("source", PROXY_SOURCE)
        index_sym = cfg.universe.index_symbol if source == INDEX_SOURCE else None
        others = [s for s in cfg.universe.constituents if s != symbol]
        if index_sym:
            others.append(index_sym)
        fetched, failed, stale, bad = [symbol] if live else [], [], [], []
        if live:
            kw = {"connect_timeout": float(mc.get("fetch_connect_timeout_seconds", 3)),
                  "read_timeout": float(mc.get("fetch_read_timeout_seconds", 6)),
                  "retries": int(mc.get("fetch_retries", 0))}

            def job(s):
                try:
                    df, meta = self._live_today(s, now, **kw)
                    return s, df, meta["report"].quality
                except DataSourceError:
                    return s, None, None

            ex = ThreadPoolExecutor(max_workers=max(1, int(mc.get("live_fetch_workers", 6))))
            futs = {ex.submit(job, s): s for s in others}
            done, not_done = wait(futs, timeout=float(mc.get("total_deadline_seconds", 25)))
            ex.shutdown(wait=False, cancel_futures=True)
            for f in done:
                s, df, quality = f.result()
                if df is None:
                    failed.append(s)
                elif quality == "BAD":
                    bad.append(s)
                else:
                    fetched.append(s)
                    today_candles[s] = df
                    if assess_freshness(df, now, cfg).is_stale:
                        stale.append(s)
            failed += [futs[f] for f in not_done]
        lrs = {}
        for s in cfg.universe.constituents:
            hist = load_clean_history(self.store, cfg, s, last_n_sessions=7)
            if s in today_candles and not today_candles[s].empty:
                hist = pd.concat([hist[hist.index.normalize() < today_candles[s].index[0].normalize()], today_candles[s]])
            if as_of is not None:
                hist = hist[hist.index <= as_of]
            if not hist.empty:
                lrs[s] = intraday_log_returns(to_minute_grid(hist, cfg))
        index_lr = None
        if index_sym:
            ih = load_clean_history(self.store, cfg, index_sym, last_n_sessions=7)
            if index_sym in today_candles:
                ih = pd.concat([ih[ih.index.normalize() < today_candles[index_sym].index[0].normalize()],
                                today_candles[index_sym]])
            if as_of is not None:
                ih = ih[ih.index <= as_of]
            index_lr = intraday_log_returns(to_minute_grid(ih, cfg)) if not ih.empty else None
        ctx = build_market_context(lrs, cfg, index_lr=index_lr)
        n_univ = len(cfg.universe.constituents)
        info = {
            "source": source,
            "description": PROXY_DESCRIPTION if source == PROXY_SOURCE else f"index series {index_sym}",
            "universe_size": n_univ,
        }
        if live:
            info.update(stocks_fetched=len([s for s in fetched if s != index_sym]),
                        stocks_failed=len([s for s in failed if s != index_sym]),
                        stocks_bad_data=len(bad), stocks_stale=len(stale))
        return ctx, info

    # ------------------------------------------------------------------ main
    def predict(
        self,
        symbol: str,
        *,
        now: datetime | None = None,
        as_of: str | datetime | None = None,
        allow_closed: bool = False,
        use_context: bool | None = None,
        remaining_session: bool = False,
    ) -> Forecast:
        cfg = self.cfg
        symbol = cfg.universe.normalize(symbol)
        now = pd.Timestamp(now or datetime.now(cfg.tz)).tz_convert(cfg.tz).to_pydatetime()
        piped = self.pipeline()
        pipe, meta = piped if piped else (None, {})
        fc = Forecast(symbol=symbol, forecast_status=C.BAD_DATA, model_version=cfg.model_version,
                      coverage_levels={k: float(v) for k, v in cfg["intervals"].items()},
                      forecast_mode=REMAINING if remaining_session else STANDARD)
        if meta:
            fc.model_trained_through = meta.get("history", {}).get("last")

        # ---------------- data --------------------------------------------------
        replay = as_of is not None
        as_of_ts = None
        future = None
        today_candles: dict[str, pd.DataFrame] = {}
        prev_close = None
        if replay:
            fc.mode = "REPLAY"
            as_of_ts = pd.Timestamp(as_of)
            as_of_ts = as_of_ts.tz_localize(cfg.tz) if as_of_ts.tzinfo is None else as_of_ts.tz_convert(cfg.tz)
            # candles starting before as_of are complete at as_of; nothing later feeds the features
            full = load_clean_history(self.store, cfg, symbol)
            hist = full[full.index < as_of_ts]
            future = full[full.index >= as_of_ts]
            if hist.empty:
                fc.status_reasons = [f"no stored history for {symbol} before {as_of_ts}"]
                fc.forecast_status = C.WARMING_UP
                return fc
            fc.data_quality = "GOOD"
            fc.freshness = {"mode": "replay", "as_of": as_of_ts.isoformat(), "last_candle": hist.index[-1].isoformat()}
            as_of_cut = as_of_ts - pd.Timedelta(minutes=1)
            if pipe is not None:
                day = as_of_ts.strftime("%Y-%m-%d")
                fc.replay_sample = "IN-SAMPLE" if day in set(pipe.train_sessions) | set(pipe.calibration_sessions) else "OUT-OF-SAMPLE"
        else:
            try:
                today, live_meta = self._live_today(symbol, now)
            except DataSourceError as exc:
                fc.status_reasons = [f"data source error: {exc}"]
                fc.freshness = {"now": pd.Timestamp(now).isoformat()}
                return fc
            rep = live_meta["report"]
            prev_close = live_meta["previous_close"]
            fc.data_quality, fc.data_issues = rep.quality, rep.issues
            fresh = assess_freshness(today, now, cfg)
            fc.freshness = {**fresh.to_dict(), "fetched_at": live_meta["fetched_at"],
                            "from_cache": live_meta["from_cache"], "raw_archive": live_meta["raw_path"]}
            if today.empty:
                fc.status_reasons = ["endpoint returned no valid candles"]
                fc.forecast_status = C.BAD_DATA if fresh.market_open else C.MARKET_CLOSED
                return fc
            today_candles[symbol] = today
            day0 = today.index[0].normalize()
            past = load_clean_history(self.store, cfg, symbol, last_n_sessions=cfg["warmup"]["history_sessions_for_features"] + 1)
            hist = pd.concat([past[past.index.normalize() < day0], today])
            as_of_cut = None
            if rep.quality == "BAD":
                fc.forecast_status = C.BAD_DATA
                fc.status_reasons = ["live candles failed validation: " + "; ".join(rep.issues)]
                return fc
            if not fresh.market_open:
                fc.mode = "ANALYSIS"
                if not allow_closed:
                    self._describe_state(fc, hist, symbol, None, prev_close)
                    fc.forecast_status = C.MARKET_CLOSED
                    fc.status_reasons = ["market is closed: there is no live next-60-minute window "
                                         "(use --analysis to evaluate the last stored state)"]
                    return fc
            elif fresh.is_stale:
                fc.forecast_status = C.BAD_DATA
                fc.status_reasons = [f"stale data: {fresh.reason}"]
                return fc

        fc.history_sessions = int(hist.index.normalize().nunique())

        # ---------------- state -------------------------------------------------
        want_ctx = (pipe.use_context if pipe else cfg["market_context"]["enabled"]) if use_context is None else use_context
        ctx, ctx_info = (None, {"source": "disabled"})
        if want_ctx:
            ctx, ctx_info = self._context(symbol, now, as_of_cut, live=not replay, today_candles=today_candles)
        row, frame = self._describe_state(fc, hist, symbol, ctx, prev_close)
        if row is None:
            fc.forecast_status = C.WARMING_UP
            fc.status_reasons = ["not enough candles to compute the current state"]
            return fc
        frac = row.get("ctx_frac")
        frac = None if frac is None or not np.isfinite(frac) else float(frac)
        ctx_ok = (not want_ctx) or (frac is not None and frac >= float(cfg["market_context"]["min_constituent_fraction"]))
        fc.market_context = {**ctx_info, "available": bool(want_ctx and ctx_ok),
                             "coverage_at_forecast_minute": None if frac is None else round(frac, 3)}
        if want_ctx and ctx_info.get("source") == PROXY_SOURCE and frac is not None:
            fc.market_context["stocks_in_proxy_at_forecast_minute"] = int(round(frac * ctx_info["universe_size"]))
        if want_ctx and not ctx_ok:
            if fc.data_quality == "GOOD":
                fc.data_quality = "DEGRADED"
            fc.data_issues = fc.data_issues + [f"market context insufficient ({ctx_info.get('source')} coverage "
                                               f"{0 if frac is None else frac:.0%})"]

        # ---------------- gates -------------------------------------------------
        def stop(status, reason):
            fc.forecast_status, fc.status_reasons = status, fc.status_reasons + [reason]
            return self._with_actual(fc, future)

        if pipe is None:
            return stop(C.WARMING_UP, "no trained model: collect real history (collect.py / import_history.py), then run train.py")
        prov = meta.get("data_provenance", {"synthetic": True})
        if (prov.get("synthetic") or prov.get("unknown")) and not cfg["provenance"].get("allow_synthetic_model", False):
            return stop(C.WARMING_UP, "the saved model was trained on synthetic or unlabelled data; "
                                      "retrain on real PSYGRID history (train.py)")
        if want_ctx and pipe.use_context and pipe.context_source != ctx_info.get("source"):
            return stop(C.NO_EDGE, f"market-context source mismatch: model trained with {pipe.context_source}, "
                                   f"engine configured for {ctx_info.get('source')}")
        n_hist = len(self.store.sessions(symbol))
        if n_hist < int(cfg["warmup"]["min_symbol_sessions"]):
            return stop(C.WARMING_UP, f"{n_hist} stored sessions for {symbol}; {cfg['warmup']['min_symbol_sessions']} required")
        today_n = int((hist.index.normalize() == hist.index[-1].normalize()).sum())
        if today_n < int(cfg["validation"]["min_candles_for_forecast"]) and not np.isfinite(row.get("log_ref_vol", np.nan)):
            return stop(C.WARMING_UP, f"only {today_n} candles today and no prior-session reference volatility")
        h_eff = int(min(cfg.horizon, cfg.session_minutes - 1 - row["_mos"]))
        fc.horizon_minutes = max(h_eff, 0)
        if h_eff < cfg.horizon and not remaining_session:
            return stop(C.NO_EDGE, f"only {max(h_eff, 0)} minutes remain in the session: a true next-{cfg.horizon}-minute "
                                   "forecast is impossible (use --remaining-session for an explicitly labelled "
                                   "remaining-session forecast)")
        if h_eff < int(cfg["horizon"]["min_effective_minutes"]):
            return stop(C.NO_EDGE, f"only {max(h_eff, 0)} minutes left in the session")
        if remaining_session and h_eff == cfg.horizon:
            fc.forecast_mode = STANDARD  # a full hour is available: this *is* the standard forecast

        ood = pipe.out_of_domain(frame.iloc[[-1]])[0]
        decision = C.decide(
            table=meta.get("confidence", {"levels": fc.coverage_levels, "cells": {}, "regimes": {}, "symbols": {}}),
            regime=row["regime"], tod=TOD_BUCKETS[int(row["tod_bucket"])], symbol=symbol,
            cfg_conf=cfg["confidence"], data_quality=fc.data_quality or "GOOD",
            out_of_domain=ood, context_ok=ctx_ok,
        )
        fc.forecast_status, fc.status_reasons, fc.evidence = decision.status, decision.reasons, decision.evidence
        cs = decision.evidence.get("cell_stats") or {}
        fc.model_evidence = {
            "training_rows": pipe.n_train_rows,
            "training_sessions": len(pipe.train_sessions),
            "calibration_sessions": len(pipe.calibration_sessions),
            "regime_match": decision.evidence.get("cell"),
            "regime_match_oos_forecasts": cs.get("n"),
            "regime_match_oos_sessions": cs.get("sessions"),
            "data_provenance": prov.get("sources"),
            "context_source": pipe.context_source,
        }
        if fc.replay_sample == "IN-SAMPLE":
            fc.status_reasons = fc.status_reasons + [
                f"IN-SAMPLE replay: {as_of_ts:%Y-%m-%d} was used to train/calibrate this model; "
                "this is NOT evidence of live accuracy (see backtest.py)"]
        if decision.status not in C.FORECAST_STATUSES:
            return self._with_actual(fc, future)

        # ---------------- forecast ----------------------------------------------
        last = frame.iloc[[-1]].copy()
        last["symbol"] = symbol
        pred = pipe.predict(last)
        price = float(row["price"])
        scale = float(row["sigma"]) * math.sqrt(h_eff) if cfg["target"].get("scaling", "volatility") != "none" else 1.0
        up, lo = float(pred["probable_up"][0]), float(pred["probable_lo"][0])
        cu, cl = float(pred["central_up"][0]), float(pred["central_lo"][0])
        fc.range = {"low": price * (1 - lo * scale), "high": price * (1 + up * scale)}
        fc.central_range = {"low": price * (1 - cl * scale), "high": price * (1 + cu * scale)}
        fc.upside_pct = (fc.range["high"] / price - 1) * 100
        fc.downside_pct = (fc.range["low"] / price - 1) * 100
        taus = cfg.quantiles
        fc.quantiles = {
            "upper_high": {f"p{int(t * 100)}": price * (1 + pred["q_up"][0, i] * scale) for i, t in enumerate(taus)},
            "lower_low": {f"p{int(t * 100)}": price * (1 - pred["q_lo"][0, i] * scale) for i, t in enumerate(taus)},
            "note": "upper_high.pX: X% chance the next-hour HIGH is at or below this price; "
                    "lower_low.pX: X% chance the next-hour LOW is at or above this price",
        }
        return self._with_actual(fc, future)

    # ------------------------------------------------------------------ helpers
    def _describe_state(self, fc: Forecast, hist, symbol, ctx, prev_close):
        frame = build_symbol_frame(hist, self.cfg, symbol, context=ctx, previous_close=prev_close, with_targets=False)
        if frame.empty:
            return None, frame
        frame = frame[frame["present"]]
        if frame.empty:
            return None, frame
        row = frame.iloc[-1]
        ts = frame.index[-1]
        fc.last_candle = ts.isoformat()
        fc.timestamp = (ts + pd.Timedelta(minutes=1)).isoformat()
        fc.current_price = float(row["price"])
        fc.regime = row["regime"]
        fc.volatility_state = row["volatility_state"]
        fc.structure = row["structure"]
        fc.relative_strength = row["relative_strength"]
        src = ctx.source if ctx is not None else "UNAVAILABLE"
        fc.explanation = describe(row, market_label=src)
        h = int(min(self.cfg.horizon, self.cfg.session_minutes - 1 - row["_mos"]))
        fc.horizon_minutes = max(h, 0)
        fc.window_end = (ts + pd.Timedelta(minutes=1 + max(h, 0))).isoformat()
        return row, frame

    def _with_actual(self, fc: Forecast, future: pd.DataFrame | None) -> Forecast:
        if future is None or future.empty or not fc.horizon_minutes or fc.timestamp is None:
            return fc
        start = pd.Timestamp(fc.timestamp)
        win = future[(future.index >= start) & (future.index < start + pd.Timedelta(minutes=fc.horizon_minutes))]
        if len(win) >= 0.9 * fc.horizon_minutes:
            a = {"high": float(win["high"].max()), "low": float(win["low"].min()), "candles": int(len(win))}
            if fc.range:
                a["inside_range"] = bool(a["high"] <= fc.range["high"] and a["low"] >= fc.range["low"])
                a["inside_central_range"] = bool(a["high"] <= fc.central_range["high"] and a["low"] >= fc.central_range["low"])
            fc.actual = a
        return fc
