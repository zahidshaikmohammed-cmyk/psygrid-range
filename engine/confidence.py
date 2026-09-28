"""No-forced-prediction system.

A forecast status is decided *only* from empirical, out-of-sample evidence
collected by the walk-forward backtest (never from a made-up number).

Definitions (thresholds in ``config/engine.json`` -> ``confidence``)
---------------------------------------------------------------------
Evidence cell   The walk-forward test records sharing the current
                (regime, time-of-day bucket).  If that cell has too few
                sessions, the regime-only cell is used and the status is
                capped at MEDIUM.

HIGH_CONFIDENCE
    cell sessions >= high_min_cell_sessions, and
    |realised probable-range coverage - nominal| <= high_coverage_tolerance, and
    pinball skill vs. the BEST of the baselines >= high_min_skill, and
    this stock's own out-of-sample coverage is within
    symbol_high_coverage_tolerance over >= symbol_min_sessions sessions, and
    data quality GOOD, state inside the training domain, context available.

    (all HIGH conditions also require the per-session skill t-statistic in the
    cell >= high_min_skill_tstat and the stock's own t >= edge_min_skill_tstat)

MEDIUM_CONFIDENCE
    cell sessions >= min_cell_sessions, coverage within
    medium_coverage_tolerance, skill > min_skill with per-session
    t-statistic >= medium_min_skill_tstat, and nothing disqualifying.

INSUFFICIENT_EDGE
    anything else that is not a data or warm-up problem: an unvalidated or
    miscalibrated state, NO DEMONSTRATED EDGE over the best baseline (in the
    state or for the stock), an out-of-domain feature vector, missing or
    mismatched market context, or fewer than 60 minutes left in the session
    for a standard forecast.

WARMING_UP   no trained model, a model not trained on real (non-synthetic)
             data, or not enough stored history for this stock.
BAD_DATA     the live data failed validation or is stale.
MARKET_CLOSED  the exchange is closed; there is no "next 60 minutes".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .evaluation import skill_stats

HIGH, MEDIUM, NO_EDGE = "HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "INSUFFICIENT_EDGE"
WARMING_UP, BAD_DATA, MARKET_CLOSED = "WARMING_UP", "BAD_DATA", "MARKET_CLOSED"
FORECAST_STATUSES = (HIGH, MEDIUM)


def _cell(g: pd.DataFrame) -> dict:
    sk = skill_stats(g)
    return {
        "n": int(len(g)),
        "sessions": int(g["session"].nunique()),
        "coverage_probable": float(g["covered_probable"].mean()),
        "coverage_central": float(g["covered_central"].mean()),
        "width_probable_pct": float(g["width_probable_pct"].mean()),
        "skill": sk["skill"],
        "skill_tstat": sk["skill_tstat"],
        "best_baseline": sk["best_baseline"],
    }


def build_confidence_table(records: pd.DataFrame, levels: dict[str, float]) -> dict:
    t: dict = {"levels": levels, "cells": {}, "regimes": {}, "symbols": {}, "vol_states": {}}
    if records.empty:
        return t
    t["overall"] = _cell(records)
    for (reg, tod), g in records.groupby(["regime", "tod_bucket"]):
        t["cells"][f"{reg}|{tod}"] = _cell(g)
    for reg, g in records.groupby("regime"):
        t["regimes"][reg] = _cell(g)
    for sym, g in records.groupby("symbol"):
        t["symbols"][sym] = _cell(g)
    for vs, g in records.groupby("volatility_state"):
        t["vol_states"][vs] = _cell(g)
    return t


@dataclass
class Decision:
    status: str
    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


def decide(
    *,
    table: dict,
    regime: str,
    tod: str,
    symbol: str,
    cfg_conf: dict,
    data_quality: str,
    out_of_domain: list[str],
    context_ok: bool,
) -> Decision:
    nominal = float(table["levels"]["probable"])
    reasons: list[str] = []
    cap_medium = False
    cell = table["cells"].get(f"{regime}|{tod}")
    ev = {"cell": f"{regime}|{tod}", "cell_stats": cell}
    if not cell or cell["sessions"] < cfg_conf["min_cell_sessions"]:
        reg_cell = table["regimes"].get(regime)
        if not reg_cell or reg_cell["sessions"] < cfg_conf["min_cell_sessions"]:
            n = 0 if not cell else cell["sessions"]
            return Decision(NO_EDGE, [f"state {regime} @ {tod} has only {n} out-of-sample sessions of evidence"], ev)
        cell = reg_cell
        ev.update(cell=regime, cell_stats=cell)
        cap_medium = True
        reasons.append(f"sparse evidence for {regime} @ {tod}; using regime-wide evidence")

    t_cell = cell.get("skill_tstat", float("nan"))
    t_cell = float("nan") if t_cell is None else t_cell
    dev = abs(cell["coverage_probable"] - nominal)
    if dev > cfg_conf["medium_coverage_tolerance"]:
        return Decision(NO_EDGE, [f"out-of-sample coverage in this state is {cell['coverage_probable']:.0%} vs nominal {nominal:.0%}"], ev)
    if not (cell["skill"] > cfg_conf["min_skill"]):
        return Decision(NO_EDGE, [f"NO DEMONSTRATED EDGE in this state: skill {cell['skill']:+.1%} "
                                  f"vs best baseline ({cell.get('best_baseline')})"], ev)
    need_t = float(cfg_conf.get("medium_min_skill_tstat", 1.0))
    if not (math.isfinite(t_cell) and t_cell >= need_t):
        return Decision(NO_EDGE, [f"NO DEMONSTRATED EDGE in this state: skill {cell['skill']:+.1%} is not "
                                  f"statistically distinguishable from the best baseline (t={t_cell:.1f} < {need_t:g})"], ev)
    if out_of_domain:
        return Decision(NO_EDGE, [f"current state outside the training domain: {', '.join(out_of_domain)}"], ev)
    if not context_ok:
        return Decision(NO_EDGE, ["market context required by the model is unavailable"], ev)

    sym = table["symbols"].get(symbol)
    ev["symbol_stats"] = sym
    if not sym or sym["sessions"] < cfg_conf["symbol_min_sessions"]:
        cap_medium = True
        reasons.append("stock-level coverage not yet verified out-of-sample")
    else:
        if not (sym["skill"] > cfg_conf["min_skill"]):
            return Decision(NO_EDGE, [f"NO DEMONSTRATED EDGE for {symbol}: out-of-sample skill {sym['skill']:+.1%} "
                                      f"vs best baseline ({sym.get('best_baseline')})"], ev)
        sdev = abs(sym["coverage_probable"] - nominal)
        if sdev > cfg_conf["medium_coverage_tolerance"]:
            return Decision(NO_EDGE, [f"{symbol} out-of-sample coverage {sym['coverage_probable']:.0%} vs nominal {nominal:.0%}"], ev)
        t_sym = sym.get("skill_tstat", float("nan"))
        if t_sym is None or not math.isfinite(t_sym) or t_sym < float(cfg_conf.get("edge_min_skill_tstat", 2.0)):
            cap_medium = True
            reasons.append(f"{symbol}'s own out-of-sample edge is not yet statistically demonstrated")
        if sdev > cfg_conf["symbol_high_coverage_tolerance"]:
            cap_medium = True
            reasons.append(f"{symbol} coverage {sym['coverage_probable']:.0%} is outside the high-confidence band")

    if data_quality != "GOOD":
        cap_medium = True
        reasons.append(f"data quality {data_quality}")
    high = (
        not cap_medium
        and cell["sessions"] >= cfg_conf["high_min_cell_sessions"]
        and dev <= cfg_conf["high_coverage_tolerance"]
        and cell["skill"] >= cfg_conf["high_min_skill"]
        and t_cell >= float(cfg_conf.get("high_min_skill_tstat", 2.0))
    )
    if high:
        return Decision(HIGH, reasons or ["state validated out-of-sample"], ev)
    if not reasons:
        if cell["sessions"] < cfg_conf["high_min_cell_sessions"]:
            reasons.append(f"{cell['sessions']} sessions of evidence (< {cfg_conf['high_min_cell_sessions']} for HIGH)")
        if dev > cfg_conf["high_coverage_tolerance"]:
            reasons.append(f"coverage {cell['coverage_probable']:.0%} vs nominal {nominal:.0%}")
        if cell["skill"] < cfg_conf["high_min_skill"]:
            reasons.append(f"skill {cell['skill']:+.1%} below the HIGH threshold")
        if t_cell < float(cfg_conf.get("high_min_skill_tstat", 2.0)):
            reasons.append(f"skill t-statistic {t_cell:.1f} below the HIGH threshold")
    return Decision(MEDIUM, reasons, ev)
