"""PHASE 8 (metrics) - Forecast records and out-of-sample evaluation.

Every out-of-sample forecast becomes a record holding the timestamp, symbol,
current price, predicted ranges, actual future high/low and the state labels,
from which we compute:

  coverage (joint, and per side)   interval width (% of price)
  pinball (quantile) loss          skill vs. EVERY baseline and vs. the best
  MAE of the median excursion      quantile calibration curve
  ... overall and by time-of-day, regime, volatility regime and stock.

Coverage is a *calibration* property ("did the stated fraction of hours stay
inside the range?").  It is not accuracy and not skill: a range can have
perfect coverage and still be useless if it is no sharper than a naive
baseline.  Skill = 1 - pinball(model) / pinball(best baseline), where the best
baseline is chosen per group by its mean loss (never per row).

Because minutes within a session are strongly dependent, the *session* is the
unit of evidence: ``skill_tstat`` is the t-statistic of the per-session mean
loss improvement over the best baseline.  "Demonstrated edge" requires
positive skill AND ``skill_tstat >= edge_min_skill_tstat`` (default 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import TOD_BUCKETS

BASE_PREFIX = "pinball_base::"


def _row_pinball(y_up, y_lo, Qu, Ql, taus) -> np.ndarray:
    t = np.asarray(taus)[None, :]
    du = y_up[:, None] - Qu
    dl = y_lo[:, None] - Ql
    lu = np.maximum(t * du, (t - 1) * du).mean(axis=1)
    ll = np.maximum(t * dl, (t - 1) * dl).mean(axis=1)
    return 0.5 * (lu + ll)


def forecast_records(frame: pd.DataFrame, pred: dict[str, np.ndarray], taus: list[float], fold: int | None = None) -> pd.DataFrame:
    price = frame["price"].to_numpy(float)
    scale = frame["target_scale"].to_numpy(float)
    y_up = frame["y_up"].to_numpy(float)
    y_lo = frame["y_lo"].to_numpy(float)
    r = pd.DataFrame({
        "timestamp": frame["timestamp"].to_numpy() if "timestamp" in frame else frame.index,
        "symbol": frame["symbol"].to_numpy(),
        "session": frame["session"].to_numpy(),
        "current_price": price,
        "horizon_minutes": frame["h_eff"].to_numpy(),
        "tod_bucket": [TOD_BUCKETS[int(b)] for b in frame["tod_bucket"].to_numpy()],
        "regime": frame["regime"].to_numpy(),
        "volatility_state": frame["volatility_state"].to_numpy(),
        "predicted_low": price * (1 - pred["probable_lo"] * scale),
        "predicted_high": price * (1 + pred["probable_up"] * scale),
        "central_low": price * (1 - pred["central_lo"] * scale),
        "central_high": price * (1 + pred["central_up"] * scale),
        "actual_low": frame["future_low"].to_numpy(float),
        "actual_high": frame["future_high"].to_numpy(float),
    })
    r["covered_probable"] = (y_up <= pred["probable_up"]) & (y_lo <= pred["probable_lo"])
    r["covered_central"] = (y_up <= pred["central_up"]) & (y_lo <= pred["central_lo"])
    r["covered_up"] = y_up <= pred["probable_up"]
    r["covered_lo"] = y_lo <= pred["probable_lo"]
    r["width_probable_pct"] = (r["predicted_high"] - r["predicted_low"]) / price * 100
    r["width_central_pct"] = (r["central_high"] - r["central_low"]) / price * 100
    r["pinball_model"] = _row_pinball(y_up, y_lo, pred["raw_up"], pred["raw_lo"], taus)
    r["pinball_calibrated"] = _row_pinball(y_up, y_lo, pred["q_up"], pred["q_lo"], taus)
    for k in pred:
        if k.startswith("base_up::"):
            name = k.split("::", 1)[1]
            r[BASE_PREFIX + name] = _row_pinball(y_up, y_lo, pred[k], pred[f"base_lo::{name}"], taus)
    i50 = taus.index(0.5) if 0.5 in taus else len(taus) // 2
    r["abs_err_median_pct"] = 50 * (
        np.abs(frame["upper_excursion"].to_numpy() - pred["q_up"][:, i50] * scale)
        + np.abs(frame["lower_excursion"].to_numpy() - pred["q_lo"][:, i50] * scale)
    )
    for i, t in enumerate(taus):
        r[f"below_q{int(t * 100)}_up"] = y_up <= pred["q_up"][:, i]
        r[f"below_q{int(t * 100)}_lo"] = y_lo <= pred["q_lo"][:, i]
    if fold is not None:
        r["fold"] = fold
    return r


def baseline_names(r: pd.DataFrame) -> list[str]:
    return [c[len(BASE_PREFIX):] for c in r.columns if c.startswith(BASE_PREFIX)]


def skill_stats(g: pd.DataFrame) -> dict:
    pm = float(g["pinball_model"].mean())
    bases = {n: float(g[BASE_PREFIX + n].mean()) for n in baseline_names(g)}
    if not bases:
        return {"pinball_model": pm, "baselines": {}, "best_baseline": None, "skill": float("nan"), "skill_vs": {}}
    best = min(bases, key=bases.get)
    tstat = float("nan")
    if "session" in g:
        diff = (g[BASE_PREFIX + best] - g["pinball_model"]).groupby(g["session"].to_numpy()).mean()
        if len(diff) >= 2 and diff.std(ddof=1) > 0:
            tstat = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    return {
        "pinball_model": pm,
        "baselines": bases,
        "best_baseline": best,
        "skill": float(1 - pm / bases[best]) if bases[best] > 0 else float("nan"),
        "skill_tstat": tstat,
        "skill_vs": {n: float(1 - pm / v) if v > 0 else float("nan") for n, v in bases.items()},
    }


def group_stats(r: pd.DataFrame, taus: list[float]) -> dict:
    if r.empty:
        return {"n": 0}
    sk = skill_stats(r)
    return {
        "n": int(len(r)),
        "sessions": int(r["session"].nunique()),
        "coverage_probable": float(r["covered_probable"].mean()),
        "coverage_central": float(r["covered_central"].mean()),
        "coverage_upper_side": float(r["covered_up"].mean()),
        "coverage_lower_side": float(r["covered_lo"].mean()),
        "width_probable_pct": float(r["width_probable_pct"].mean()),
        "width_central_pct": float(r["width_central_pct"].mean()),
        "pinball_model": sk["pinball_model"],
        "pinball_calibrated": float(r["pinball_calibrated"].mean()),
        "pinball_baselines": sk["baselines"],
        "best_baseline": sk["best_baseline"],
        "skill_vs_baseline": sk["skill"],
        "skill_tstat": sk["skill_tstat"],
        "skill_vs_each_baseline": sk["skill_vs"],
        "mae_median_excursion_pct": float(r["abs_err_median_pct"].mean()),
        "quantile_calibration": {
            f"q{int(t * 100)}": {
                "upper": float(r[f"below_q{int(t * 100)}_up"].mean()),
                "lower": float(r[f"below_q{int(t * 100)}_lo"].mean()),
            }
            for t in taus
        },
    }


def has_edge(stats: dict, nominal: float, cfg_conf: dict, min_sessions: int | None = None) -> tuple[bool, str]:
    """Demonstrated edge = enough sessions, positive skill over the best
    baseline, and realised coverage within the medium tolerance."""
    need = cfg_conf.get("edge_min_sessions", cfg_conf["symbol_min_sessions"]) if min_sessions is None else min_sessions
    if not stats or stats.get("n", 0) == 0 or stats.get("sessions", 0) < need:
        return False, f"only {stats.get('sessions', 0) if stats else 0} out-of-sample sessions (< {need})"
    sk = stats.get("skill_vs_baseline", stats.get("skill"))
    if sk is None or not np.isfinite(sk) or sk <= cfg_conf["min_skill"]:
        return False, f"NO DEMONSTRATED EDGE: skill {sk:+.1%} vs best baseline"
    t = stats.get("skill_tstat", float("nan"))
    need_t = float(cfg_conf.get("edge_min_skill_tstat", 2.0))
    if not np.isfinite(t) or t < need_t:
        return False, f"NO DEMONSTRATED EDGE: skill {sk:+.1%} is not statistically distinguishable (t={t:.1f} < {need_t:g})"
    cov = stats["coverage_probable"]
    if abs(cov - nominal) > cfg_conf["medium_coverage_tolerance"]:
        return False, f"miscalibrated: coverage {cov:.1%} vs nominal {nominal:.0%}"
    return True, f"skill {sk:+.1%} (t={t:.1f}), coverage {cov:.1%}"


GROUPINGS = {
    "by_time_of_day": ["tod_bucket"],
    "by_regime": ["regime"],
    "by_volatility_state": ["volatility_state"],
    "by_symbol": ["symbol"],
    "by_regime_and_time": ["regime", "tod_bucket"],
}


def summarize(records: pd.DataFrame, taus: list[float], nominal: float | None = None, cfg_conf: dict | None = None) -> dict:
    out = {"overall": group_stats(records, taus)}
    std = records[records["horizon_minutes"] >= 60] if "horizon_minutes" in records else records
    out["standard_60m"] = group_stats(std, taus)
    for name, keys in GROUPINGS.items():
        rows = []
        for k, g in records.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            rows.append({**dict(zip(keys, k)), **group_stats(g, taus)})
        out[name] = rows
    if nominal is not None and cfg_conf is not None:
        edge, no_edge = {}, {}
        for row in out["by_symbol"]:
            ok, why = has_edge(row, nominal, cfg_conf)
            (edge if ok else no_edge)[row["symbol"]] = why
        out["symbols_with_edge"], out["symbols_without_edge"] = edge, no_edge
    return out


def edge_verdict(stats: dict, cfg_conf: dict | None = None) -> str:
    cc = cfg_conf or {"min_skill": 0.0, "edge_min_skill_tstat": 2.0}
    sk, t = stats.get("skill_vs_baseline"), stats.get("skill_tstat", float("nan"))
    if sk is None or not np.isfinite(sk) or sk <= cc["min_skill"]:
        return "NO DEMONSTRATED EDGE"
    if not np.isfinite(t) or t < float(cc.get("edge_min_skill_tstat", 2.0)):
        return f"NO DEMONSTRATED EDGE (positive but not statistically distinguishable, t={t:.1f})"
    return f"EDGE DEMONSTRATED out-of-sample (t={t:.1f} over {stats.get('sessions')} sessions)"


def format_report(summary: dict, levels: dict[str, float], cfg_conf: dict | None = None) -> str:
    o = summary["overall"]
    if o.get("n", 0) == 0:
        return "No out-of-sample records."
    s60 = summary.get("standard_60m", {})
    lines = [
        "OUT-OF-SAMPLE WALK-FORWARD RESULTS",
        f"  records {o['n']:,}   sessions {o['sessions']}",
        "",
        "  CALIBRATION (coverage = fraction of held-out hours whose full high AND low stayed inside;",
        "               a calibration check, NOT an accuracy or hit-rate claim)",
        f"    probable range: nominal {levels['probable']:.0%}  realised {o['coverage_probable']:.1%}  mean width {o['width_probable_pct']:.2f}%",
        f"    central  range: nominal {levels['central']:.0%}  realised {o['coverage_central']:.1%}  mean width {o['width_central_pct']:.2f}%",
    ]
    if s60.get("n"):
        lines.append(f"    standard 60-minute forecasts only: realised {s60['coverage_probable']:.1%} "
                     f"(n={s60['n']:,}); the rest are remaining-session forecasts after 14:30")
    lines += ["", "  SKILL (pinball loss, lower is better; skill = improvement over the BEST baseline)",
              f"    model                 {o['pinball_model']:.4f}"]
    for n, v in sorted(o["pinball_baselines"].items(), key=lambda kv: kv[1]):
        lines.append(f"    baseline {n:21s}{v:.4f}   model skill vs it {o['skill_vs_each_baseline'][n]:+.1%}")
    lines.append(f"    => skill vs best baseline ({o['best_baseline']}): {o['skill_vs_baseline']:+.1%}   "
                 f"{edge_verdict(o, cfg_conf)}")
    lines += [f"  MAE of median excursion {o['mae_median_excursion_pct']:.3f}% of price",
              "  quantile calibration (fraction of outcomes below each predicted quantile):"]
    for q, v in o["quantile_calibration"].items():
        lines.append(f"    {q}: upper {v['upper']:.3f}  lower {v['lower']:.3f}")
    for name in ("by_time_of_day", "by_regime", "by_volatility_state", "by_symbol"):
        lines.append(f"  {name.replace('_', ' ')}:")
        key = GROUPINGS[name][0]
        for row in sorted(summary[name], key=lambda r: -r["n"]):
            lines.append(
                f"    {str(row[key]):24s} n={row['n']:>7,}  sess={row['sessions']:>3}  "
                f"cov80={row['coverage_probable']:.1%}  cov50={row['coverage_central']:.1%}  "
                f"width={row['width_probable_pct']:.2f}%  skill={row['skill_vs_baseline']:+.1%}"
            )
    if "symbols_with_edge" in summary:
        lines.append(f"  stocks WITH demonstrated edge ({len(summary['symbols_with_edge'])}): "
                     + (", ".join(sorted(summary["symbols_with_edge"])) or "none"))
        lines.append(f"  stocks WITHOUT demonstrated edge ({len(summary['symbols_without_edge'])}):")
        for sym, why in sorted(summary["symbols_without_edge"].items()):
            lines.append(f"    {sym:12s} {why}")
    return "\n".join(lines)
