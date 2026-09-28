"""PHASE 8 (metrics) - Forecast records and out-of-sample evaluation.

Every out-of-sample forecast becomes a record holding the timestamp, symbol,
current price, predicted ranges, actual future high/low and the state labels,
from which we compute:

  coverage (joint, and per side)   interval width (% of price)
  pinball (quantile) loss          skill vs. the empirical baseline
  MAE of the median excursion      quantile calibration curve
  ... overall and by time-of-day, regime, volatility regime and stock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import TOD_BUCKETS


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
    r["pinball_baseline"] = _row_pinball(y_up, y_lo, pred["base_up"], pred["base_lo"], taus)
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


def group_stats(r: pd.DataFrame, taus: list[float]) -> dict:
    if r.empty:
        return {"n": 0}
    pm, pb = r["pinball_model"].mean(), r["pinball_baseline"].mean()
    out = {
        "n": int(len(r)),
        "sessions": int(r["session"].nunique()),
        "coverage_probable": float(r["covered_probable"].mean()),
        "coverage_central": float(r["covered_central"].mean()),
        "coverage_upper_side": float(r["covered_up"].mean()),
        "coverage_lower_side": float(r["covered_lo"].mean()),
        "width_probable_pct": float(r["width_probable_pct"].mean()),
        "width_central_pct": float(r["width_central_pct"].mean()),
        "pinball_model": float(pm),
        "pinball_calibrated": float(r["pinball_calibrated"].mean()),
        "pinball_baseline": float(pb),
        "skill_vs_baseline": float(1 - pm / pb) if pb > 0 else float("nan"),
        "mae_median_excursion_pct": float(r["abs_err_median_pct"].mean()),
        "quantile_calibration": {
            f"q{int(t * 100)}": {
                "upper": float(r[f"below_q{int(t * 100)}_up"].mean()),
                "lower": float(r[f"below_q{int(t * 100)}_lo"].mean()),
            }
            for t in taus
        },
    }
    return out


GROUPINGS = {
    "by_time_of_day": ["tod_bucket"],
    "by_regime": ["regime"],
    "by_volatility_state": ["volatility_state"],
    "by_symbol": ["symbol"],
    "by_regime_and_time": ["regime", "tod_bucket"],
}


def summarize(records: pd.DataFrame, taus: list[float]) -> dict:
    out = {"overall": group_stats(records, taus)}
    for name, keys in GROUPINGS.items():
        rows = []
        for k, g in records.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            rows.append({**dict(zip(keys, k)), **group_stats(g, taus)})
        out[name] = rows
    return out


def format_report(summary: dict, levels: dict[str, float]) -> str:
    o = summary["overall"]
    if o.get("n", 0) == 0:
        return "No out-of-sample records."
    lines = [
        "OUT-OF-SAMPLE WALK-FORWARD RESULTS",
        f"  records {o['n']:,}   sessions {o['sessions']}",
        f"  probable range: nominal {levels['probable']:.0%}  realised {o['coverage_probable']:.1%}  mean width {o['width_probable_pct']:.2f}%",
        f"  central  range: nominal {levels['central']:.0%}  realised {o['coverage_central']:.1%}  mean width {o['width_central_pct']:.2f}%",
        f"  pinball loss  model {o['pinball_model']:.4f}  baseline {o['pinball_baseline']:.4f}  skill {o['skill_vs_baseline']:+.1%}",
        f"  MAE of median excursion {o['mae_median_excursion_pct']:.3f}% of price",
        "  quantile calibration (fraction of outcomes below each predicted quantile):",
    ]
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
    return "\n".join(lines)
