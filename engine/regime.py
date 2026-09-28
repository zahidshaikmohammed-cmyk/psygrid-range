"""Market-state / regime layer.

Regimes are inferred from *several* volatility-normalised features voting
together, never from a single indicator.  All thresholds are in units of the
stock's own volatility (t-statistics), so they transfer across stocks.

Outputs (vectorised over a feature frame)
  regime            primary state (see ``REGIMES``)
  volatility_state  HIGH_VOLATILITY / NORMAL_VOLATILITY / LOW_VOLATILITY / UNKNOWN
  structure         HIGHER_HIGHS_HIGHER_LOWS / LOWER_HIGHS_LOWER_LOWS / MIXED / UNKNOWN
  relative_strength STRONG_OUTPERFORM ... STRONG_UNDERPERFORM / UNAVAILABLE

``regime_code`` and ``vol_state_code`` are fed to the model as categorical
features and the string labels are used to slice calibration/validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = [
    "RANGE", "TREND_UP", "TREND_DOWN", "COMPRESSION", "VOLATILITY_EXPANSION",
    "REVERSAL", "BREAKOUT", "BREAKDOWN",
]
VOL_STATES = ["NORMAL_VOLATILITY", "HIGH_VOLATILITY", "LOW_VOLATILITY", "UNKNOWN"]
STRUCTURES = ["MIXED", "HIGHER_HIGHS_HIGHER_LOWS", "LOWER_HIGHS_LOWER_LOWS", "UNKNOWN"]
RS_LABELS = ["INLINE", "OUTPERFORM", "STRONG_OUTPERFORM", "UNDERPERFORM", "STRONG_UNDERPERFORM", "UNAVAILABLE"]
REGIME_FEATURE_COLUMNS = ["regime_code", "vol_state_code"]

HIGH_VOL_RATIO = 1.5
LOW_VOL_RATIO = 0.7


def _nz(s: pd.Series, default: float = 0.0) -> pd.Series:
    return s.fillna(default)


def classify(f: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with regime labels/codes for every row of ``f``."""
    out = pd.DataFrame(index=f.index)

    # ---- volatility state: current EWMA vol vs. the stock's own recent sessions
    vr = f["vol_ratio_ref"]
    vs = np.select([vr.isna(), vr >= HIGH_VOL_RATIO, vr <= LOW_VOL_RATIO],
                   ["UNKNOWN", "HIGH_VOLATILITY", "LOW_VOLATILITY"], "NORMAL_VOLATILITY")
    out["volatility_state"] = vs

    # ---- structure
    ss = f["struct_score"]
    out["structure"] = np.select([ss.isna(), ss >= 2, ss <= -2],
                                 ["UNKNOWN", "HIGHER_HIGHS_HIGHER_LOWS", "LOWER_HIGHS_LOWER_LOWS"], "MIXED")

    # ---- trend votes (each in {-1, 0, +1})
    votes = pd.concat([
        np.sign(_nz(f["ema_spread_9_20"])) * (_nz(f["ema_spread_9_20"]).abs() > 0.5),
        np.sign(_nz(f["ema_spread_20_50"])) * (_nz(f["ema_spread_20_50"]).abs() > 0.5),
        np.sign(_nz(f["ema20_slope"])) * (_nz(f["ema20_slope"]).abs() > 0.5),
        np.sign(_nz(f["d_vwap"])) * (_nz(f["d_vwap"]).abs() > 1.0),
        np.sign(_nz(f["struct_score"])) * (_nz(f["struct_score"]).abs() >= 2),
        np.sign(_nz(f["persistence_30"])) * (_nz(f["persistence_30"]).abs() >= 0.3),
    ], axis=1).sum(axis=1)
    t30 = _nz(f["z_ret_30"])
    trend_up = (votes >= 4) & (t30 > 1.0)
    trend_dn = (votes <= -4) & (t30 < -1.0)

    t5 = _nz(f["z_ret_5"])
    rvol = _nz(f["rvol_15"])
    breakout = (_nz(f["breakout_60"], -1) > 0) & (t5 > 1.0) & (rvol > np.log(1.2))
    breakdown = (_nz(f["breakdown_60"], 1) < 0) & (t5 < -1.0) & (rvol > np.log(1.2))
    reversal = (
        ((_nz(f["z_ret_60"]) > 1.5) & (t5 < -1.5))
        | ((_nz(f["z_ret_60"]) < -1.5) & (t5 > 1.5))
        | ((_nz(f["reversal_signal_5"]).abs() >= 2) & (_nz(f["z_ret_60"]).abs() > 1.0))
    )
    expansion = (_nz(f["vol_acc"], 1) > 1.6) | (_nz(f["compress_5_60"], 1) > 1.8)
    compression = (
        (_nz(f["compress_5_60"], 1) < 0.55) & (_nz(f["vol_acc"], 1) < 0.75) & (out["volatility_state"] != "HIGH_VOLATILITY")
    )
    out["regime"] = np.select(
        [breakout, breakdown, reversal, expansion, trend_up, trend_dn, compression],
        ["BREAKOUT", "BREAKDOWN", "REVERSAL", "VOLATILITY_EXPANSION", "TREND_UP", "TREND_DOWN", "COMPRESSION"],
        "RANGE",
    )
    out["trend_votes"] = votes

    # ---- relative strength versus the market proxy
    rs = f["rs_30"]
    out["relative_strength"] = np.select(
        [rs.isna(), rs >= 2.0, rs >= 0.75, rs <= -2.0, rs <= -0.75],
        ["UNAVAILABLE", "STRONG_OUTPERFORM", "OUTPERFORM", "STRONG_UNDERPERFORM", "UNDERPERFORM"],
        "INLINE",
    )
    out["regime_code"] = pd.Categorical(out["regime"], categories=REGIMES).codes.astype(int)
    out["vol_state_code"] = pd.Categorical(out["volatility_state"], categories=VOL_STATES).codes.astype(int)
    return out
