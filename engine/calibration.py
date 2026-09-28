"""PHASE 7 - Conformalised quantile calibration on a held-out period.

Ranges
  A forecast range [low, high] "covers" an hour when the *entire* future path
  stays inside it, i.e. ``future_high <= high`` AND ``future_low >= low``.
  In volatility units that is ``y_up <= U`` and ``y_lo <= L``.

Joint CQR (Romano et al. 2019, adapted to two one-sided bounds)
  Start from the model's quantiles at tau = (1 + level) / 2 on each side and
  compute the conformity score on the calibration set::

      s_i = max(y_up_i - Qup_i(tau), y_lo_i - Qlo_i(tau))

  With q_hat = the ceil((n+1)*level)-th smallest score, the bounds
  ``U = Qup(tau) + q_hat`` and ``L = Qlo(tau) + q_hat`` have (approximately,
  under exchangeability) ``level`` joint coverage.  Asymmetry learned by the
  quantile models is preserved because the same offset is added to both
  sides' *different* base quantiles.

Conditionality
  * q_hat is estimated per time-of-day bucket (Mondrian conformal) with a
    global fallback when a bucket has too few calibration rows.
  * A stock-specific adjustment ``w * (q_hat_stock - q_hat_global)`` is added
    with ``w = n_sessions / (n_sessions + k)`` (shrinkage towards the pool).

Marginal quantiles for display are recalibrated per side with one-sided
conformal offsets so that P(y <= Q_tau + delta_tau) ~= tau.

Minute-level samples within a session are strongly autocorrelated, so the
guarantee is approximate; *actual* coverage is always measured out-of-sample
by the walk-forward backtest and reported per regime / time / stock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MIN_BUCKET_ROWS = 200


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    s = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * level)
    if k > n:
        return math.inf
    return float(s[k - 1])


def interp_quantile(Q: np.ndarray, taus: list[float], tau: float) -> np.ndarray:
    """Column for quantile ``tau`` (linear interpolation across the grid)."""
    t = np.asarray(taus)
    if tau in taus:
        return Q[:, taus.index(tau)]
    j = int(np.clip(np.searchsorted(t, tau), 1, len(t) - 1))
    w = (tau - t[j - 1]) / (t[j] - t[j - 1])
    return Q[:, j - 1] * (1 - w) + Q[:, j] * w


@dataclass
class ConformalCalibrator:
    quantiles: list[float]
    levels: dict[str, float]
    shrinkage_sessions: float = 10.0
    q_global: dict[str, float] = field(default_factory=dict)
    q_tod: dict[str, dict[str, float]] = field(default_factory=dict)
    q_symbol_adj: dict[str, dict[str, float]] = field(default_factory=dict)
    symbol_sessions: dict[str, int] = field(default_factory=dict)
    delta_up: list[float] = field(default_factory=list)
    delta_lo: list[float] = field(default_factory=list)
    n_rows: int = 0
    n_sessions: int = 0

    def base_tau(self, level: float) -> float:
        return (1.0 + level) / 2.0

    def fit(self, Qu, Ql, y_up, y_lo, tod, symbols, sessions) -> "ConformalCalibrator":
        ok = np.isfinite(y_up) & np.isfinite(y_lo)
        Qu, Ql, y_up, y_lo = Qu[ok], Ql[ok], np.asarray(y_up)[ok], np.asarray(y_lo)[ok]
        tod, symbols, sessions = np.asarray(tod)[ok], np.asarray(symbols)[ok], np.asarray(sessions)[ok]
        self.n_rows = int(ok.sum())
        self.n_sessions = int(len(np.unique(sessions)))
        for name, level in self.levels.items():
            tau = self.base_tau(level)
            s = np.maximum(y_up - interp_quantile(Qu, self.quantiles, tau), y_lo - interp_quantile(Ql, self.quantiles, tau))
            g = conformal_quantile(s, level)
            self.q_global[name] = g
            self.q_tod[name] = {}
            for b in np.unique(tod):
                m = tod == b
                if m.sum() >= MIN_BUCKET_ROWS:
                    self.q_tod[name][str(int(b))] = conformal_quantile(s[m], level)
            self.q_symbol_adj[name] = {}
            for sym in np.unique(symbols):
                m = symbols == sym
                n_s = len(np.unique(sessions[m]))
                self.symbol_sessions[str(sym)] = n_s
                if m.sum() >= 50:
                    w = n_s / (n_s + self.shrinkage_sessions)
                    adj = w * (conformal_quantile(s[m], level) - g)
                    if math.isfinite(adj):
                        self.q_symbol_adj[name][str(sym)] = adj
        self.delta_up = [float(np.quantile(y_up - Qu[:, i], t)) for i, t in enumerate(self.quantiles)]
        self.delta_lo = [float(np.quantile(y_lo - Ql[:, i], t)) for i, t in enumerate(self.quantiles)]
        return self

    def offset(self, name: str, tod: np.ndarray, symbols: np.ndarray) -> np.ndarray:
        base = np.full(len(tod), self.q_global[name], dtype=float)
        for b, q in self.q_tod.get(name, {}).items():
            base[np.asarray(tod) == int(b)] = q
        adj = np.zeros(len(tod))
        for sym, a in self.q_symbol_adj.get(name, {}).items():
            adj[np.asarray(symbols) == sym] = a
        return base + adj

    def apply(self, Qu, Ql, tod, symbols) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, level in self.levels.items():
            tau = self.base_tau(level)
            off = self.offset(name, tod, symbols)
            out[f"{name}_up"] = interp_quantile(Qu, self.quantiles, tau) + off
            out[f"{name}_lo"] = interp_quantile(Ql, self.quantiles, tau) + off
        out["q_up"] = np.sort(Qu + np.asarray(self.delta_up)[None, :], axis=1)
        out["q_lo"] = np.sort(Ql + np.asarray(self.delta_lo)[None, :], axis=1)
        return out

    def is_usable(self) -> bool:
        return all(math.isfinite(v) for v in self.q_global.values())

    def to_dict(self) -> dict:
        return {
            "method": "joint_conformalized_quantile_regression",
            "quantiles": self.quantiles, "levels": self.levels,
            "shrinkage_sessions": self.shrinkage_sessions,
            "q_global": self.q_global, "q_tod": self.q_tod, "q_symbol_adj": self.q_symbol_adj,
            "symbol_sessions": self.symbol_sessions,
            "delta_up": self.delta_up, "delta_lo": self.delta_lo,
            "n_rows": self.n_rows, "n_sessions": self.n_sessions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalCalibrator":
        c = cls(quantiles=d["quantiles"], levels=d["levels"], shrinkage_sessions=d["shrinkage_sessions"])
        c.q_global, c.q_tod, c.q_symbol_adj = d["q_global"], d["q_tod"], d["q_symbol_adj"]
        c.symbol_sessions, c.delta_up, c.delta_lo = d["symbol_sessions"], d["delta_up"], d["delta_lo"]
        c.n_rows, c.n_sessions = d["n_rows"], d["n_sessions"]
        return c
