"""PHASE 7 - conformal calibration."""
import math

import numpy as np

from engine.calibration import ConformalCalibrator, conformal_quantile, interp_quantile

Q = [0.1, 0.25, 0.5, 0.75, 0.9]


def test_conformal_quantile_finite_sample():
    s = np.arange(1, 101, dtype=float)
    assert conformal_quantile(s, 0.8) == 81.0  # ceil(101 * 0.8) = 81st smallest
    assert math.isinf(conformal_quantile(np.arange(3.0), 0.9))
    assert math.isinf(conformal_quantile(np.array([]), 0.5))


def test_interp_quantile():
    Qm = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert interp_quantile(Qm, Q, 0.75)[0] == 4.0
    assert np.isclose(interp_quantile(Qm, Q, 0.8)[0], 4.0 + (0.05 / 0.15))


def _biased(n, rng):
    """Quantile forecasts that are systematically too narrow."""
    y_up = np.abs(rng.standard_normal(n))
    y_lo = np.abs(rng.standard_normal(n)) * 1.3
    true_q = np.quantile(np.abs(rng.standard_normal(200000)), Q)
    Qu = np.tile(true_q * 0.7, (n, 1))
    Ql = np.tile(true_q * 0.7 * 1.3, (n, 1))
    return Qu, Ql, y_up, y_lo


def test_joint_coverage_restored_out_of_sample():
    rng = np.random.default_rng(0)
    cal = _biased(4000, rng)
    tod = rng.integers(0, 5, 4000)
    sym = rng.choice(["A", "B"], 4000)
    ses = rng.integers(0, 20, 4000).astype(str)
    c = ConformalCalibrator(Q, {"probable": 0.8, "central": 0.5}).fit(*cal, tod, sym, ses)
    Qu, Ql, yu, yl = _biased(20000, rng)
    out = c.apply(Qu, Ql, rng.integers(0, 5, 20000), rng.choice(["A", "B"], 20000))
    cov80 = ((yu <= out["probable_up"]) & (yl <= out["probable_lo"])).mean()
    cov50 = ((yu <= out["central_up"]) & (yl <= out["central_lo"])).mean()
    raw80 = ((yu <= Qu[:, 4]) & (yl <= Ql[:, 4])).mean()
    assert raw80 < 0.7  # uncalibrated forecasts under-cover
    assert abs(cov80 - 0.8) < 0.03 and abs(cov50 - 0.5) < 0.03
    # asymmetry preserved: the lower side is wider because its base quantile is
    assert (out["probable_lo"] > out["probable_up"]).all()
    # marginal quantiles recalibrated per side
    assert np.allclose((yu[:, None] <= out["q_up"]).mean(axis=0), Q, atol=0.03)
    # serialisation
    c2 = ConformalCalibrator.from_dict(c.to_dict())
    o2 = c2.apply(Qu[:5], Ql[:5], np.zeros(5), np.array(["A"] * 5))
    o1 = c.apply(Qu[:5], Ql[:5], np.zeros(5), np.array(["A"] * 5))
    assert np.allclose(o1["probable_up"], o2["probable_up"])
    assert c.is_usable()
