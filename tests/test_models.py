"""PHASES 5-6 - baseline and quantile models."""
import numpy as np
import pandas as pd
import pytest

from engine.models import (
    EmpiricalQuantileBaseline,
    LightGBMQuantileModel,
    TwoSidedModel,
    load_model,
    make_model,
    pinball_loss,
)

Q = [0.1, 0.25, 0.5, 0.75, 0.9]


def _data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 2.0, n)
    tod = rng.integers(0, 5, n)
    y = np.abs(rng.standard_normal(n)) * x  # heteroscedastic: spread grows with x
    return pd.DataFrame({"x": x, "noise": rng.standard_normal(n), "tod_bucket": tod}), y


def test_baseline_quantiles_by_bucket():
    X, y = _data()
    y = y + X["tod_bucket"].to_numpy()  # shift per bucket
    m = EmpiricalQuantileBaseline(Q).fit(X, y)
    p = m.predict(X)
    for b in range(5):
        sel = X["tod_bucket"].to_numpy() == b
        assert np.allclose(p[sel][0], np.quantile(y[sel], Q))


def test_lightgbm_learns_conditional_spread_and_does_not_cross(tmp_path):
    X, y = _data()
    m = LightGBMQuantileModel(Q, n_estimators=150, learning_rate=0.05, min_data_in_leaf=50, num_leaves=15).fit(X, y)
    p = m.predict(X)
    assert (np.diff(p, axis=1) >= 0).all()
    lo, hi = X["x"] < 0.8, X["x"] > 1.7
    assert (p[hi, 4] - p[hi, 0]).mean() > 1.5 * (p[lo, 4] - p[lo, 0]).mean()
    # conditional model beats the unconditional baseline out of sample
    Xt, yt = _data(seed=1)
    base = EmpiricalQuantileBaseline(Q).fit(X, y)
    assert pinball_loss(yt, m.predict(Xt), Q).mean() < pinball_loss(yt, base.predict(Xt), Q).mean()
    # empirical coverage of each quantile is roughly right
    cover = (yt[:, None] <= m.predict(Xt)).mean(axis=0)
    assert np.allclose(cover, Q, atol=0.05)
    # persistence round trip
    m.save(tmp_path / "m")
    m2 = load_model(tmp_path / "m")
    assert np.allclose(m2.predict(Xt), m.predict(Xt))
    assert m.feature_importance().index[0] == "x"


def test_two_sided_asymmetry(tmp_path):
    rng = np.random.default_rng(3)
    n = 5000
    X = pd.DataFrame({"rs": rng.uniform(-2, 2, n), "tod_bucket": 0})
    y_up = np.abs(rng.standard_normal(n)) * (1 + 0.4 * X["rs"].clip(lower=0))
    y_lo = np.abs(rng.standard_normal(n)) * (1 + 0.4 * (-X["rs"]).clip(lower=0))
    kw = dict(n_estimators=100, min_data_in_leaf=50)
    m = TwoSidedModel(make_model("lightgbm_quantile", Q, kw), make_model("lightgbm_quantile", Q, kw)).fit(X, y_up, y_lo)
    strong = pd.DataFrame({"rs": [1.9], "tod_bucket": [0]})
    up, lo = m.predict(strong)
    assert up[0, 4] > lo[0, 4] * 1.2  # larger upside tail for a strong stock
    m.save(tmp_path / "two")
    up2, lo2 = TwoSidedModel.load(tmp_path / "two").predict(strong)
    assert np.allclose(up, up2) and np.allclose(lo, lo2)


def test_unknown_model_type():
    with pytest.raises(KeyError):
        make_model("nope", Q)


def test_scaled_baselines():
    from engine.models import BASELINES, make_baseline

    assert set(BASELINES) == {"empirical_tod", "volatility_only", "recent_realized_vol"}
    rng = np.random.default_rng(0)
    n = 4000
    rv = rng.uniform(0.5, 2.0, n)
    X = pd.DataFrame({"rv_60": rv, "tod_bucket": 0})
    y = np.abs(rng.standard_normal(n)) * rv
    b = make_baseline("recent_realized_vol", Q).fit(X, y)
    p = b.predict(pd.DataFrame({"rv_60": [1.0, 2.0, np.nan], "tod_bucket": 0}))
    assert np.allclose(p[1], 2 * p[0]) and np.all(np.isfinite(p[2]))  # scales with recent vol, NaN -> fallback
    v = make_baseline("volatility_only", Q).fit(X, y)
    assert np.allclose(v.predict(X)[0], np.quantile(y, Q))


def test_baseline_roundtrip(tmp_path):
    from engine.models import make_baseline

    X, y = _data()
    X["rv_60"] = 1.0
    b = make_baseline("recent_realized_vol", Q).fit(X, y)
    b.save(tmp_path / "b")
    assert np.allclose(load_model(tmp_path / "b").predict(X), b.predict(X))
