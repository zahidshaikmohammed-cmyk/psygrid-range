"""PHASE 8 - walk-forward validation, purge/embargo, evaluation records."""
import pandas as pd

from engine.evaluation import summarize
from engine.walkforward import make_folds, purge, required_sessions


def test_folds_are_chronological_disjoint_and_embargoed(cfg):
    sessions = [f"2026-01-{d:02d}" for d in range(1, 31)]
    folds = make_folds(sessions, cfg)
    assert folds, "expected at least one fold"
    emb = cfg["walk_forward"]["embargo_sessions"]
    prev_test_end = None
    for f in folds:
        assert max(f.train) < min(f.calibration) < min(f.test)
        assert set(f.train).isdisjoint(f.calibration) and set(f.calibration).isdisjoint(f.test)
        i_tr, i_c = sessions.index(max(f.train)), sessions.index(min(f.calibration))
        i_ce, i_t = sessions.index(max(f.calibration)), sessions.index(min(f.test))
        assert i_c - i_tr - 1 == emb and i_t - i_ce - 1 == emb
        assert len(f.train) >= cfg["walk_forward"]["min_train_sessions"]
        if prev_test_end:
            assert min(f.test) > prev_test_end
        prev_test_end = max(f.test)
    assert folds[-1].test[-1] == sessions[-1]  # most recent data is always tested
    assert make_folds(sessions[: required_sessions(cfg) - 1], cfg) == []


def test_purge_drops_labels_reaching_boundary():
    tz = "Asia/Kolkata"
    df = pd.DataFrame({"target_end": pd.to_datetime(["2026-01-05 15:29", "2026-01-06 09:20"]).tz_localize(tz)})
    assert len(purge(df, "2026-01-06")) == 1


def test_backtest_records_and_metrics(trained_home):
    _, res = trained_home
    r = res.records
    required = {"timestamp", "symbol", "current_price", "predicted_low", "predicted_high", "actual_low", "actual_high",
                "covered_probable", "regime", "tod_bucket", "volatility_state", "pinball_model", "pinball_baseline"}
    assert required.issubset(r.columns)
    assert (r["predicted_low"] < r["current_price"]).mean() > 0.95
    assert (r["predicted_high"] > r["current_price"]).mean() > 0.95
    s = res.summary
    for g in ("by_time_of_day", "by_regime", "by_volatility_state", "by_symbol"):
        assert s[g]
    o = s["overall"]
    # pipeline sanity on synthetic data: calibrated coverage in the right neighbourhood
    assert 0.65 < o["coverage_probable"] < 0.92 and 0.35 < o["coverage_central"] < 0.65
    assert o["width_central_pct"] < o["width_probable_pct"]
    # no test session was ever used for training or calibration of its fold
    assert r.groupby("fold")["session"].nunique().max() <= 3
    s2 = summarize(r, [0.1, 0.25, 0.5, 0.75, 0.9])
    assert s2["overall"]["n"] == len(r)
