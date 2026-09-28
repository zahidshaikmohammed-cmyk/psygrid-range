"""PHASE 8 - Chronological walk-forward validation with purge and embargo.

No random splitting.  Sessions are ordered in time and each fold is::

    [ train (expanding) ] [embargo] [ calibration ] [embargo] [ test ]

* **Purge.**  Any training/calibration row whose label window
  (``target_end``) reaches the start of the next block is dropped.  Labels
  never cross the session close, so this is normally a no-op, but it is
  enforced rather than assumed.
* **Embargo.**  ``embargo_sessions`` whole sessions are discarded between
  blocks to absorb serial dependence in features and volatility.

The next fold advances by ``test_sessions`` so test blocks never overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import EngineConfig
from .evaluation import forecast_records
from .pipeline import fit_pipeline


@dataclass
class Fold:
    index: int
    train: list[str]
    calibration: list[str]
    test: list[str]


def required_sessions(cfg: EngineConfig) -> int:
    w = cfg["walk_forward"]
    return w["min_train_sessions"] + w["calibration_sessions"] + w["test_sessions"] + 2 * w["embargo_sessions"]


def make_folds(sessions: list[str], cfg: EngineConfig) -> list[Fold]:
    w = cfg["walk_forward"]
    tr, ca, te, em = w["min_train_sessions"], w["calibration_sessions"], w["test_sessions"], w["embargo_sessions"]
    sessions = sorted(sessions)
    folds: list[Fold] = []
    start_test = tr + em + ca + em
    # anchor folds to the end so the most recent data is always tested
    n_folds = max(0, (len(sessions) - start_test) // te)
    n_folds = min(n_folds, int(w.get("max_folds", n_folds)))
    first_test = len(sessions) - n_folds * te
    for i in range(n_folds):
        t0 = first_test + i * te
        c0 = t0 - em - ca
        train_end = c0 - em
        folds.append(Fold(i, sessions[:train_end], sessions[c0 : c0 + ca], sessions[t0 : t0 + te]))
    return folds


def purge(frame: pd.DataFrame, boundary_session: str) -> pd.DataFrame:
    """Drop rows whose label window reaches ``boundary_session``."""
    boundary = pd.Timestamp(boundary_session).tz_localize(frame["target_end"].dt.tz)
    return frame[frame["target_end"] < boundary]


def run_walk_forward(data: pd.DataFrame, cfg: EngineConfig, *, use_context: bool, progress=None) -> tuple[pd.DataFrame, list[Fold]]:
    sessions = sorted(data["session"].unique())
    folds = make_folds(sessions, cfg)
    records = []
    for f in folds:
        if progress:
            progress(f"fold {f.index + 1}/{len(folds)}: train {f.train[0]}..{f.train[-1]} "
                     f"({len(f.train)}), calib {f.calibration[0]}..{f.calibration[-1]}, test {f.test[0]}..{f.test[-1]}")
        train = purge(data[data["session"].isin(f.train)], f.calibration[0])
        calib = purge(data[data["session"].isin(f.calibration)], f.test[0])
        test = data[data["session"].isin(f.test) & data["target_valid"].astype(bool)]
        if train.empty or calib.empty or test.empty:
            continue
        pipe = fit_pipeline(train, calib, cfg, use_context=use_context)
        pred = pipe.predict(test)
        records.append(forecast_records(test, pred, cfg.quantiles, fold=f.index))
    if not records:
        return pd.DataFrame(), folds
    return pd.concat(records, ignore_index=True), folds
