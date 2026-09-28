"""Training orchestration used by ``train.py`` and ``backtest.py``.

1. Build the pooled dataset from local history (warm-up gate).
2. Walk-forward backtest -> out-of-sample records, summary, confidence table.
3. Fit the production pipeline: train on everything except the most recent
   ``calibration_sessions`` (+ embargo) and calibrate on those.
4. Save model (``models/<version>/``) and calibration + evidence
   (``data/calibration/<version>.json``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .confidence import build_confidence_table
from .config import EngineConfig
from .dataset import build_training_frame
from .evaluation import summarize
from .pipeline import fit_pipeline
from .storage import HistoryStore
from .walkforward import purge, required_sessions, run_walk_forward


@dataclass
class TrainResult:
    status: str  # TRAINED | WARM_UP | FAILED
    message: str
    sessions: int = 0
    required: int = 0
    summary: dict = field(default_factory=dict)
    records: pd.DataFrame | None = None


def model_dir(cfg: EngineConfig):
    return cfg.paths.models / cfg.model_version


def calibration_file(cfg: EngineConfig):
    return cfg.paths.calibration / f"{cfg.model_version}.json"


def history_status(cfg: EngineConfig, store: HistoryStore | None = None) -> dict:
    store = store or HistoryStore(cfg)
    per = {s: len(store.sessions(s)) for s in cfg.universe.constituents}
    all_sessions = sorted({d for s in cfg.universe.constituents for d in store.sessions(s)})
    return {"per_symbol": per, "sessions": all_sessions, "required": required_sessions(cfg)}


def train(
    cfg: EngineConfig,
    symbols: list[str] | None = None,
    *,
    backtest_only: bool = False,
    use_context: bool | None = None,
    progress=print,
) -> TrainResult:
    store = HistoryStore(cfg)
    need = required_sessions(cfg)
    use_context = cfg["market_context"]["enabled"] if use_context is None else use_context
    progress("building dataset from local history ...")
    data, _ = build_training_frame(cfg, symbols, store=store, use_context=use_context, progress=None)
    if data.empty:
        return TrainResult("WARM_UP", "no stored history: run collect.py during market hours or import_history.py", 0, need)
    sessions = sorted(data["session"].unique())
    if len(sessions) < need:
        return TrainResult(
            "WARM_UP",
            f"{len(sessions)} sessions of history stored, {need} required for one walk-forward fold "
            f"(train {cfg['walk_forward']['min_train_sessions']} + calibration + test + embargo)",
            len(sessions), need,
        )
    if use_context and data["ctx_frac"].notna().mean() < 0.5:
        progress("market context unavailable for most rows (too few constituents stored) -> training without it")
        use_context = False

    progress(f"dataset: {len(data):,} rows, {data['symbol'].nunique()} symbols, {len(sessions)} sessions")
    records, folds = run_walk_forward(data, cfg, use_context=use_context, progress=progress)
    levels = {k: float(v) for k, v in cfg["intervals"].items()}
    summary = summarize(records, cfg.quantiles) if not records.empty else {"overall": {"n": 0}}
    conf = build_confidence_table(records, levels)
    out_dir = cfg.paths.processed
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records.empty:
        records.to_csv(out_dir / f"backtest_{cfg.model_version}.csv", index=False)
    if backtest_only:
        return TrainResult("BACKTESTED", f"{len(folds)} folds", len(sessions), need, summary, records)

    w = cfg["walk_forward"]
    cal_s = sessions[-w["calibration_sessions"]:]
    train_s = sessions[: -(w["calibration_sessions"] + w["embargo_sessions"])]
    progress(f"final model: train {train_s[0]}..{train_s[-1]} ({len(train_s)}), calibrate {cal_s[0]}..{cal_s[-1]}")
    train_df = purge(data[data["session"].isin(train_s)], cal_s[0])
    cal_df = data[data["session"].isin(cal_s)]
    pipe = fit_pipeline(train_df, cal_df, cfg, use_context=use_context)
    extra = {
        "model_version": cfg.model_version,
        "trained_at": datetime.now(cfg.tz).isoformat(),
        "history": {"sessions": len(sessions), "first": sessions[0], "last": sessions[-1],
                    "symbols": sorted(data["symbol"].unique())},
        "walk_forward": {"folds": [{"train": [f.train[0], f.train[-1]], "calibration": [f.calibration[0], f.calibration[-1]],
                                    "test": [f.test[0], f.test[-1]]} for f in folds], "config": w},
        "confidence": conf,
        "backtest_summary": summary,
    }
    pipe.save(model_dir(cfg), calibration_file(cfg), extra)
    return TrainResult("TRAINED", f"saved {model_dir(cfg)}", len(sessions), need, summary, records)
