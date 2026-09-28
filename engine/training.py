"""Training orchestration used by ``train.py`` and ``backtest.py``.

1. Build the pooled dataset from local history (warm-up gate).
2. Walk-forward backtest -> out-of-sample records, summary, confidence table.
3. Fit the production pipeline: train on everything except the most recent
   ``calibration_sessions`` (+ embargo) and calibrate on those.
4. Save model (``models/<version>/``) and calibration + evidence
   (``data/calibration/<version>.json``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .confidence import build_confidence_table
from .config import EngineConfig
from .dataset import build_training_frame
from .evaluation import summarize
from .pipeline import fit_pipeline
from .storage import SYNTHETIC, HistoryStore
from .walkforward import purge, required_sessions, run_walk_forward


@dataclass
class TrainResult:
    status: str  # TRAINED | WARM_UP | FAILED
    message: str
    sessions: int = 0
    required: int = 0
    summary: dict = field(default_factory=dict)
    records: pd.DataFrame | None = None
    provenance: dict = field(default_factory=dict)


def model_dir(cfg: EngineConfig):
    return cfg.paths.models / cfg.model_version


def calibration_file(cfg: EngineConfig):
    return cfg.paths.calibration / f"{cfg.model_version}.json"


def data_provenance(store: HistoryStore, symbols: list[str], sessions: set[str] | None = None) -> dict:
    """Sources behind the given sessions; ``synthetic`` is True if any
    synthetic data (or data of unknown origin) is present."""
    sources: dict[str, int] = {}
    for s in symbols:
        for d, srcs in store.session_sources(s).items():
            if sessions is not None and d not in sessions:
                continue
            for x in srcs:
                key = x.split(":", 1)[0]
                sources[key] = sources.get(key, 0) + 1
    return {"sources": sources, "synthetic": SYNTHETIC in sources, "unknown": "unknown" in sources}


def history_status(cfg: EngineConfig, store: HistoryStore | None = None) -> dict:
    store = store or HistoryStore(cfg)
    per = {s: len(store.sessions(s)) for s in cfg.universe.constituents}
    all_sessions = sorted({d for s in cfg.universe.constituents for d in store.sessions(s)})
    total_candles = 0
    for s in cfg.universe.constituents:
        for d in store.sessions(s):
            with open(store.session_file(s, d), "rb") as fh:
                total_candles += max(sum(1 for _ in fh) - 1, 0)
    prov = data_provenance(store, list(cfg.universe.constituents))
    return {"per_symbol": per, "sessions": all_sessions, "required": required_sessions(cfg),
            "stocks_available": sum(1 for n in per.values() if n), "total_candles": total_candles,
            "last_collected": all_sessions[-1] if all_sessions else None, "provenance": prov}


def model_status(cfg: EngineConfig) -> dict:
    """Read the saved model/calibration metadata (no boosters are loaded)."""
    md, cf = model_dir(cfg), calibration_file(cfg)
    if not (md / "bundle.json").exists() or not cf.exists():
        return {"exists": False}
    meta = json.loads((md / "bundle.json").read_text())
    blob = json.loads(cf.read_text())
    summ = blob.get("backtest_summary", {})
    o = summ.get("overall", {})
    cal = blob.get("calibrator", {})
    return {
        "exists": True,
        "model_version": blob.get("model_version", cfg.model_version),
        "trained_at": blob.get("trained_at"),
        "history": blob.get("history", {}),
        "synthetic": bool(meta.get("data_provenance", {}).get("synthetic", True)),
        "provenance": meta.get("data_provenance", {}),
        "context_source": meta.get("context_source"),
        "calibration_usable": all(isinstance(v, (int, float)) and v != float("inf") for v in cal.get("q_global", {}).values())
        and bool(cal.get("q_global")),
        "calibration_sessions": meta.get("calibration_sessions", []),
        "overall": o,
        "test_sessions": o.get("sessions", 0),
        "symbols_with_edge": summ.get("symbols_with_edge", {}),
        "symbols_without_edge": summ.get("symbols_without_edge", {}),
    }


def accuracy_verdict(ms: dict) -> str:
    """One honest line about what is (not) established."""
    if not ms.get("exists"):
        return "NOT ESTABLISHED (no trained model)"
    if ms.get("synthetic") or ms.get("provenance", {}).get("unknown"):
        return "NOT ESTABLISHED (model trained on synthetic or unlabelled data)"
    o = ms.get("overall", {})
    if not o.get("n"):
        return "NOT ESTABLISHED (no out-of-sample records)"
    from .evaluation import edge_verdict

    v = edge_verdict(o)
    if v.startswith("NO DEMONSTRATED EDGE"):
        return f"{v}; out-of-sample skill {o.get('skill_vs_baseline', float('nan')):+.1%} over {o.get('sessions')} test sessions"
    return f"EVALUATED - {v}; skill {o['skill_vs_baseline']:+.1%}. Evidence, not a guarantee of future accuracy"


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
    summary = (summarize(records, cfg.quantiles, levels["probable"], cfg["confidence"])
               if not records.empty else {"overall": {"n": 0}})
    conf = build_confidence_table(records, levels)
    out_dir = cfg.paths.processed
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records.empty:
        records.to_csv(out_dir / f"backtest_{cfg.model_version}.csv", index=False)
    prov = data_provenance(store, sorted(data["symbol"].unique()), set(sessions))
    if backtest_only:
        return TrainResult("BACKTESTED", f"{len(folds)} folds", len(sessions), need, summary, records, prov)

    w = cfg["walk_forward"]
    cal_s = sessions[-w["calibration_sessions"]:]
    train_s = sessions[: -(w["calibration_sessions"] + w["embargo_sessions"])]
    progress(f"final model: train {train_s[0]}..{train_s[-1]} ({len(train_s)}), calibrate {cal_s[0]}..{cal_s[-1]}")
    train_df = purge(data[data["session"].isin(train_s)], cal_s[0])
    cal_df = data[data["session"].isin(cal_s)]
    pipe = fit_pipeline(train_df, cal_df, cfg, use_context=use_context)
    if prov["synthetic"] or prov["unknown"]:
        progress("WARNING: training data includes synthetic or unlabelled sessions; "
                 "the saved model will refuse production forecasts")
    extra = {
        "bundle": {"data_provenance": prov},
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
    return TrainResult("TRAINED", f"saved {model_dir(cfg)}", len(sessions), need, summary, records, prov)
