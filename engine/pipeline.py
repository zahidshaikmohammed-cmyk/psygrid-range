"""Train -> calibrate -> predict pipeline shared by the walk-forward backtest
and by the production model, so what is validated is exactly what runs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import ConformalCalibrator
from .config import EngineConfig
from .features import CONTEXT_FEATURES
from .models import TwoSidedModel, make_baseline, make_model

DOMAIN_FEATURES = [
    "log_sigma", "vol_ratio_ref", "rv_15", "z_ret_15", "z_ret_60", "gap",
    "vol_z", "rvol_15", "range_5", "m_vol_ratio", "m_z_ret_30",
]
FEATURE_EXCLUDE = {
    "symbol", "session", "present", "_mos", "price", "volatility_state", "structure", "regime",
    "relative_strength", "trend_votes", "h_eff", "window_coverage", "future_high", "future_low",
    "upper_excursion", "lower_excursion", "target_scale", "y_up", "y_lo", "target_end", "target_valid",
    "timestamp", "sigma",
}


def select_features(frame: pd.DataFrame, use_context: bool) -> list[str]:
    cols = [c for c in frame.columns if c not in FEATURE_EXCLUDE]
    if not use_context:
        cols = [c for c in cols if c not in CONTEXT_FEATURES]
    return cols


@dataclass
class FittedPipeline:
    model: TwoSidedModel
    baselines: dict[str, TwoSidedModel]
    calibrator: ConformalCalibrator
    features: list[str]
    domain: dict[str, list[float]]
    use_context: bool
    context_source: str | None = None
    train_sessions: list[str] = field(default_factory=list)
    calibration_sessions: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    n_train_rows: int = 0

    # ------------------------------------------------------------------ predict
    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        X = frame[self.features]
        Qu, Ql = self.model.predict(X)
        cal = self.calibrator.apply(Qu, Ql, frame["tod_bucket"].to_numpy(), frame["symbol"].to_numpy())
        cal.update(raw_up=Qu, raw_lo=Ql)
        for name, b in self.baselines.items():
            Bu, Bl = b.predict(frame)
            cal[f"base_up::{name}"], cal[f"base_lo::{name}"] = Bu, Bl
        return cal

    def out_of_domain(self, frame: pd.DataFrame) -> list[list[str]]:
        res = []
        for _, row in frame.iterrows():
            bad = []
            for k, (lo, hi) in self.domain.items():
                v = row.get(k)
                if v is None or not np.isfinite(v):
                    continue
                if v < lo or v > hi:
                    bad.append(k)
            res.append(bad)
        return res

    # ------------------------------------------------------------------ persistence
    def save(self, model_dir: Path, calibration_file: Path, extra: dict | None = None) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(model_dir / "model")
        for name, b in self.baselines.items():
            b.save(model_dir / "baselines" / name)
        meta = {
            "features": self.features, "domain": self.domain, "use_context": self.use_context,
            "context_source": self.context_source, "baselines": sorted(self.baselines),
            "train_sessions": self.train_sessions, "calibration_sessions": self.calibration_sessions,
            "symbols": self.symbols, "n_train_rows": self.n_train_rows,
            "created_at": datetime.now().astimezone().isoformat(),
            **(extra or {}).get("bundle", {}),
        }
        (model_dir / "bundle.json").write_text(json.dumps(meta, indent=2))
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        blob = {"calibrator": self.calibrator.to_dict(), **{k: v for k, v in (extra or {}).items() if k != "bundle"}}
        calibration_file.write_text(json.dumps(blob, indent=2, default=_json_default))

    @classmethod
    def load(cls, model_dir: Path, calibration_file: Path) -> tuple["FittedPipeline", dict]:
        meta = json.loads((model_dir / "bundle.json").read_text())
        blob = json.loads(calibration_file.read_text())
        pipe = cls(
            model=TwoSidedModel.load(model_dir / "model"),
            baselines={n: TwoSidedModel.load(model_dir / "baselines" / n) for n in meta.get("baselines", [])},
            calibrator=ConformalCalibrator.from_dict(blob["calibrator"]),
            features=meta["features"], domain=meta["domain"], use_context=meta["use_context"],
            context_source=meta.get("context_source"),
            train_sessions=meta["train_sessions"], calibration_sessions=meta["calibration_sessions"],
            symbols=meta["symbols"], n_train_rows=int(meta.get("n_train_rows", 0)),
        )
        return pipe, {**meta, **{k: v for k, v in blob.items() if k != "calibrator"}}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def training_rows(frame: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    """Rows with a valid label, subsampled every ``train_stride_minutes``
    (adjacent minutes carry almost identical information)."""
    stride = max(1, int(cfg["model"].get("train_stride_minutes", 1)))
    m = frame["target_valid"].astype(bool) & np.isfinite(frame["y_up"]) & np.isfinite(frame["y_lo"])
    m &= (frame["_mos"] % stride) == 0
    return frame[m]


def fit_pipeline(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    cfg: EngineConfig,
    *,
    use_context: bool,
) -> FittedPipeline:
    q = cfg.quantiles
    features = select_features(train, use_context)
    tr = training_rows(train, cfg)
    if tr.empty:
        raise ValueError("no labelled training rows")

    # chronological early-stopping split inside the training window
    sessions = sorted(tr["session"].unique())
    n_val = int(round(len(sessions) * float(cfg["model"].get("early_stopping_fraction", 0.15))))
    val_s = set(sessions[-n_val:]) if n_val >= 1 and len(sessions) - n_val >= 3 else set()
    is_val = tr["session"].isin(val_s)
    fit_rows, val_rows = tr[~is_val], tr[is_val]

    mcfg = cfg["model"]
    model = TwoSidedModel(make_model(mcfg["type"], q, mcfg.get("params")), make_model(mcfg["type"], q, mcfg.get("params")))
    model.fit(
        fit_rows[features], fit_rows["y_up"].to_numpy(), fit_rows["y_lo"].to_numpy(),
        X_val=val_rows[features] if len(val_rows) else None,
        y_up_val=val_rows["y_up"].to_numpy() if len(val_rows) else None,
        y_lo_val=val_rows["y_lo"].to_numpy() if len(val_rows) else None,
    )
    baselines = {}
    for bname in cfg["baselines"]:
        b = TwoSidedModel(make_baseline(bname, q), make_baseline(bname, q))
        baselines[bname] = b.fit(tr, tr["y_up"].to_numpy(), tr["y_lo"].to_numpy())

    cal = calib[calib["target_valid"].astype(bool)]
    if cal.empty:
        raise ValueError("no labelled calibration rows")
    Qu, Ql = model.predict(cal[features])
    calibrator = ConformalCalibrator(
        quantiles=q, levels={k: float(v) for k, v in cfg["intervals"].items()},
        shrinkage_sessions=float(cfg["calibration"]["symbol_shrinkage_sessions"]),
    ).fit(Qu, Ql, cal["y_up"].to_numpy(), cal["y_lo"].to_numpy(), cal["tod_bucket"].to_numpy(),
          cal["symbol"].to_numpy(), cal["session"].to_numpy())

    dq = float(cfg["confidence"]["domain_quantile"])
    domain = {}
    for k in DOMAIN_FEATURES:
        if k in tr.columns and k in features:
            v = tr[k].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) > 100:
                domain[k] = [float(np.quantile(v, dq)), float(np.quantile(v, 1 - dq))]
    return FittedPipeline(
        model=model, baselines=baselines, calibrator=calibrator, features=features, domain=domain,
        use_context=use_context, context_source=cfg["market_context"].get("source") if use_context else None,
        train_sessions=sessions, calibration_sessions=sorted(cal["session"].unique()),
        symbols=sorted(tr["symbol"].unique()), n_train_rows=int(len(tr)),
    )
