"""PHASES 5-6 - Baseline and quantile models behind one pluggable interface.

Every model maps features -> a *set of quantiles* of one excursion target
(in volatility units).  Upper and lower excursions are modelled by separate
models (:class:`TwoSidedModel`), so the forecast range is asymmetric whenever
the conditional distributions are.

Implementations
  empirical_tod      Baseline: empirical quantiles per time-of-day bucket.
                     "What does an hour usually look like at this time?"
  lightgbm_quantile  One LightGBM booster per quantile (pinball objective).

Add a model by subclassing :class:`QuantileModel` and registering it in
``MODEL_REGISTRY``; training, calibration and evaluation are model-agnostic.
Quantile crossing is removed by monotone rearrangement (sorting), which can
only reduce pinball loss (Chernozhukov et al., 2010).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES


def rearrange(q: np.ndarray) -> np.ndarray:
    return np.sort(q, axis=1)


def pinball_loss(y: np.ndarray, q: np.ndarray, taus: list[float]) -> np.ndarray:
    """Mean pinball loss per quantile column."""
    y = np.asarray(y, float)[:, None]
    t = np.asarray(taus)[None, :]
    d = y - q
    return np.nanmean(np.maximum(t * d, (t - 1) * d), axis=0)


class QuantileModel(ABC):
    name = "abstract"

    def __init__(self, quantiles: list[float], **params):
        self.quantiles = [float(q) for q in quantiles]
        self.params = params
        self.features: list[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, *, X_val=None, y_val=None) -> "QuantileModel": ...

    @abstractmethod
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return rearrange(self.predict_raw(X))

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "QuantileModel": ...


class EmpiricalQuantileBaseline(QuantileModel):
    """Unconditional-by-state baseline: quantiles of the (vol-normalised)
    target per time-of-day bucket, pooled across stocks."""

    name = "empirical_tod"

    def __init__(self, quantiles, key: str = "tod_bucket", **params):
        super().__init__(quantiles, **params)
        self.key = key
        self.table: dict[str, list[float]] = {}
        self.overall: list[float] = []

    def fit(self, X, y, *, X_val=None, y_val=None):
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        self.overall = np.quantile(y[ok], self.quantiles).tolist()
        keys = X[self.key].to_numpy()
        for k in np.unique(keys[ok]):
            m = ok & (keys == k)
            if m.sum() >= 50:
                self.table[str(int(k))] = np.quantile(y[m], self.quantiles).tolist()
        self.features = [self.key]
        return self

    def predict_raw(self, X):
        keys = X[self.key].to_numpy()
        out = np.tile(np.asarray(self.overall), (len(X), 1))
        for k, qs in self.table.items():
            out[keys == int(k)] = qs
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "baseline.json").write_text(json.dumps(
            {"name": self.name, "quantiles": self.quantiles, "key": self.key, "table": self.table, "overall": self.overall}
        ))

    @classmethod
    def load(cls, directory: Path) -> "EmpiricalQuantileBaseline":
        d = json.loads((directory / "baseline.json").read_text())
        m = cls(d["quantiles"], key=d["key"])
        m.table, m.overall, m.features = d["table"], d["overall"], [d["key"]]
        return m


class LightGBMQuantileModel(QuantileModel):
    name = "lightgbm_quantile"

    def __init__(self, quantiles, **params):
        super().__init__(quantiles, **params)
        self.boosters = []
        self.best_iterations: list[int] = []

    def _lgb_params(self, alpha: float) -> dict:
        p = dict(self.params)
        p.pop("n_estimators", None)
        p.pop("early_stopping_rounds", None)
        p.update(objective="quantile", alpha=alpha, verbosity=-1, seed=int(p.get("seed", 17)), num_threads=int(p.get("num_threads", 0)))
        return p

    def fit(self, X, y, *, X_val=None, y_val=None):
        import lightgbm as lgb

        self.features = list(X.columns)
        cats = [c for c in self.features if c in CATEGORICAL_FEATURES]
        y = np.asarray(y, float)
        n_rounds = int(self.params.get("n_estimators", 500))
        es = int(self.params.get("early_stopping_rounds", 50))
        self.boosters, self.best_iterations = [], []
        for tau in self.quantiles:
            dtrain = lgb.Dataset(X, y, categorical_feature=cats, free_raw_data=False)
            valid_sets, callbacks = [], []
            if X_val is not None and len(X_val) > 100:
                dval = lgb.Dataset(X_val, np.asarray(y_val, float), reference=dtrain, categorical_feature=cats)
                valid_sets = [dval]
                callbacks = [lgb.early_stopping(es, verbose=False)]
            b = lgb.train(self._lgb_params(tau), dtrain, num_boost_round=n_rounds, valid_sets=valid_sets, callbacks=callbacks)
            self.boosters.append(b)
            self.best_iterations.append(int(b.best_iteration or n_rounds))
        return self

    def predict_raw(self, X):
        X = X[self.features]
        return np.column_stack([b.predict(X, num_iteration=it) for b, it in zip(self.boosters, self.best_iterations)])

    def feature_importance(self) -> pd.Series:
        imp = sum(b.feature_importance("gain") for b in self.boosters)
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(self.boosters):
            b.save_model(str(directory / f"q{i}.txt"), num_iteration=self.best_iterations[i])
        (directory / "model.json").write_text(json.dumps(
            {"name": self.name, "quantiles": self.quantiles, "params": self.params, "features": self.features}
        ))

    @classmethod
    def load(cls, directory: Path) -> "LightGBMQuantileModel":
        import lightgbm as lgb

        d = json.loads((directory / "model.json").read_text())
        m = cls(d["quantiles"], **d["params"])
        m.features = d["features"]
        m.boosters = [lgb.Booster(model_file=str(directory / f"q{i}.txt")) for i in range(len(m.quantiles))]
        m.best_iterations = [b.current_iteration() for b in m.boosters]
        return m


MODEL_REGISTRY: dict[str, type[QuantileModel]] = {
    EmpiricalQuantileBaseline.name: EmpiricalQuantileBaseline,
    LightGBMQuantileModel.name: LightGBMQuantileModel,
}


def make_model(kind: str, quantiles: list[float], params: dict | None = None) -> QuantileModel:
    if kind not in MODEL_REGISTRY:
        raise KeyError(f"unknown model type {kind!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[kind](quantiles, **(params or {}))


def load_model(directory: Path) -> QuantileModel:
    for fname in ("model.json", "baseline.json"):
        if (directory / fname).exists():
            name = json.loads((directory / fname).read_text())["name"]
            return MODEL_REGISTRY[name].load(directory)
    raise FileNotFoundError(f"no model in {directory}")


class TwoSidedModel:
    """Separate quantile models for upper and lower excursions."""

    def __init__(self, upper: QuantileModel, lower: QuantileModel):
        self.upper, self.lower = upper, lower

    @property
    def quantiles(self) -> list[float]:
        return self.upper.quantiles

    def fit(self, X, y_up, y_lo, *, X_val=None, y_up_val=None, y_lo_val=None):
        self.upper.fit(X, y_up, X_val=X_val, y_val=y_up_val)
        self.lower.fit(X, y_lo, X_val=X_val, y_val=y_lo_val)
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        return self.upper.predict(X), self.lower.predict(X)

    def save(self, directory: Path) -> None:
        self.upper.save(directory / "upper")
        self.lower.save(directory / "lower")

    @classmethod
    def load(cls, directory: Path) -> "TwoSidedModel":
        return cls(load_model(directory / "upper"), load_model(directory / "lower"))
