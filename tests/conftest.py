import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.config import load_config  # noqa: E402
from engine.storage import HistoryStore  # noqa: E402
from synthetic import generate_market  # noqa: E402

TEST_SYMBOLS = ["CIPLA", "TCS", "INFY", "SUNPHARMA", "DRREDDY", "WIPRO", "HDFCBANK", "RELIANCE"]

SMALL = {
    "market_context": {"min_constituent_fraction": 0.1},
    "model": {"train_stride_minutes": 5, "params": {"n_estimators": 60, "min_data_in_leaf": 100, "num_threads": 2}},
    "walk_forward": {"min_train_sessions": 10, "calibration_sessions": 3, "test_sessions": 3, "embargo_sessions": 1},
    "warmup": {"min_symbol_sessions": 5},
    "confidence": {"min_cell_sessions": 2, "high_min_cell_sessions": 5, "symbol_min_sessions": 2},
}


class FakeResponse:
    def __init__(self, payload=None, status=200, text=None):
        self._payload, self.status_code, self._text = payload, status, text

    def json(self):
        if self._text is not None:
            raise ValueError("not json")
        return self._payload


class FakeHTTP:
    """Stands in for requests: maps symbol -> payload / exception."""

    def __init__(self, payloads=None, exc=None, status=200):
        self.payloads, self.exc, self.status, self.calls = payloads or {}, exc, status, []

    def get(self, url, timeout=None, headers=None):
        self.calls.append(url)
        if self.exc:
            raise self.exc
        sym = url.rsplit("/", 1)[-1].removesuffix(".json")
        from urllib.parse import unquote

        sym = unquote(sym)
        if sym not in self.payloads:
            return FakeResponse(status=404)
        return FakeResponse(self.payloads[sym], status=self.status)


@pytest.fixture
def cfg(tmp_path):
    return load_config(SMALL, root=tmp_path)


@pytest.fixture(scope="session")
def market():
    cfg = load_config(SMALL)
    return generate_market(TEST_SYMBOLS, cfg.universe.sectors, n_sessions=24, seed=11)


@pytest.fixture(scope="session")
def trained_home(tmp_path_factory, market):
    """A PSYGRID_HOME with 24 stored sessions and a trained model (all synthetic)."""
    from engine.training import train

    root = tmp_path_factory.mktemp("home")
    cfg = load_config(SMALL, root=root)
    store = HistoryStore(cfg)
    for s, df in market.items():
        store.save(s, df)
    res = train(cfg, progress=lambda *_: None)
    assert res.status == "TRAINED", res.message
    return root, res


def ist(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_localize("Asia/Kolkata")
