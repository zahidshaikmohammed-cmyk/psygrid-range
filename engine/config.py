"""Central configuration: engine parameters, universe and filesystem layout.

Everything tunable lives in ``config/engine.json`` and the stock universe in
``config/<universe>.json``.  No other module hard-codes symbols or thresholds.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class Paths:
    """Filesystem layout.  ``root`` defaults to the project directory and can be
    redirected with the ``PSYGRID_HOME`` environment variable (used by tests)."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def raw_live(self) -> Path:
        """Archived raw endpoint documents (audit trail), kept apart from the
        per-symbol candle history."""
        return self.raw / "live"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def calibration(self) -> Path:
        return self.data / "calibration"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    @property
    def models(self) -> Path:
        return self.root / "models"


@dataclass(frozen=True)
class Universe:
    name: str
    constituents: tuple[str, ...]
    sectors: dict[str, str]
    index_symbol: str | None = None
    as_of: str | None = None

    def normalize(self, symbol: str) -> str:
        s = symbol.strip().upper()
        if s not in self.sectors:
            raise KeyError(
                f"{symbol!r} is not in the {self.name} universe "
                f"(edit config/{self.name.lower()}.json to add it)"
            )
        return s

    def sector_of(self, symbol: str) -> str:
        return self.sectors.get(symbol.upper(), "UNKNOWN")


@dataclass
class EngineConfig:
    raw: dict[str, Any]
    universe: Universe
    paths: Paths = field(default_factory=lambda: Paths(PROJECT_ROOT))

    # --- convenience accessors -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def model_version(self) -> str:
        return self.raw["model_version"]

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.raw["session"]["timezone"])

    @property
    def session_open(self) -> time:
        return time.fromisoformat(self.raw["session"]["open"])

    @property
    def session_close(self) -> time:
        return time.fromisoformat(self.raw["session"]["close"])

    @property
    def session_minutes(self) -> int:
        o, c = self.session_open, self.session_close
        return (c.hour * 60 + c.minute) - (o.hour * 60 + o.minute)

    @property
    def horizon(self) -> int:
        return int(self.raw["horizon"]["minutes"])

    @property
    def quantiles(self) -> list[float]:
        return [float(q) for q in self.raw["quantiles"]]


def load_universe(filename: str, config_dir: Path = CONFIG_DIR) -> Universe:
    with open(config_dir / filename, encoding="utf-8") as fh:
        u = json.load(fh)
    cons = tuple(c["symbol"].upper() for c in u["constituents"])
    if len(set(cons)) != len(cons):
        raise ValueError(f"duplicate symbols in {filename}")
    sectors = {c["symbol"].upper(): c.get("sector", "UNKNOWN") for c in u["constituents"]}
    return Universe(
        name=u.get("universe", Path(filename).stem.upper()),
        constituents=cons,
        sectors=sectors,
        index_symbol=u.get("index_symbol"),
        as_of=u.get("as_of"),
    )


def load_config(overrides: dict | None = None, root: Path | str | None = None) -> EngineConfig:
    with open(CONFIG_DIR / "engine.json", encoding="utf-8") as fh:
        raw = json.load(fh)
    if overrides:
        raw = _deep_update(raw, overrides)
    env_url = os.environ.get("PSYGRID_BASE_URL")
    if env_url:
        raw["data_source"]["base_url"] = env_url.strip().rstrip("/")
    universe = load_universe(raw["universe_file"])
    root_path = Path(root or os.environ.get("PSYGRID_HOME") or PROJECT_ROOT)
    return EngineConfig(raw=raw, universe=universe, paths=Paths(root_path))
