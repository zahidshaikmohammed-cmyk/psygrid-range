"""PHASE 2 - Local historical storage of genuine 1-minute candles.

Layout::

    data/raw/<SYMBOL>/<YYYY-MM-DD>.csv      one file per session, IST timestamps

Every live fetch is merged into the store, so running the engine (or
``collect.py``) during market hours accumulates history.  Bulk history from
other sources can be imported with ``import_history.py``.  Merging never
creates candles: it is a union keyed on timestamp, where a newer record for
the same minute replaces the older one.

Provenance
    ``data/raw/<SYMBOL>/_provenance.json`` records, per session, every source
    that wrote candles into it (``psygrid_live``, ``import:<file>``,
    ``synthetic``...).  ``save`` requires an explicit ``source`` so synthetic
    or test data can never enter the store unlabelled, and training refuses
    to call a model "real" if any synthetic session was used.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from .config import EngineConfig

COLS = ["open", "high", "low", "close", "volume"]
SYNTHETIC = "synthetic"
LIVE = "psygrid_live"


class HistoryStore:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.root = cfg.paths.raw

    # ------------------------------------------------------------------ paths
    def symbol_dir(self, symbol: str) -> Path:
        return self.root / quote(symbol.upper(), safe="")

    def session_file(self, symbol: str, day) -> Path:
        return self.symbol_dir(symbol) / f"{pd.Timestamp(day).date().isoformat()}.csv"

    # ------------------------------------------------------------------ read
    def sessions(self, symbol: str) -> list[str]:
        d = self.symbol_dir(symbol)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.csv"))

    def symbols(self) -> list[str]:
        if not self.root.exists():
            return []
        from urllib.parse import unquote

        return sorted(unquote(p.name) for p in self.root.iterdir()
                      if p.is_dir() and p.name != "live" and any(p.glob("*.csv")))

    def _read_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        idx = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(self.cfg.tz)
        out = df[COLS].astype(float)
        out.index = pd.DatetimeIndex(idx, name="timestamp")
        return out

    def load(
        self,
        symbol: str,
        *,
        start: str | None = None,
        end: str | None = None,
        last_n_sessions: int | None = None,
        exclude: str | None = None,
    ) -> pd.DataFrame:
        days = self.sessions(symbol)
        if exclude:
            days = [d for d in days if d != exclude]
        if start:
            days = [d for d in days if d >= start]
        if end:
            days = [d for d in days if d <= end]
        if last_n_sessions is not None:
            days = days[-last_n_sessions:] if last_n_sessions > 0 else []
        frames = [self._read_file(self.session_file(symbol, d)) for d in days]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz=self.cfg.tz, name="timestamp"), dtype=float)
        out = pd.concat(frames).sort_index()
        return out[~out.index.duplicated(keep="last")]

    # ------------------------------------------------------------------ write
    def provenance(self, symbol: str) -> dict[str, list[str]]:
        p = self.symbol_dir(symbol) / "_provenance.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}

    def session_sources(self, symbol: str) -> dict[str, list[str]]:
        """Sources per stored session ("unknown" when no record exists)."""
        prov = self.provenance(symbol)
        return {d: prov.get(d, ["unknown"]) for d in self.sessions(symbol)}

    def _record_provenance(self, symbol: str, days: list[str], source: str) -> None:
        prov = self.provenance(symbol)
        for d in days:
            srcs = set(prov.get(d, []))
            srcs.add(source)
            prov[d] = sorted(srcs)
        p = self.symbol_dir(symbol) / "_provenance.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(prov, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(p)

    def save(self, symbol: str, candles: pd.DataFrame, *, source: str) -> list[str]:
        """Merge validated candles (IST-indexed) into the store; returns the
        sessions touched.  ``source`` is mandatory (see Provenance)."""
        if not source:
            raise ValueError("HistoryStore.save requires an explicit data source label")
        if candles.empty:
            return []
        touched = []
        for day, part in candles.groupby(candles.index.normalize()):
            path = self.session_file(symbol, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged = part[COLS]
            if path.exists():
                old = self._read_file(path)
                merged = pd.concat([old, part[COLS]])
                merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()
            tmp = path.with_suffix(".tmp")
            out = merged.copy()
            out.index = out.index.map(lambda t: t.isoformat())
            out.index.name = "timestamp"
            out.to_csv(tmp)
            tmp.replace(path)
            touched.append(pd.Timestamp(day).date().isoformat())
        self._record_provenance(symbol, touched, source)
        return touched

    def summary(self) -> pd.DataFrame:
        rows = []
        for s in self.symbols():
            days = self.sessions(s)
            rows.append({"symbol": s, "sessions": len(days), "first": days[0], "last": days[-1],
                         "sources": sorted({x for v in self.session_sources(s).values() for x in v})})
        return pd.DataFrame(rows)
