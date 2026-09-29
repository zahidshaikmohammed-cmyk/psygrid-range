"""Import historical 1-minute OHLCV from CSV files into data/raw/.

    python import_history.py CIPLA path/to/cipla_1m.csv
    python import_history.py --dir path/to/folder     (files named <SYMBOL>.csv)

CSV columns (case-insensitive): timestamp, open, high, low, close, volume.
Timestamps without a timezone are interpreted as IST.  Every candle passes the
same validation as live data; nothing is filled or synthesised.  Each session
is labelled with its provenance ("import:<file name>") in the store.
Only import genuine market data: synthetic data must never be imported.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from engine.cli import _utf8_stdout
from engine.config import load_config
from engine.ingestion import parse_candles
from engine.storage import HistoryStore
from engine.validation import clean_candles


def import_file(path: Path, symbol: str, cfg, store) -> str:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for alt in ("datetime", "date", "time", "ts"):
        if "timestamp" not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: "timestamp"})
    need = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return f"{symbol}: missing columns {missing}"
    frame, bad, _ = parse_candles(df[need].values.tolist(), cfg.tz,
                                  cfg["data_source"].get("timestamp_tz_abbreviations"))
    clean, rep = clean_candles(frame, cfg, symbol, n_malformed=bad)
    touched = store.save(symbol, clean, source=f"import:{path.name}")
    return (f"{symbol}: {rep.n_valid} candles in {len(touched)} sessions, quality {rep.quality}"
            + (f" [{'; '.join(rep.issues)}]" if rep.issues else ""))


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol", nargs="?")
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--dir")
    args = ap.parse_args()
    cfg = load_config()
    store = HistoryStore(cfg)
    jobs = []
    if args.dir:
        for p in sorted(Path(args.dir).glob("*.csv")):
            jobs.append((p, p.stem.upper()))
    elif args.symbol and args.csv:
        jobs.append((Path(args.csv), args.symbol.upper()))
    else:
        ap.print_help()
        return 2
    for path, sym in jobs:
        try:
            if sym != (cfg.universe.index_symbol or "").upper():
                sym = cfg.universe.normalize(sym)
        except KeyError as exc:
            print(f"skip {path}: {exc.args[0]}")
            continue
        print(import_file(path, sym, cfg, store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
