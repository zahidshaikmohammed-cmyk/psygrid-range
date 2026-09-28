"""Snapshot today's 1-minute candles for the universe into data/raw/.

The live PSYGRID endpoint only serves the current session, so history is
built by running this every trading day (ideally once shortly after the
15:30 close, and optionally during the day).  On Windows, schedule it with
Task Scheduler, e.g.  python C:\\path\\to\\psygrid-range\\collect.py

    python collect.py                 all universe symbols
    python collect.py CIPLA TCS       selected symbols
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from engine.cli import _utf8_stdout
from engine.config import load_config
from engine.ingestion import DataSourceError, get_snapshot
from engine.storage import HistoryStore
from engine.validation import clean_candles


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="*")
    args = ap.parse_args()
    cfg = load_config()
    store = HistoryStore(cfg)
    symbols = [cfg.universe.normalize(s) for s in args.symbols] or list(cfg.universe.constituents)
    now = datetime.now(cfg.tz)

    def one(sym):
        try:
            snap = get_snapshot(sym, cfg, now=now, use_cache=False)
        except DataSourceError as exc:
            return sym, None, str(exc)
        clean, rep = clean_candles(snap.candles, cfg, sym, n_malformed=snap.n_malformed, now=now)
        store.save(sym, clean)
        return sym, rep, None

    ok = 0
    with ThreadPoolExecutor(max_workers=int(cfg["market_context"]["live_fetch_workers"])) as ex:
        for sym, rep, err in ex.map(one, symbols):
            if err:
                print(f"  {sym:12s} FAILED  {err}")
                continue
            ok += 1
            print(f"  {sym:12s} {rep.quality:8s} {rep.n_valid:4d} candles  {rep.last_candle or '-'}"
                  + (f"  [{'; '.join(rep.issues)}]" if rep.issues else ""))
    print(f"stored {ok}/{len(symbols)} symbols in {store.root}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
