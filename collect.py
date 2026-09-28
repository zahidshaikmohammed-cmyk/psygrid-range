"""Store today's genuine 1-minute candles from the live PSYGRID endpoint.

The live endpoint serves only the current session, so history is built by
running this every trading day - ideally once shortly after the 15:30 close
(and optionally during the day).  Re-running is safe: candles are merged by
timestamp, never duplicated, never interpolated.

    python collect.py --all-nifty50          every symbol in config/nifty50.json
    python collect.py --symbol CIPLA         one symbol (repeatable)
    python collect.py CIPLA TCS              same, positional

Raw responses are archived under data/raw/live/ for auditing.
Windows Task Scheduler: program  <repo>\\.venv\\Scripts\\python.exe
                        arguments  collect.py --all-nifty50
                        start in   <repo>
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from engine.cli import _utf8_stdout
from engine.config import load_config
from engine.ingestion import DataSourceError, get_snapshot
from engine.storage import LIVE, HistoryStore
from engine.validation import classify_data_state, clean_candles


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--symbol", action="append", default=[], help="symbol to collect (repeatable)")
    ap.add_argument("--all-nifty50", action="store_true", help="collect every symbol in the universe file")
    args = ap.parse_args()
    cfg = load_config()
    store = HistoryStore(cfg)
    requested = args.symbols + args.symbol
    if args.all_nifty50 or not requested:
        symbols = list(cfg.universe.constituents)
    else:
        try:
            symbols = [cfg.universe.normalize(s) for s in requested]
        except KeyError as exc:
            print(f"error: {exc.args[0]}", file=sys.stderr)
            return 2
    if cfg.universe.index_symbol and cfg["market_context"].get("source") == "NIFTY50_INDEX":
        symbols.append(cfg.universe.index_symbol)
    now = datetime.now(cfg.tz)
    print(f"collecting {len(symbols)} symbols from {cfg['data_source']['base_url']} at {now:%Y-%m-%d %H:%M:%S} IST")

    def one(sym):
        try:
            snap = get_snapshot(sym, cfg, now=now, use_cache=False)
        except DataSourceError as exc:
            return sym, None, None, f"{exc.__class__.__name__}: {exc}"
        clean, rep = clean_candles(snap.candles, cfg, sym, n_malformed=snap.n_malformed, now=now,
                                   schema_issues=snap.schema_issues, single_session=True)
        store.save(sym, clean, source=LIVE)
        st = classify_data_state(endpoint_state=snap.endpoint_state, has_candles=snap.has_candles, clean=clean,
                                 report=rep if snap.has_candles else None, now=now, cfg=cfg)
        return sym, rep if snap.has_candles else None, st, None

    counts = {"OK": 0, "MARKET_CLOSED": 0, "STALE_DATA": 0, "BAD_DATA": 0, "FAILED": 0}
    with ThreadPoolExecutor(max_workers=int(cfg["market_context"]["live_fetch_workers"])) as ex:
        for sym, rep, st, err in ex.map(one, symbols):
            if err:
                counts["FAILED"] += 1
                print(f"  {sym:12s} FAILED         {err}")
                continue
            counts[st.state] += 1
            if rep is None:
                print(f"  {sym:12s} {st.state:14s} {st.reason}")
                continue
            print(f"  {sym:12s} {st.state:14s} {rep.quality:8s} {rep.n_valid:4d} candles  last {rep.last_candle or '-'}"
                  + (f"  [{'; '.join(rep.issues)}]" if rep.issues else ""))
    print("summary: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    if counts["MARKET_CLOSED"] and not counts["OK"] and not counts["STALE_DATA"]:
        print("the endpoint served no current-session candles (market closed) - nothing new was stored; "
              "run this again after/during the next session")
    sessions = sorted({d for s in cfg.universe.constituents for d in store.sessions(s)})
    print(f"store: {store.root}")
    print(f"history now spans {len(sessions)} sessions"
          + (f" ({sessions[0]} .. {sessions[-1]})" if sessions else "") + "; check with: python train.py --status")
    return 0 if counts["FAILED"] + counts["BAD_DATA"] < len(symbols) else 1


if __name__ == "__main__":
    sys.exit(main())
