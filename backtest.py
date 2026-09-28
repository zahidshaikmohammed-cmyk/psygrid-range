"""Chronological walk-forward backtest (no model is saved).

    python backtest.py
    python backtest.py --json report.json

Every out-of-sample forecast record (timestamp, symbol, current price,
predicted low/high, actual future low/high, ...) is written to
data/processed/backtest_<model_version>.csv.
"""
import argparse
import json
import sys

from engine.cli import _utf8_stdout
from engine.config import load_config
from engine.evaluation import format_report
from engine.training import train


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--json", metavar="PATH", help="write the full summary (all groupings) as JSON")
    args = ap.parse_args()
    cfg = load_config()
    res = train(cfg, args.symbols, backtest_only=True, use_context=False if args.no_context else None)
    if res.status == "WARM_UP":
        print(f"WARM-UP: {res.message}")
        return 1
    print(format_report(res.summary, cfg["intervals"]))
    print(f"\nrecords: {cfg.paths.processed / ('backtest_' + cfg.model_version + '.csv')}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res.summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
