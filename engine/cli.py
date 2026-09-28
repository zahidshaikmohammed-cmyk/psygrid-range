"""Shared command-line front end.  Every per-stock script is a one-line call
to :func:`run` so no forecasting logic is ever duplicated."""
from __future__ import annotations

import argparse
import sys


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def build_parser(symbol: str | None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="range.py" if symbol is None else None,
        description="PSYGRID 1-hour probabilistic range forecast",
    )
    if symbol is None:
        p.add_argument("symbol", help="NSE symbol, e.g. CIPLA")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    p.add_argument("--save-json", metavar="PATH", help="also write the JSON forecast to PATH")
    p.add_argument("--analysis", action="store_true", help="when the market is closed, evaluate the last stored state")
    p.add_argument("--as-of", metavar="'YYYY-MM-DD HH:MM'", help="replay from local history at this IST time (no network)")
    p.add_argument("--no-context", action="store_true", help="skip fetching the other constituents for market context")
    p.add_argument("--ascii", action="store_true", help="print 'Rs.' instead of the rupee sign")
    p.add_argument("-v", "--verbose", action="store_true", help="show quantiles, data issues and context details")
    return p


def run(symbol: str | None = None, argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser(symbol).parse_args(argv)
    sym = symbol or args.symbol
    from .forecasting import RangeEngine
    from .output import render_json, render_text

    engine = RangeEngine()
    try:
        fc = engine.predict(
            sym,
            as_of=args.as_of,
            allow_closed=args.analysis,
            use_context=False if args.no_context else None,
        )
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    js = render_json(fc)
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as fh:
            fh.write(js)
    if args.json:
        print(js)
    else:
        print(render_text(fc, currency="Rs." if args.ascii else None, verbose=args.verbose))
    return 0


def predict(symbol: str) -> dict:
    """Programmatic helper: forecast as a JSON-compatible dict."""
    from .forecasting import RangeEngine
    from .output import to_json_dict

    return to_json_dict(RangeEngine().predict(symbol))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
