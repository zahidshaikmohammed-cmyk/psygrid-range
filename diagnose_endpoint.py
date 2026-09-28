"""Check the real PSYGRID endpoint from THIS machine.

    python diagnose_endpoint.py CIPLA
    python diagnose_endpoint.py CIPLA --json
    python diagnose_endpoint.py CIPLA --save-fixture   (writes tests/fixtures/captured/CIPLA.json)

Base URL: config/engine.json -> data_source.base_url, or the PSYGRID_BASE_URL
environment variable.

Verdicts: OK (valid, current candles), MARKET_CLOSED (healthy response, no
live session), STALE_DATA (valid candles, too old while the market is open),
BAD_DATA (network failure, malformed/unexpected payload, invalid candles).
Exit code 0 for OK and MARKET_CLOSED, 1 otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

from engine.cli import _utf8_stdout
from engine.config import load_config
from engine.diagnostics import render_diagnostic, run_diagnostic


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol")
    ap.add_argument("--json", action="store_true", help="print the diagnostic as JSON")
    ap.add_argument("--save-fixture", action="store_true",
                    help="save the raw response to tests/fixtures/captured/<SYMBOL>.json for the parser tests")
    args = ap.parse_args()
    cfg = load_config()
    try:
        sym = cfg.universe.normalize(args.symbol)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    fixture = Path(__file__).parent / "tests" / "fixtures" / "captured" / f"{sym}.json" if args.save_fixture else None
    d = run_diagnostic(sym, cfg, save_fixture=fixture)
    print(json.dumps(d, indent=2, default=str) if args.json else render_diagnostic(d))
    return 0 if d["verdict"] in ("OK", "MARKET_CLOSED") else 1


if __name__ == "__main__":
    sys.exit(main())
