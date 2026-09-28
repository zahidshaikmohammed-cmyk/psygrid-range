"""Train the global quantile model from local history.

    python train.py              walk-forward backtest + final calibrated model
    python train.py --status     show how much history is stored (warm-up check)

Training refuses to produce a model until enough sessions exist for at least
one full out-of-sample walk-forward fold (WARM-UP mode).
"""
import argparse
import sys

from engine.cli import _utf8_stdout
from engine.config import load_config
from engine.evaluation import format_report
from engine.training import history_status, train


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="report stored history and exit")
    ap.add_argument("--symbols", nargs="*", help="restrict training rows to these symbols (context still uses all)")
    ap.add_argument("--no-context", action="store_true", help="train without NIFTY/sector context features")
    args = ap.parse_args()
    cfg = load_config()
    st = history_status(cfg)
    if args.status:
        have = {s: n for s, n in st["per_symbol"].items() if n}
        print(f"stored sessions (any symbol): {len(st['sessions'])}   required for training: {st['required']}")
        for s, n in sorted(have.items()):
            print(f"  {s:12s} {n}")
        missing = [s for s, n in st["per_symbol"].items() if not n]
        if missing:
            print(f"  no history: {', '.join(missing)}")
        print("mode:", "READY" if len(st["sessions"]) >= st["required"] else "WARM-UP")
        return 0
    res = train(cfg, args.symbols, use_context=False if args.no_context else None)
    print(f"\n{res.status}: {res.message}")
    if res.status == "WARM_UP":
        print("The engine stays in WARM-UP mode; forecasts will report WARMING_UP until training succeeds.")
        return 1
    print()
    print(format_report(res.summary, cfg["intervals"]))
    print("\nThese are out-of-sample walk-forward results on the stored history. They are the only")
    print("basis for any accuracy statement; re-run after collecting more sessions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
