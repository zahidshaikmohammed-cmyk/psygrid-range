"""Train the global quantile model from local history.

    python train.py              walk-forward backtest + final calibrated model
    python train.py --status     history, model and evidence status (no training)

Training refuses to produce a model until enough sessions exist for at least
one full out-of-sample walk-forward fold (WARM-UP).  The minimum (32
sessions by default) is enough to BEGIN evaluation - it is not evidence of
accuracy.  Only the out-of-sample walk-forward results decide whether the
model has an edge over the baselines.
"""
import argparse
import sys

from engine.cli import _utf8_stdout
from engine.config import load_config
from engine.evaluation import format_report
from engine.training import accuracy_verdict, history_status, model_status, train


def print_status(cfg) -> None:
    st = history_status(cfg)
    ms = model_status(cfg)
    n, need = len(st["sessions"]), st["required"]
    prov = st["provenance"]["sources"]
    print("TRAINING STATUS")
    print("===============")
    print()
    print(f"{'Sessions available':<26}: {n}")
    print(f"{'Required minimum':<26}: {need}  (enough to BEGIN evaluation, not proof of accuracy)")
    print(f"{'Stocks available':<26}: {st['stocks_available']} / {len(cfg.universe.constituents)}")
    print(f"{'Total candles':<26}: {st['total_candles']:,}")
    print(f"{'Last collected session':<26}: {st['last_collected'] or 'none'}")
    print(f"{'Data sources':<26}: " + (", ".join(f"{k} ({v} symbol-sessions)" for k, v in sorted(prov.items())) or "none"))
    if st["provenance"]["synthetic"]:
        print(f"{'':<26}  WARNING: synthetic data present in data/raw - a model trained on it will refuse to forecast")
    print(f"{'History status':<26}: " + ("READY TO TRAIN" if n >= need else f"WARM-UP ({need - n} more sessions needed)"))
    print()
    print(f"{'Model version':<26}: {cfg.model_version}")
    if not ms["exists"]:
        print(f"{'Model status':<26}: " + ("WARMING_UP (not trained)" if n < need else "NOT TRAINED (run: python train.py)"))
    else:
        o = ms["overall"]
        print(f"{'Model status':<26}: TRAINED" + ("  [SYNTHETIC/UNLABELLED DATA - forecasts refused]" if ms["synthetic"] else ""))
        print(f"{'Last training':<26}: {ms['trained_at']}")
        h = ms["history"]
        print(f"{'Trained on':<26}: {h.get('sessions')} sessions {h.get('first')} .. {h.get('last')}, "
              f"{len(h.get('symbols', []))} stocks")
        print(f"{'Market context':<26}: {ms['context_source'] or 'not used'}")
        print(f"{'Calibration status':<26}: " + ("CALIBRATED on " + str(len(ms["calibration_sessions"])) + " held-out sessions"
                                                if ms["calibration_usable"] else "UNUSABLE (too few calibration rows)"))
        if o.get("n"):
            print(f"{'Validation coverage':<26}: {o['coverage_probable']:.1%} realised vs "
                  f"{cfg['intervals']['probable']:.0%} nominal over {o['sessions']} out-of-sample sessions "
                  "(calibration check, not accuracy)")
            print(f"{'Baseline skill':<26}: {o['skill_vs_baseline']:+.1%} vs best baseline ({o['best_baseline']})")
        we, wo = ms["symbols_with_edge"], ms["symbols_without_edge"]
        print(f"{'Stocks with edge':<26}: {len(we)}" + (f"  {', '.join(sorted(we))}" if we else ""))
        print(f"{'Stocks without edge':<26}: {len(wo)}" + (f"  {', '.join(sorted(wo))}" if wo else ""))
    print(f"{'Real-data accuracy':<26}: {accuracy_verdict(ms)}")
    missing = [s for s, k in st["per_symbol"].items() if not k]
    if missing and len(missing) < len(cfg.universe.constituents):
        print(f"{'No history for':<26}: {', '.join(missing)}")


def main() -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="report history/model status and exit")
    ap.add_argument("--symbols", nargs="*", help="restrict training rows to these symbols (context still uses all)")
    ap.add_argument("--no-context", action="store_true", help="train without NIFTY50_PROXY/sector context features")
    args = ap.parse_args()
    cfg = load_config()
    if args.status:
        print_status(cfg)
        return 0
    res = train(cfg, args.symbols, use_context=False if args.no_context else None)
    print(f"\n{res.status}: {res.message}")
    if res.status == "WARM_UP":
        print("The engine stays in WARM-UP mode; forecasts will report WARMING_UP until training succeeds.")
        return 1
    print()
    print(format_report(res.summary, cfg["intervals"], cfg["confidence"]))
    print()
    print("Real-data accuracy:", accuracy_verdict(model_status(cfg)))
    print("These out-of-sample walk-forward results are the ONLY basis for any accuracy statement.")
    print("Completing training does not by itself mean the model is accurate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
