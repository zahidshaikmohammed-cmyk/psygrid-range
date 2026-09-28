# PSYGRID RANGE ENGINE

A probabilistic engine that estimates, from genuine 1-minute OHLCV, where a
NIFTY-50 stock is likely to trade over the next 60 minutes. It predicts the
distribution of the stock's next-hour high and low. It then turns that into a
calibrated range.

```powershell
python cipla.py
python tcs.py
python range.py RELIANCE
```

It is **not** an ATR band, a fixed-percentage band or a direction call. The
upside and downside are modelled separately and conditioned on the current
market state. When the evidence is weak, the engine says so:

```
NO RELIABLE FORECAST / INSUFFICIENT EDGE
```

> **Current status: WARM-UP.** The live PSYGRID endpoint only serves the
> current session. No accuracy claim exists until history has been collected
> and `train.py` has produced out-of-sample walk-forward results on real data.
> Until then every forecast reports `WARMING_UP`.

---

## 1. Quick start (Windows PowerShell)

```powershell
cd psygrid-range
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt       # includes tzdata (needed for IST on Windows)

python collect.py                     # snapshot today's candles for all 50 stocks
python train.py --status              # how much history is stored? WARM-UP or READY
python train.py                       # walk-forward backtest + calibrated model
python cipla.py                       # live forecast
```

### Building history

The live endpoint only has today's candles, so history has to be accumulated:

| Method | How |
|---|---|
| Daily snapshot | Run `python collect.py` every trading day after 15:30 IST, for example from Windows Task Scheduler. Every `cipla.py` run also stores what it fetched. |
| Import | `python import_history.py --dir D:\exports`. Put one `<SYMBOL>.csv` per stock with columns `timestamp,open,high,low,close,volume`. Naive timestamps are treated as IST. |

Training needs **32 sessions** by default: 20 train + 5 calibration + 5 test +
2 embargo, which is enough for one walk-forward fold. That is a floor, not a
target. HIGH_CONFIDENCE needs 20 out-of-sample sessions of evidence in the
current state, so plan on roughly 60+ sessions before expecting HIGH outputs.

### CLI options (every stock script and `range.py`)

| Flag | Meaning |
|---|---|
| `--json` | Machine-readable output only |
| `--save-json PATH` | Also write the JSON forecast to a file |
| `-v` | Show the quantile table, data issues and market-context details |
| `--analysis` | When the market is closed, evaluate the last stored state (labelled ANALYSIS) |
| `--as-of "2026-09-25 11:30"` | Replay from local history, no network, and show what actually happened. Sessions inside the model's train/calibration window are flagged IN-SAMPLE. |
| `--no-context` | Skip fetching the other 49 stocks for NIFTY/sector context |
| `--ascii` | Print `Rs.` instead of `₹` (for old consoles) |

---

## 2. Output

```
============================================================
PSYGRID 1-HOUR RANGE ENGINE
============================================================

SYMBOL       : CIPLA
PRICE        : ₹1,512.40
TIMESTAMP    : 2026-09-28 11:30 IST
DATA AGE     : last candle closed 0.4 min ago

FORECAST HORIZON
NEXT 60 MINUTES

PROBABLE RANGE  (calibrated for 80% of hours to stay inside)
LOW          : ₹1,503.10
HIGH         : ₹1,524.90

CENTRAL RANGE   (calibrated for 50%)
...
UPSIDE EXTENSION   : +0.83%
DOWNSIDE EXTENSION : -0.61%

MARKET REGIME      : TREND_UP
VOLATILITY STATE   : NORMAL_VOLATILITY
STRUCTURE          : HIGHER_HIGHS_HIGHER_LOWS
RELATIVE STRENGTH  : OUTPERFORM

FORECAST STATUS    : MEDIUM_CONFIDENCE
  - 14 sessions of evidence (< 20 for HIGH)
EVIDENCE           : 14 out-of-sample sessions in this state, 79% realised coverage, skill +6.1% vs baseline
```
*(Illustrative layout only. The numbers are not real results.)*

**What the ranges mean.** A range "covers" an hour when that hour's entire
high and low stay inside it. The *probable range* is calibrated so that, on
held-out data, 80% of hours stayed fully inside. The *central range* is
calibrated to 50%. Whether that held in each market state is measured by the
backtest and drives the status, as described below.

**JSON** (`--json`) contains every field from the specification (`symbol`,
`timestamp`, `current_price`, `horizon_minutes`, `range`, `central_range`,
`upside_pct`, `downside_pct`, `regime`, `volatility_state`,
`forecast_status`, `data_quality`, `model_version`) plus:

* `quantiles`: P10–P90 of the next-hour high and low, as prices.
* `status_reasons` and `evidence`: the out-of-sample statistics behind the status.
* `freshness`, `data_issues`, `market_context`, `structure`, `relative_strength`, `window_end`, `mode`.
* `actual`: replay mode only.

### Forecast statuses

| Status | Meaning (thresholds in `config/engine.json`) |
|---|---|
| `HIGH_CONFIDENCE` | In walk-forward test data for the current *(regime, time-of-day)* cell: at least 20 sessions of evidence, realised coverage within ±5 pp of nominal, and pinball-loss skill of at least 5% over the baseline. This stock's own out-of-sample coverage is within ±7 pp. Data quality is GOOD, the state is inside the training domain, and market context is available. |
| `MEDIUM_CONFIDENCE` | At least 5 sessions of evidence, coverage within ±10 pp, positive skill, and nothing disqualifying. |
| `INSUFFICIENT_EDGE` | No range is given because of one of: an unvalidated state, a state that was miscalibrated out of sample, no skill over the baseline, features outside the training domain, missing market context, or fewer than 15 minutes left in the session. |
| `WARMING_UP` | There is no trained model yet, or too little stored history for this stock. |
| `BAD_DATA` | Network or endpoint failure, a malformed payload, candles that failed validation, or stale data (the last candle is more than 3 minutes old while the market is open). |
| `MARKET_CLOSED` | There is no live next-60-minute window. Use `--analysis` or `--as-of` instead. |

No status carries a made-up percentage. The only percentages shown are
nominal coverage levels and realised out-of-sample statistics.

---

## 3. Architecture

```
psygrid-range/
├── engine/
│   ├── config.py        engine.json + universe loading, paths (PSYGRID_HOME)
│   ├── ingestion.py     PHASE 1  HTTP fetch, timeouts/retries, payload parsing, TTL cache
│   ├── validation.py    PHASE 1  ordering, duplicates, OHLC, session, gaps, freshness
│   ├── storage.py       PHASE 2  data/raw/<SYMBOL>/<YYYY-MM-DD>.csv merge-store
│   ├── features.py      PHASE 3  causal features + NIFTY/sector context
│   ├── regime.py                 multi-feature regime / volatility / structure / RS labels
│   ├── targets.py       PHASE 4  next-60-minute high/low excursions
│   ├── dataset.py                one code path for training and live frames
│   ├── models.py        PHASE 5/6 empirical baseline + LightGBM quantile models (pluggable)
│   ├── calibration.py   PHASE 7  joint conformalised quantile calibration
│   ├── pipeline.py               fit/predict/persist the model + calibrator + domain
│   ├── walkforward.py   PHASE 8  chronological folds, purge + embargo
│   ├── evaluation.py    PHASE 8  forecast records and metrics by time/regime/vol/stock
│   ├── confidence.py             evidence-based forecast status
│   ├── training.py               warm-up gate -> backtest -> final model
│   ├── forecasting.py   PHASE 9  RangeEngine.predict(symbol)
│   ├── output.py                 terminal + JSON rendering
│   └── cli.py           PHASE 10 shared CLI used by every script
├── config/engine.json, config/nifty50.json
├── data/raw | processed | calibration | cache
├── models/<model_version>/
├── cipla.py tcs.py infy.py reliance.py hdfcbank.py   (5-line wrappers)
├── range.py train.py backtest.py collect.py import_history.py
└── tests/
```

There is one engine. `cipla.py` is literally `run("CIPLA")`. The universe
is defined only in `config/nifty50.json`: edit it at NSE rebalances, or
point `universe_file` at a larger list to expand beyond NIFTY-50.

### Data handling
* Timestamps can be epoch seconds or milliseconds, or ISO strings. Everything is converted to explicit `Asia/Kolkata`.
* Malformed records are dropped and counted. Duplicates are collapsed (identical) or resolved to the latest record and flagged (conflicting). OHLC-violating candles and candles outside the session are dropped. Out-of-order records are re-sorted and flagged.
* **Missing minutes are never filled.** Features sit on a minute grid where a gap stays NaN.
* A still-forming last candle is discarded. Data older than 3 minutes during market hours is refused. The cache TTL is 20 s, and stale cache is never used.

### Features (all strictly causal; `tests/test_features.py` verifies this by altering the future)
Most features are normalised by the stock's own volatility, so one pooled model can learn across stocks.

* **Returns:** 1/3/5/10/15/30/60-minute returns (within the session), acceleration, return since the open, overnight gap, distance from the 30/60-minute and session high/low, position within the session range.
* **Volatility:** realised vol (15/30/60), EWMA vol (30/120), volatility relative to the stock's previous 5 sessions, vol acceleration, ATR (as a feature only), range compression/expansion, vol-of-vol.
* **Candles:** body, wicks, body/range ratio, bullish fraction, signed run length, reversal patterns (hammer/shooting-star/engulfing after a move).
* **Volume:** volume ratios, z-score, acceleration, signed volume, and relative volume versus the time-of-day profile of the prior 10 sessions. Volume enters only as state, never as a hard-coded "bullish" rule.
* **Trend / structure:** EMA 9/20/50 distance, spreads and slopes, VWAP distance, trend persistence, higher-high/higher-low score, breakout/breakdown versus the prior 60-minute range, rejection wicks.
* **Momentum:** RSI 14, momentum acceleration, z-distance from the 20-minute mean.
* **Time:** minutes since open and to close, time-of-day bucket (OPEN / MORNING / MIDDAY / AFTERNOON / CLOSE), weekday, and intraday seasonality of volatility (current and expected over the next hour) from prior sessions.
* **Context:**
  * NIFTY proxy: an equal-weight average of 1-minute returns across the stored or fetched NIFTY-50 constituents. PSYGRID exposes no index endpoint; set `index_symbol` in `nifty50.json` if one becomes available.
  * Market returns and volatility state.
  * Stock-versus-market relative strength (5/15/30/60-minute), rolling beta and correlation.
  * Leave-one-out sector relative strength.

### Regime layer
The regime is decided by several vol-normalised signals voting together
(EMA alignment, EMA slope, VWAP, structure, persistence, return
t-statistics, volume and range dynamics):

* **Primary regime:** `BREAKOUT`, `BREAKDOWN`, `REVERSAL`, `VOLATILITY_EXPANSION`, `TREND_UP`, `TREND_DOWN`, `COMPRESSION` or `RANGE`.
* **Volatility state:** HIGH / NORMAL / LOW, versus the stock's own recent sessions.
* **Structure** and **relative strength** labels.

Regime and volatility codes are model inputs. The labels are also used to
slice calibration diagnostics and to choose the evidence cell for the status.

### Target
For every minute *t*, with price = the close of candle *t*:

* `upper_excursion = (max high[t+1..t+60] − price) / price`
* `lower_excursion = (price − min low[t+1..t+60]) / price`

The window stops at the close (the horizon is truncated after 14:30 and shown
as such) and needs at least 90% of its candles to be present. For pooling,
excursions are divided by `sigma_t * sqrt(H)`, where `sigma_t` is the causal
per-minute volatility.

### Model
**Global model with stock-specific features and stock-specific calibration.**

* LightGBM quantile regression, with one booster per quantile (P10/P25/P50/P75/P90) **per side**.
* Stock-specific inputs: volatility level, turnover level, sector, beta.
* Quantile crossing is removed by monotone rearrangement.
* The baseline is the empirical per-time-of-day quantiles of the same vol-scaled target. "Skill" is the pinball-loss improvement over that baseline, a deliberately non-trivial bar.
* Other models plug in through `MODEL_REGISTRY` in `models.py`. A per-stock model can be added the same way once a stock has enough history. `use_symbol_feature` enables a symbol categorical.

### Calibration
This is joint conformalised quantile regression on a held-out calibration
block:

1. Start from each side's quantile at τ = (1+level)/2.
2. Compute the score `s = max(y_up − Q_up, y_lo − Q_lo)`.
3. Take the ⌈(n+1)·level⌉-th smallest score as the offset, and add it to both sides. The two sides keep different base quantiles, so asymmetry is preserved.

The offset is estimated per time-of-day bucket (Mondrian). It is then
shifted toward each stock's own calibration offset with shrinkage
`w = n_sessions / (n_sessions + 10)`. The displayed P10–P90 quantiles get
one-sided conformal corrections.

### Validation
* Chronological walk-forward only: an expanding train window, then embargo, calibration, embargo, and test. Folds are anchored at the most recent data.
* Labels never cross a session close. Purging by label end time is also enforced explicitly, and whole-session embargoes separate the blocks.
* Every test forecast is written to `data/processed/backtest_<version>.csv` with its timestamp, symbol, price, predicted low/high, actual future low/high, state labels and losses.
* The summary reports coverage (joint and per side), width, pinball loss, skill, MAE of the median excursion, and quantile calibration. Each is broken down **by time of day, regime, volatility state and stock**.

---

## 4. Tests

```powershell
python -m pytest -q
```

There are 53 tests covering every phase: parsing and malformed payloads,
timeouts, cache TTL, duplicates, gaps, stale data, storage merges, feature
causality, target brute-force checks, regimes, quantile models, conformal
coverage, fold/embargo structure, end-to-end live forecasts through a mocked
HTTP layer, every refusal path, and the CLI.

Tests use `tests/synthetic.py`, a synthetic market generator. **Synthetic
results only prove the plumbing works. They say nothing about real-market
accuracy.** The generator lives in `tests/` and cannot feed a production
forecast.

## 5. Known limitations / next steps
* There is no real history yet, so no accuracy claim can be made. Collect or import at least 32 sessions, then read the `train.py` report.
* The NIFTY context is an equal-weight proxy of the constituents, not the official cap-weighted index. A live forecast fetches all 50 stocks (about 5 s with 8 workers; skip with `--no-context`).
* Minute-level samples are autocorrelated, so conformal guarantees are approximate. That is why realised coverage per state is what gates the status.
* There is no exchange holiday calendar. On a holiday the engine reports stale or `BAD_DATA` rather than forecasting.
* `config/nifty50.json` reflects the constituents as of the date in its `as_of` field. Verify it against NSE after each rebalance.
