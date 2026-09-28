# PSYGRID RANGE ENGINE

A probabilistic engine that estimates, from genuine 1-minute OHLCV, where a
NIFTY-50 stock is likely to trade over the **next 60 minutes**.

It forecasts the distributions of the next-hour **high** and **low**
separately, using quantile models conditioned on the current market state.
It calibrates them on held-out data, validates them walk-forward against
three baselines, and refuses to output a range when the evidence is weak.

```
NO RELIABLE FORECAST / INSUFFICIENT EDGE
```

That refusal is a successful result, not a failure.

> **Current status: WARM-UP. Real-data accuracy: NOT ESTABLISHED.**
> The code has only been exercised on synthetic data, because the build
> sandbox cannot reach the PSYGRID endpoint. Synthetic results show that the
> mechanics work. They are **not** evidence of real-market skill. Real
> evidence exists only after you collect real history, run `train.py`, and it
> reports positive, statistically distinguishable out-of-sample skill.

---

## 1. Workflow (Windows PowerShell)

### First time

```powershell
cd C:\path\to\psygrid-range
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                      # optional: run the test suite
```

If activation is blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### 1. Test the real endpoint

```powershell
python diagnose_endpoint.py CIPLA
python diagnose_endpoint.py CIPLA --save-fixture   # also saves tests\fixtures\captured\CIPLA.json
python -m pytest -q tests\test_real_fixture.py     # parse the real capture with the production parser
```

This makes a real HTTP request from your machine and reports:

* HTTP status and response time
* service, schema, endpoint status and security ID
* candle count, first and last candle, and latest price
* latest candle age
* duplicates, missing minutes and impossible OHLC
* timestamp format and timezone
* data quality

If the request fails, it reports `BAD_DATA` and shows `n/a` for every value
it could not measure. Nothing is filled in.

Every live response is classified into one **data state**:

| State | Meaning |
|---|---|
| `OK` | Valid, current candles. |
| `MARKET_CLOSED` | HTTP 200 with a healthy envelope but no current-session candles (for example the minimal `service`/`status`/`symbol` document served outside market hours), or valid candles after the close. Nothing is inferred from it and nothing is stored. Data quality is `NOT_APPLICABLE` when no candles were served. |
| `STALE_DATA` | Valid candles, but the latest is older than the freshness limit while the market is open. Forecasts are refused (`BAD_DATA` status, `data_state` = `STALE_DATA`). |
| `BAD_DATA` | Network failure, invalid JSON, missing `status`/`symbol`, a symbol mismatch, an error or unrecognised `status`, `candles_1m` that is not a list, candles failing validation, or no candles while the endpoint claims to be live during session hours. |

Recognised status values are listed in `config\engine.json` under
`data_source.ok_statuses` and `data_source.closed_statuses`. If the endpoint
uses another value to mean "closed", the diagnostic prints it (`Raw Status
Field`) and explains how to add it.

A closed-market response verifies the **connection** only. The candle schema
is reported as `NOT VALIDATED` until a response with candles has been
parsed; run the diagnostic again during market hours.

A different server can be used without editing files:

```powershell
$env:PSYGRID_BASE_URL = "http://140.245.226.102:10000"
```

### 2. Collect real history (every trading day)

```powershell
python collect.py --all-nifty50          # all 50 symbols from config\nifty50.json
python collect.py --symbol CIPLA         # one symbol (repeatable)
```

* The live endpoint only serves the current session. Run the collector every
  trading day, ideally once shortly after the 15:30 close.
* Re-running is safe: candles merge by timestamp, never duplicate, and are
  never interpolated.
* Raw responses are archived in `data\raw\live\<SYMBOL>_<YYYYMMDD>.json`. A
  smaller later capture never overwrites a fuller one.
* Every `cipla.py` run also stores what it fetched.

To schedule it with Task Scheduler, create a daily task at 15:40, Monday to Friday:
* Program: `C:\path\to\psygrid-range\.venv\Scripts\python.exe`
* Arguments: `collect.py --all-nifty50`
* Start in: `C:\path\to\psygrid-range`

If you have genuine historical 1-minute CSVs, you can import them. Each file
is named `<SYMBOL>.csv` with columns `timestamp,open,high,low,close,volume`:

```powershell
python import_history.py --dir D:\exports
```

Never import synthetic data.

### 3. Check history

```powershell
python train.py --status
```

This shows:
* sessions and stocks available, total candles, last collected date, and data sources
* model version, last training date and calibration status
* validation coverage and skill against the best baseline
* which stocks do and do not have a demonstrated edge
* one line: **Real-data accuracy**

### 4. Train (once the minimum real history exists)

```powershell
python train.py
```

32 sessions is the minimum needed to **begin** evaluation: 20 train +
5 calibration + 5 test + 2 embargo. It is not evidence of accuracy.

With about 60 sessions there are more walk-forward folds and per-state
evidence, so calibration becomes more reliable. That is still not evidence
of accuracy by itself.

Only the out-of-sample report decides. Training never runs as part of a
forecast.

### 5. Evaluate

```powershell
python backtest.py
python backtest.py --json backtest_summary.json
```

### 6. Live forecast

```powershell
python cipla.py
python tcs.py
python range.py CIPLA
python range.py RELIANCE --json
```

| Flag | Meaning |
|---|---|
| `--json` / `--save-json PATH` | Machine-readable output |
| `-v` | Quantile table, data issues, context details, raw-archive path |
| `--as-of "2026-09-28 12:30"` | Replay from local history, with no network. Only candles before that time are used. Shows what actually happened and is labelled **OUT-OF-SAMPLE** or **IN-SAMPLE**. |
| `--remaining-session` | After 14:30, gives an explicitly labelled forecast up to the close. The standard command never calls a shorter window "60 minutes". |
| `--analysis` | When the market is closed, evaluates the last stored state (labelled ANALYSIS) |
| `--no-context` | Skip fetching the other 49 stocks. The forecast is then refused if the model needs context. |
| `--ascii` | Print `Rs.` instead of `₹` |

---

## 2. Coverage is not accuracy

| Term | Meaning |
|---|---|
| **Coverage** | The fraction of held-out hours whose *entire* next-hour high **and** low stayed inside the range. It checks *calibration*. A very wide range always has perfect coverage and is useless. |
| **Width** | How wide the range is (% of price). Narrower is more useful at equal coverage. |
| **Skill** | `1 − pinball_loss(model) / pinball_loss(best baseline)`, out-of-sample. It shows whether the model knows more than simple rules. |
| **Demonstrated edge** | Skill > 0 **and** a per-session t-statistic ≥ 2 **and** coverage within tolerance, over ≥ 10 test sessions for a stock. The session is the unit of evidence because minutes within a session are strongly dependent. |

Examples:
* "78.5% coverage vs 80% nominal" on synthetic data means the calibration
  arithmetic works. It is **not** "78.5% accuracy".
* "+0.4% skill, t = 0.6" means **NO DEMONSTRATED EDGE**.

### Baselines (never tuned to flatter the model)
* `empirical_tod`: empirical excursion quantiles for each time-of-day bucket.
* `volatility_only`: one set of quantiles times the current volatility estimate.
* `recent_realized_vol`: quantiles times the last-60-minute realised volatility.

The model is compared with the **best** of these by mean loss in every group
(overall, per stock, per regime × time cell).

---

## 3. Forecast statuses (no invented percentages)

| Status | When |
|---|---|
| `HIGH_CONFIDENCE` | All of the following hold: <br>• The current *(regime, time-of-day)* cell has at least 20 out-of-sample sessions. <br>• Coverage in that cell is within ±5 pp of nominal. <br>• Skill is at least 5% with a session t-statistic of at least 2. <br>• The stock's own out-of-sample skill has t ≥ 2 and coverage within ±7 pp. <br>• Data is GOOD, the state is in the training domain, and context is available. |
| `MEDIUM_CONFIDENCE` | At least 5 sessions of evidence, coverage within ±10 pp, skill > 0 with t ≥ 1, and nothing disqualifying. |
| `INSUFFICIENT_EDGE` | No range is given. Causes: <br>• the state is unvalidated or miscalibrated <br>• **NO DEMONSTRATED EDGE** in that state, or for that stock <br>• features are outside the training domain <br>• market context is insufficient or its source doesn't match the model <br>• fewer than 60 minutes remain (unless `--remaining-session` is used) |
| `WARMING_UP` | No trained model, a model trained on synthetic or unlabelled data, or too little stored history. |
| `BAD_DATA` | Network or HTTP failure, a malformed payload, candles failing validation, or stale data (the last candle closed more than 3 minutes ago while the market is open). |
| `MARKET_CLOSED` | There is no live next-60-minute window. |

The only percentages ever printed are nominal calibration targets and
measured out-of-sample statistics. All thresholds are in
`config\engine.json` under `confidence`.

---

## 4. Output

```
============================================================
                PSYGRID 1-HOUR RANGE ENGINE
============================================================

SYMBOL              : CIPLA
CURRENT PRICE       : ₹XXXX.XX
DATA TIME           : YYYY-MM-DD HH:MM:SS IST  (last complete candle HH:MM)
SYSTEM TIME         : YYYY-MM-DD HH:MM:SS IST
DATA AGE            : X.X min after candle close

FORECAST HORIZON
NEXT 60 MINUTES

------------------------------------------------------------
PROBABLE RANGE  (calibration target: 80% of held-out hours stayed fully inside)
------------------------------------------------------------
LOW                 : ₹XXXX.XX
HIGH                : ₹XXXX.XX
CENTRAL RANGE ...
UPSIDE EXTENSION    : +X.XX%
DOWNSIDE EXTENSION  : -X.XX%
------------------------------------------------------------
MARKET REGIME / VOLATILITY STATE / STRUCTURE / RELATIVE STRENGTH / FORECAST STATUS
STATE (inputs the model saw; descriptive, not trading rules)
MODEL EVIDENCE  (historical samples, regime match, market context, data quality, version)
============================================================
```
*(These are layout placeholders, not real values.)*

The JSON output (`--json`) contains:
* `symbol`, `timestamp`, `current_price`, `horizon_minutes`, `forecast_mode`
* `range`, `central_range`, `upside_pct`, `downside_pct`, `quantiles` (P10–P90)
* `regime`, `volatility_state`, `structure`, `relative_strength`, `explanation`
* `forecast_status`, `status_reasons`, `evidence`, `model_evidence`, `replay_sample`
* `data_quality`, `data_issues`, `freshness`, `market_context`, `model_version`

---

## 5. Data handling

* **Source:** `data_source.base_url` + `/public/stock/{SYMBOL}.json`. The environment variable `PSYGRID_BASE_URL` overrides the base URL.
* **Network:**
  * separate connect (5 s) and read (10 s) timeouts, with bounded retries
  * HTTP 429 and 5xx are retried with back-off and a bounded `Retry-After`
  * responses over 20 MB are rejected
  * nothing waits indefinitely
* **Untrusted input:** the endpoint is only parsed as JSON data. Nothing from it is executed, evaluated or passed to a shell. Symbols are checked against `config\nifty50.json` before a URL is built.
* **Schema:**
  * `status`, `symbol` and `candles_1m` are required. A wrong status or symbol mismatch is rejected.
  * Missing documented fields (`service`, `schema_version`, `security_id`, `previous_close`, `today_open`) or non-numeric prices are flagged, and quality becomes DEGRADED.
* **Candles:**
  * **Timestamps:** epoch seconds or milliseconds, or ISO with/without an offset. The detected format is reported. Naive times are treated as IST, and everything is converted to `Asia/Kolkata`.
  * **Duplicates:** identical ones are collapsed. Conflicting ones are flagged, and the latest record is kept.
  * **Dropped and flagged, never repaired:** impossible OHLC (high < max(open, close), low > min(open, close), high < low), negative volume, non-positive prices, candles outside the session, and candles stamped in the future.
  * **Forming candle:** a candle still in progress is discarded.
  * **Missing minutes** stay missing: they are counted, never filled.
* **Freshness:** the engine tracks the latest candle, system time and age. Data older than 3 minutes during market hours is refused as `BAD_DATA`.
* **Provenance:**
  * Every stored session records its source (`psygrid_live`, `import:<file>` or `synthetic`) in `data\raw\<SYMBOL>\_provenance.json`.
  * A model trained on any synthetic or unlabelled session refuses production forecasts (`WARMING_UP`).
  * Synthetic generators live only in `tests\fixtures\`.

### NIFTY50_PROXY: not the NIFTY 50 index
PSYGRID exposes no index endpoint, so market context comes from
**`NIFTY50_PROXY`**. This is an *equal-weight* average of the constituents'
1-minute returns, built by this engine. It is **not** the official
free-float cap-weighted NIFTY 50, and it is labelled as a proxy wherever it
appears.

If a genuine index OHLCV feed becomes available:
1. Set `index_symbol` in `config\nifty50.json`.
2. Set `market_context.source` to `NIFTY50_INDEX`.
3. Collect its history and retrain.

The two sources are never mixed. A model records its context source and
refuses a different one at inference.

For live forecasts, the other 49 stocks are fetched with:
* at most 6 concurrent requests
* 3 s connect / 6 s read timeouts and no retries
* a hard 25 s overall deadline

Fetched, failed and stale counts are reported. If fewer than 60% of
constituents are present at the forecast minute, data quality is downgraded
and the forecast is refused.

---

## 6. Model design (unchanged architecture)

* **Target:** for every minute *t*, measured as excursions relative to the close at *t*:
  * `upper = (max high[t+1..t+60] − close_t) / close_t`
  * `lower = (close_t − min low[t+1..t+60]) / close_t`
  * Windows never cross the close. The two sides are modelled separately and never symmetrised.
* **Features:** strictly causal, verified by tests that alter or remove the future. They cover:
  * returns (1–60 minutes)
  * realised and EWMA volatility, range compression, ATR (as a feature only)
  * candle structure and runs, rejection behaviour
  * volume, including relative-to-time-of-day
  * EMA 9/20/50, VWAP, swing structure
  * RSI, ROC and acceleration
  * time of day
  * NIFTY50_PROXY and sector context
* **Regime layer:** a multi-feature vote producing regime, volatility state, structure and relative strength.
* **Model:** a global LightGBM model with one quantile booster per P10/P25/P50/P75/P90 **per side**. It uses stock-specific inputs and is not assumed to be calibrated.
* **Calibration:** joint conformalised quantile calibration on a held-out block, per time-of-day bucket, with shrunk per-stock adjustments. Its guarantees are approximate, because minutes are dependent and markets are non-stationary. That is why realised coverage per state is what gates a forecast.
* **Validation:** chronological walk-forward only, with purge by label end time and whole-session embargoes. Every out-of-sample record goes to `data\processed\backtest_<version>.csv`, with timestamp, symbol, price, predicted low/high, actual low/high, regime, time of day, volatility state and losses.
* **Extension point (not active):** a historical-analog component or an ensemble can be added as another `QuantileModel` and combined before calibration. It would only enter production after the backtest shows it adds out-of-sample value.

### Layout
```
engine/      config ingestion validation storage features regime targets dataset
             models calibration pipeline walkforward evaluation confidence
             training forecasting output cli diagnostics
config/      engine.json  nifty50.json            (the universe lives only here)
data/raw/<SYMBOL>/<date>.csv + _provenance.json   genuine candles
data/raw/live/                                    archived raw endpoint responses
data/processed/  data/calibration/  models/<version>/
tests/       unit tests; tests/fixtures/ = synthetic generator, documented-schema example, captured/
cipla.py tcs.py infy.py reliance.py hdfcbank.py range.py        thin wrappers: run("CIPLA")
diagnose_endpoint.py collect.py import_history.py train.py backtest.py
```

---

## 7. Known limitations

* **No real-data evidence yet.** Nothing here has been validated on real PSYGRID data.
* **Parser not yet checked against the real endpoint.** It follows the documented field list and handles several timestamp formats. Confirm it with `diagnose_endpoint.py CIPLA --save-fixture` and `tests\test_real_fixture.py`.
* **Placeholder schema fixture.** `tests\fixtures\documented_schema_CIPLA.json` was constructed by hand from the field list; it is not a real capture.
* **Proxy, not the index.** NIFTY50_PROXY is not the official index.
* **Slower live runs.** A live forecast queries 50 endpoints, taking a few seconds (at most 25 s).
* **Multiple comparisons.** With 50 stocks, a few stocks will pass t ≥ 2 by chance. Treat per-stock "edge" lists as provisional until they persist across more sessions and folds.
* **No holiday calendar.** On holidays the engine reports stale data or `BAD_DATA` rather than forecasting.
* **Undetectable synthetic imports.** Data imported with `import_history.py` is trusted to be genuine. Provenance cannot detect synthetic data that is imported as if it were real.
* **Universe list needs checking.** `config\nifty50.json` must be checked against NSE after each index rebalance.
