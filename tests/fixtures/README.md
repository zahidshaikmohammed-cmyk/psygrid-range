# Test fixtures

Everything in this folder is **test-only** and must never be imported into
`data/raw/` or used by the production engine.

| File | What it is |
|---|---|
| `synthetic.py` | Synthetic 1-minute market generator used by the unit tests. Results on it prove the plumbing works, **not** real-market accuracy. |
| `documented_schema_CIPLA.json` | A document **constructed by hand from the documented field list** (`service`, `schema_version`, `status`, `symbol`, `security_id`, `previous_close`, `today_open`, `candles_1m`). It is **not** a captured PSYGRID response; its values (`security_id`, `schema_version`, prices) are placeholders. |
| `transcribed_CIPLA_20260929.json` | The first three **real** candle records from a live CIPLA response on 2026-09-29, as printed by `diagnose_endpoint.py` ("Malformed examples"), plus the envelope values it showed (`PSYGRID`, schema `4.0`, status `OK`, security ID `694`, previous close 1383.7, today open 1390.0). Transcribed by hand from a screenshot, not a byte-exact capture: the JSON types of `schema_version` / `security_id` were not visible. It pins the real timestamp format `"YYYY-MM-DD HH:MM:SS IST"`. |
| `captured/` | Real PSYGRID responses. Create them on a machine that can reach the endpoint with `python diagnose_endpoint.py CIPLA --save-fixture`; `tests/test_real_fixture.py` then parses every file here with the production parser. The folder is empty in the repository because the build sandbox could not reach the endpoint. |
