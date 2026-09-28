# Test fixtures

Everything in this folder is **test-only** and must never be imported into
`data/raw/` or used by the production engine.

| File | What it is |
|---|---|
| `synthetic.py` | Synthetic 1-minute market generator used by the unit tests. Results on it prove the plumbing works, **not** real-market accuracy. |
| `documented_schema_CIPLA.json` | A document **constructed by hand from the documented field list** (`service`, `schema_version`, `status`, `symbol`, `security_id`, `previous_close`, `today_open`, `candles_1m`). It is **not** a captured PSYGRID response; its values (`security_id`, `schema_version`, prices) are placeholders. |
| `captured/` | Real PSYGRID responses. Create them on a machine that can reach the endpoint with `python diagnose_endpoint.py CIPLA --save-fixture`; `tests/test_real_fixture.py` then parses every file here with the production parser. The folder is empty in the repository because the build sandbox could not reach the endpoint. |
