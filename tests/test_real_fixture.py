"""Parser checks against REAL captured PSYGRID responses.

Captures are created on a machine that can reach the endpoint:

    python diagnose_endpoint.py CIPLA --save-fixture

and land in tests/fixtures/captured/.  Without captures these tests skip -
no response is ever invented to stand in for one.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

from engine.config import load_config
from engine.ingestion import parse_payload
from engine.validation import clean_candles

CAPTURED = sorted((Path(__file__).parent / "fixtures" / "captured").glob("*.json"))


@pytest.mark.skipif(not CAPTURED, reason="no captured real PSYGRID responses (run diagnose_endpoint.py --save-fixture)")
@pytest.mark.parametrize("path", CAPTURED, ids=[p.stem for p in CAPTURED])
def test_captured_response_parses(path, tmp_path):
    cfg = load_config(root=tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    snap = parse_payload(doc, path.stem.upper(), cfg, datetime.now(cfg.tz))
    assert len(snap.candles) > 0, "no candles parsed from a real response"
    assert snap.n_malformed == 0, f"malformed records: {snap.malformed_examples}"
    assert not snap.timestamp_format.startswith("MIXED")
    assert str(snap.candles["timestamp"].dt.tz) == "Asia/Kolkata"
    clean, rep = clean_candles(snap.candles, cfg, snap.symbol, schema_issues=snap.schema_issues, single_session=True)
    assert rep.n_ohlc_violations == 0 and rep.quality in ("GOOD", "DEGRADED"), rep.issues
