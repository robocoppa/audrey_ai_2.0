"""Keep the manual baseline input separate from its source-checked answer key."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from audrey.tax_review import _calculation

ROOT = Path(__file__).resolve().parents[1]


def test_manual_tax_baseline_is_frozen_without_uploading_the_expected_answer():
    manifest = json.loads((ROOT / "evals/cases/tax_review_baseline.json").read_text())
    assert manifest["language"] == "en" and manifest["synthetic"] is True
    assert len(manifest["cases"]) == 1
    case = manifest["cases"][0]
    raw = (ROOT / case["fixture"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case["fixture_sha256"]
    current, prior = raw.decode().split("PRIOR-YEAR COMPARISON: 2024", 1)
    expected = case["expected"]
    facts = expected["facts"]
    assert f"2025 W-2 box 1 wages, tips and other compensation: {facts['w2_box_1_wages_usd']} USD" in current
    assert f"2025 W-2 box 2 federal income tax withheld: {facts['w2_box_2_withholding_usd']} USD" in current
    assert f"2024 wages: {expected['prior_year_exclusion']['wages_usd']} USD" in prior
    for answer in (expected["rule"]["basic_standard_deduction_usd"], expected["calculation"]["result"]):
        assert answer not in raw.decode() and answer not in case["goal"]
    for source in manifest["sources"]:
        parsed = urlsplit(source["url"])
        assert parsed.scheme == "https" and parsed.hostname == "www.irs.gov"
    sources = {source["id"]: source for source in manifest["sources"]}
    assert set(expected["rule"]["accepted_source_ids"]) <= sources.keys()
    assert sources["irs-p501-2025"]["year_location"]["pdf_page"] == 1
    assert expected["rule"]["saved_tax_year_status"] == "proposed"
    calculation = expected["calculation"]
    computed = _calculation({key: calculation[key] for key in ("label", "inputs", "expression", "rounding")})
    assert computed["result"] == calculation["result"]
