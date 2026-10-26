"""Reader references retain exact PDF text without model transcription."""
from __future__ import annotations

import dataclasses
import json
from io import BytesIO

import pytest
from openpyxl import load_workbook
from test_tax_review import _call, _create
from test_tax_review import state as state

from audrey.processing_policy import processing_scope
from audrey.tax_exports import TaxReviewExports
from audrey.tax_review import TaxEvidenceReadings, tax_review_spec
from audrey.tools.private_dispatch import PrivateToolArgumentsError
from audrey.tools.private_handlers import make_private_tool_handlers

TEXT = "Termination date. The termination date changed from December 31, 2032,\nto June 30, 2026. Y ou can't claim the credit after that date."


def _policy(state):
    return dataclasses.replace(state.policy, allowed_tools=frozenset({"tax_review", "read_reference"}))


def _handlers(state):
    return make_private_tool_handlers(state.app, state.owner,
                                     tax_conversation_id=state.conversation.conversation_id)


async def _read(state, handlers, **extra):
    return await handlers["read_reference"]({"document_id": state.metadata["document_id"], "page": 1, **extra})


@pytest.mark.parametrize("state", [TEXT], indirect=True)
async def test_exact_reader_capture_survives_pdf_spacing_and_reaches_workbook(state, tmp_path):
    await _create(state)
    with processing_scope(_policy(state)):
        handlers = _handlers(state)
        read = await _read(state, handlers)
        assert "Y ou" in read["text"]
        # Rewriting the PDF artifact remains invalid; no fuzzy matching is added.
        legacy = await handlers["tax_review"]({"operation": "record_evidence", "expected_revision": 1, "data": {
            "claim": "Cutoff passage", "document_id": read["document_id"],
            "revision": read["revision_sha256"], "reference_release": state.release["id"],
            "page": 1, "passage": read["text"].replace("Y ou", "You"),
        }})
        assert legacy["error"] == "tax_evidence_passage_not_found"
        saved = await handlers["tax_review"]({"operation": "record_evidence", "expected_revision": 1,
            "data": {"claim": "Retained source passage", "source_ref": read["source_ref"]}})
        assert saved["persistence"]["status"] == "saved"
    case = (await state.client.get(state.path)).json()["case"]
    assert case["evidence"][0]["passage"] == read["text"].strip()
    assert "December 31, 2032" in case["evidence"][0]["passage"]
    assert "June 30, 2026" in case["evidence"][0]["passage"]
    assert "source_ref" not in case["evidence"][0]
    review = await _call(state, "save_review", expected_revision=saved["revision"], data={
        "summary": "Retained the selected official excerpt; other topics unresolved.",
        "findings": [{"label": "Source passage", "detail": "The selected passage is retained verbatim.",
                      "evidence_ids": [saved["updated_entry"]["id"]]}], "next_question": "",
    })
    assert review["persistence"]["status"] == "saved"
    state.app.state.tax_review_exports = TaxReviewExports(tmp_path / "exports")
    prepared = await state.client.post(state.path + "/exports", json={"expected_revision": review["revision"]})
    assert prepared.status_code == 200, prepared.text
    downloaded = await state.client.get(prepared.json()["export"]["download_url"])
    assert downloaded.status_code == 200
    workbook = load_workbook(BytesIO(downloaded.content), read_only=True)
    try:
        values = [cell for row in workbook["Evidence"].values for cell in row]
        assert read["text"].strip() in values
    finally:
        workbook.close()


@pytest.mark.parametrize("override", [{"passage": "changed"}, {"page": 2}, {"revision": "b" * 64}, {"tax_year_status": "verified"}])
async def test_reference_cannot_override_source_or_applicability(state, override):
    await _create(state)
    with processing_scope(_policy(state)):
        handlers = _handlers(state)
        read = await _read(state, handlers)
        with pytest.raises(PrivateToolArgumentsError):
            await handlers["tax_review"]({"operation": "record_evidence", "expected_revision": 1,
                "data": {"claim": "Source", "source_ref": read["source_ref"], **override}})
    assert (await state.client.get(state.path)).json()["case"]["revision"] == 1


async def test_forged_and_previous_run_references_are_refused(state):
    await _create(state)
    with processing_scope(_policy(state)):
        first = _handlers(state)
        read = await _read(state, first)
        second = _handlers(state)
        for reference in ("taxsrc_forged", read["source_ref"]):
            result = await second["tax_review"]({"operation": "record_evidence", "expected_revision": 1,
                "data": {"claim": "Source", "source_ref": reference}})
            assert result["error"] == "tax_evidence_source_ref_invalid"
    assert (await state.client.get(state.path)).json()["case"]["revision"] == 1


async def test_capture_rechecks_exact_revision_after_read(state, monkeypatch):
    await _create(state)
    with processing_scope(_policy(state)):
        handlers = _handlers(state)
        read = await _read(state, handlers)
        monkeypatch.setattr(state.app.state.reference_releases, "document", lambda *_: None)
        result = await handlers["tax_review"]({"operation": "record_evidence", "expected_revision": 1,
            "data": {"claim": "Source", "source_ref": read["source_ref"]}})
        assert result["error"] == "tax_evidence_not_in_release"
    assert (await state.client.get(state.path)).json()["case"]["evidence"] == []


def test_bounded_capture_matches_the_visible_excerpt_and_skips_historical_reads():
    readings = TaxEvidenceReadings()
    source = {"library_id": "irs-federal", "catalog": "current", "document_id": "a" * 64,
              "revision_sha256": "b" * 64, "page": 1, "offset": 17, "text": "word " * 800}
    captured = readings.offer(source)
    assert len(captured["text"]) <= 2000
    assert captured["has_more"] and captured["next_offset"] == 17 + len(captured["text"])
    assert readings.resolve(captured["source_ref"])["passage"] == captured["text"]
    assert "source_ref" not in readings.offer({**source, "catalog": "prior"})
    old = captured["source_ref"]
    for _ in range(64):
        readings.offer(source)
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        readings.resolve(old)
    assert len(readings._readings) == 64
    assert "source_ref" in json.dumps(tax_review_spec("http://custom-tools:8001").parameters)


def test_capture_metadata_never_displaces_an_otherwise_readable_source():
    source = {"library_id": "irs-federal", "catalog": "current", "document_id": "a" * 64,
              "revision_sha256": "b" * 64, "page": 1, "offset": 0, "text": "Public source text.",
              "description": ""}
    source["description"] = "x" * (3499 - len(json.dumps(source)))
    assert len(json.dumps(source)) == 3499
    assert TaxEvidenceReadings().offer(source) == source
