"""Stage 0 tests for the research ledger plumbing (Phase 26).

Covers the two foundation pieces, both shipped dark (nothing calls them yet):
  - OllamaClient.chat forwards the `format` (JSON-schema) field to the payload.
  - ledger.py parsers tolerate clean / fenced / prose-wrapped / garbage output
    and never raise.
"""
from __future__ import annotations

import json

import httpx
import pytest

from audrey.models.ollama import OllamaClient
from audrey.pipeline.ledger import (
    Claim,
    FactCheckResult,
    ResearchResult,
    Source,
    inlined_schema,
    parse_factcheck_result,
    parse_research_result,
)


class TestInlinedSchema:
    """Ollama `format` constrained-decoding chokes on $ref/$defs across some
    cloud models; the inlined schema must be self-contained."""

    def test_research_result_schema_is_ref_free(self):
        s = inlined_schema(ResearchResult)
        txt = json.dumps(s)
        assert "$ref" not in txt
        assert "$defs" not in txt

    def test_factcheck_result_schema_is_ref_free(self):
        s = inlined_schema(FactCheckResult)
        txt = json.dumps(s)
        assert "$ref" not in txt
        assert "$defs" not in txt

    def test_nested_claim_object_is_inlined(self):
        # The Claim sub-object must appear expanded inline under claims.items,
        # not as a reference.
        s = inlined_schema(ResearchResult)
        claim_props = s["properties"]["claims"]["items"]["properties"]
        assert {"id", "text", "source_ids", "risk", "needs_hedge"} <= set(claim_props)

    def test_inlined_schema_still_validates_real_data(self):
        # A round-trip sanity check: data valid against the model parses fine
        # (inlining is for the wire format, not validation).
        raw = json.dumps({"summary_notes": "n", "claims": [
            {"id": "c1", "text": "x", "risk": "low"}], "sources": []})
        assert parse_research_result(raw) is not None


def _client_capturing(captured: dict) -> OllamaClient:
    """An OllamaClient whose transport records the request payload and returns
    a minimal valid /api/chat response."""
    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    client = OllamaClient(base_url="http://unused")
    client._client = httpx.AsyncClient(
        base_url="http://unused", transport=httpx.MockTransport(handler))
    return client


class TestChatFormatForwarding:
    @pytest.mark.asyncio
    async def test_format_schema_forwarded(self):
        captured: dict = {}
        client = _client_capturing(captured)
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        await client.chat(model="m", messages=[{"role": "user", "content": "hi"}],
                          format=schema)
        assert captured["payload"]["format"] == schema

    @pytest.mark.asyncio
    async def test_format_string_forwarded(self):
        captured: dict = {}
        client = _client_capturing(captured)
        await client.chat(model="m", messages=[{"role": "user", "content": "hi"}],
                          format="json")
        assert captured["payload"]["format"] == "json"

    @pytest.mark.asyncio
    async def test_format_omitted_when_none(self):
        captured: dict = {}
        client = _client_capturing(captured)
        await client.chat(model="m", messages=[{"role": "user", "content": "hi"}])
        assert "format" not in captured["payload"]


class TestParseResearchResult:
    def test_clean_json(self):
        raw = json.dumps({
            "summary_notes": "notes",
            "claims": [{"id": "c1", "text": "x", "source_ids": ["s1"], "risk": "high"}],
            "sources": [{"id": "s1", "title": "T", "url": "https://e.com",
                         "source_type": "official", "supports": ["c1"]}],
        })
        r = parse_research_result(raw)
        assert isinstance(r, ResearchResult)
        assert r.claims[0].id == "c1"
        assert r.claims[0].risk == "high"
        assert r.sources[0].source_type == "official"

    def test_code_fenced_json(self):
        inner = json.dumps({"summary_notes": "n", "claims": [], "sources": []})
        raw = f"Here is the result:\n```json\n{inner}\n```\nDone."
        r = parse_research_result(raw)
        assert isinstance(r, ResearchResult)
        assert r.summary_notes == "n"

    def test_prose_wrapped_braces(self):
        inner = json.dumps({"summary_notes": "n", "claims": [], "sources": []})
        raw = f"Sure! {inner} Hope that helps."
        r = parse_research_result(raw)
        assert isinstance(r, ResearchResult)

    def test_defaults_fill_missing_fields(self):
        # Only summary_notes given — claims/sources default to empty.
        r = parse_research_result('{"summary_notes": "just notes"}')
        assert isinstance(r, ResearchResult)
        assert r.claims == []
        assert r.unresolved_questions == []

    def test_garbage_returns_none(self):
        assert parse_research_result("not json at all") is None

    def test_empty_returns_none(self):
        assert parse_research_result("") is None

    def test_wrong_shape_returns_none(self):
        # Valid JSON, wrong type for claims → ValidationError → None, not a raise.
        assert parse_research_result('{"claims": "should-be-a-list"}') is None


class TestParseFactCheckResult:
    def test_clean(self):
        raw = json.dumps({
            "checks": [{"claim_id": "c1", "verdict": "unsupported",
                        "corrected_text": None, "notes": "Conics is lost"}],
            "fatal_errors": [],
        })
        r = parse_factcheck_result(raw)
        assert isinstance(r, FactCheckResult)
        assert r.checks[0].verdict == "unsupported"

    def test_bad_verdict_returns_none(self):
        raw = json.dumps({"checks": [{"claim_id": "c1", "verdict": "made-up"}]})
        assert parse_factcheck_result(raw) is None

    def test_garbage_returns_none(self):
        assert parse_factcheck_result("¯\\_(ツ)_/¯") is None

    def test_bare_top_level_array_wrapped(self):
        # Observed on the box: the model returns a bare array of checks, not the
        # {"checks": [...]} object. We wrap it.
        raw = json.dumps([
            {"claim_id": "w0_c9", "verdict": "needs_hedge",
             "corrected_text": "reportedly released around late 2024"},
            {"claim_id": "w0_c3", "verdict": "supported"},
        ])
        r = parse_factcheck_result(raw)
        assert isinstance(r, FactCheckResult)
        assert len(r.checks) == 2
        assert r.checks[0].verdict == "needs_hedge"

    def test_fenced_array(self):
        raw = "```json\n[{\"claim_id\": \"c1\", \"verdict\": \"supported\"}]\n```"
        r = parse_factcheck_result(raw)
        assert isinstance(r, FactCheckResult)
        assert r.checks[0].claim_id == "c1"


def test_models_construct_directly():
    # Sanity: the models are usable as plain Python objects too.
    s = Source(id="s1", title="t", url="https://e.com", source_type="company_claim")
    c = Claim(id="c1", text="beats GPT-4o", source_ids=["s1"], risk="high")
    assert s.source_type == "company_claim"
    assert c.needs_hedge is False
