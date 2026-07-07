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
    hedge_policy,
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

    def test_unescaped_newline_in_string_value(self):
        # The real box failure: a model puts multi-line prose in a string value
        # with a RAW newline (not \n). strict json.loads rejects it; we parse
        # with strict=False.
        raw = '{"summary_notes": "line one\nline two", "claims": [], "sources": []}'
        r = parse_research_result(raw)
        assert isinstance(r, ResearchResult)
        assert "line one" in r.summary_notes

    def test_claim_without_id_is_backfilled(self):
        # The 2/3-worker-drop bug: models omit `id`. It must default + backfill,
        # not ValidationError-discard the whole ledger.
        raw = '{"claims": [{"text": "Euclid lived ~300 BCE", "risk": "high"}], "sources": []}'
        r = parse_research_result(raw)
        assert isinstance(r, ResearchResult)
        assert r.claims[0].id == "c1"

    def test_integer_source_ids_coerced(self):
        # Models emit source_ids as ints; coerce to str rather than reject.
        raw = '{"claims": [{"id": "c1", "text": "x", "source_ids": [1, 2], "risk": "low"}]}'
        r = parse_research_result(raw)
        assert r.claims[0].source_ids == ["1", "2"]

    def test_title_as_source_id_repaired(self):
        # The 2026-07-06 rust-async failure shape: the model cites a source by
        # its TITLE, not its id — unrepaired, the Sources block and hedge_policy
        # both lose the linkage. Match is case-insensitive.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "medium",
                        "source_ids": ["glommio repository (datadog)"]}],
            "sources": [{"id": "s3", "title": "Glommio repository (Datadog)",
                         "url": "https://github.com/DataDog/glommio",
                         "source_type": "official"}],
        })
        r = parse_research_result(raw)
        assert r.claims[0].source_ids == ["s3"]

    def test_url_as_source_id_repaired(self):
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low",
                        "source_ids": ["https://e.com"]}],
            "sources": [{"id": "s1", "title": "T", "url": "https://e.com",
                         "source_type": "reference"}],
        })
        r = parse_research_result(raw)
        assert r.claims[0].source_ids == ["s1"]

    def test_case_variant_source_id_repaired(self):
        # The 2026-07-07 euclid/pythagoras failure shape: the model cites
        # "S1" while the source's id is "s1". The id itself is an alias, so
        # the lowercased lookup repairs case variants.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low",
                        "source_ids": ["S1"]}],
            "sources": [{"id": "s1", "title": "T", "url": "https://e.com",
                         "source_type": "reference"}],
        })
        r = parse_research_result(raw)
        assert r.claims[0].source_ids == ["s1"]

    def test_unresolvable_source_id_kept_as_is(self):
        # Garbage refs stay put — downstream treats unknown ids as no-linkage.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low",
                        "source_ids": ["nonsense"]}],
            "sources": [{"id": "s1", "title": "T", "url": "https://e.com",
                         "source_type": "reference"}],
        })
        r = parse_research_result(raw)
        assert r.claims[0].source_ids == ["nonsense"]

    def test_source_without_id_backfilled_and_unknown_type(self):
        raw = ('{"claims": [], "sources": [{"title": "T", "url": "https://e.com", '
               '"source_type": "wikipedia"}]}')  # unknown type + no id
        r = parse_research_result(raw)
        assert r.sources[0].id == "s1"
        assert r.sources[0].source_type == "unknown"

    def test_empty_text_claims_dropped(self):
        raw = '{"claims": [{"text": "", "risk": "low"}, {"text": "real", "risk": "low"}]}'
        r = parse_research_result(raw)
        assert len(r.claims) == 1
        assert r.claims[0].text == "real"

    def test_off_enum_risk_normalized(self):
        # Models emit 'High', ints, descriptive risk — one off-enum value per
        # claim was discarding whole ledgers (errors scaled with claim count).
        for bad, want in [("High", "high"), ("high - speculative", "high"),
                          (2, "medium"), ("LOW", "low")]:
            raw = json.dumps({"claims": [{"id": "c1", "text": "x", "risk": bad}]})
            r = parse_research_result(raw)
            assert r is not None and r.claims[0].risk == want, f"{bad!r} → {want}"

    def test_capitalized_source_type_normalized(self):
        raw = '{"sources": [{"id": "s1", "title": "t", "url": "u", "source_type": "Reference"}]}'
        r = parse_research_result(raw)
        assert r.sources[0].source_type == "reference"

    def test_null_url_coerced_not_discarded(self):
        # The 2/3-worker-drop bug after the off-enum fix: a model emits `url: null`
        # for a source it couldn't link, and `str` rejected None — discarding the
        # whole worker ledger (`ValidationError on fields ['sources.0.url', ...]`).
        raw = '{"sources": [{"id": "s1", "title": "T", "url": null, "source_type": "news"}]}'
        r = parse_research_result(raw)
        assert r is not None
        assert r.sources[0].url == ""

    def test_null_title_coerced(self):
        raw = '{"sources": [{"id": "s1", "title": null, "url": "https://e.com"}]}'
        r = parse_research_result(raw)
        assert r is not None and r.sources[0].title == ""

    def test_int_url_coerced_to_str(self):
        raw = '{"sources": [{"id": "s1", "title": "T", "url": 123}]}'
        r = parse_research_result(raw)
        assert r is not None and r.sources[0].url == "123"

    def test_one_null_url_does_not_drop_sibling_sources(self):
        # The exact box shape: source 0 is fine, source 4 has url null. The good
        # sources must survive (previously the whole worker was discarded).
        raw = json.dumps({"sources": [
            {"id": "s0", "title": "A", "url": "https://a.com", "source_type": "news"},
            {"id": "s4", "title": "B", "url": None, "source_type": "blog"},
        ]})
        r = parse_research_result(raw)
        assert r is not None and len(r.sources) == 2

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

    def test_bad_verdict_normalized_to_irrelevant(self):
        # An off-enum verdict must NOT discard the whole result (that was the
        # over-strict behaviour that dropped worker ledgers); normalize to
        # 'irrelevant' (ignored downstream) and keep the rest.
        raw = json.dumps({"checks": [{"claim_id": "c1", "verdict": "made-up"}]})
        r = parse_factcheck_result(raw)
        assert isinstance(r, FactCheckResult)
        assert r.checks[0].verdict == "irrelevant"

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


class TestHedgePolicy:
    """Stage 4: the pure disposition table. The plan's three worked examples
    become the first three cases; the rest pin the rule ordering and the
    conservative empty-source default."""

    def test_official_low_risk_states_plainly(self):
        # "DeepSeek released R1 on 2025-01-20" — official, low risk → plain.
        c = Claim(id="c1", text="DeepSeek released R1 on 2025-01-20", risk="low")
        assert hedge_policy(c, {"official"}) == "state_plainly"

    def test_company_claim_is_attributed(self):
        # "Meta claims Maverick beats GPT-4o" — vendor's own benchmark → attribute.
        c = Claim(id="c1", text="Maverick beats GPT-4o", risk="medium")
        assert hedge_policy(c, {"company_claim"}) == "attribute_to_company"

    def test_needs_hedge_claim_hedges(self):
        # An ancient anecdote the checker flagged → hedge.
        c = Claim(id="c1", text="Euclid said 'no royal road'", needs_hedge=True)
        assert hedge_policy(c, {"reference"}) == "hedge"

    def test_company_claim_wins_over_other_signals(self):
        # company_claim is evaluated first: even high-risk + also-official, the
        # honest framing is to attribute the vendor's assertion.
        c = Claim(id="c1", text="our model is SOTA", risk="high")
        assert hedge_policy(c, {"company_claim", "official"}) == "attribute_to_company"

    def test_needs_hedge_beats_high_risk(self):
        c = Claim(id="c1", text="x", risk="high", needs_hedge=True)
        assert hedge_policy(c, {"reference"}) == "hedge"

    def test_high_risk_authoritative_hedges_or_cites(self):
        c = Claim(id="c1", text="x", risk="high")
        assert hedge_policy(c, {"official"}) == "hedge_or_cite_strongly"

    def test_non_authoritative_source_hedges(self):
        # news/blog/unknown on their own don't earn a plain statement.
        c = Claim(id="c1", text="x", risk="low")
        assert hedge_policy(c, {"news", "blog"}) == "hedge"

    def test_empty_sources_defaults_to_hedge(self):
        # A surviving claim the model never linked to a source → conservative hedge.
        c = Claim(id="c1", text="x", risk="low")
        assert hedge_policy(c, set()) == "hedge"

    def test_any_authoritative_type_states_plainly(self):
        for st in ("official", "primary_paper", "scholarly", "reference"):
            c = Claim(id="c1", text="x", risk="medium")
            assert hedge_policy(c, {st}) == "state_plainly"
