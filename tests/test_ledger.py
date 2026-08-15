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

    def test_decoder_is_told_source_ids_is_mandatory(self):
        # The whole point of the required stamp. `model_json_schema()` emits a
        # field WITH a default as optional, so the schema we hand Ollama
        # `format` let a constrained decoder close a Claim right after `text` —
        # which is how one worker returned ~41 claims and 5 cited URLs with
        # `source_ids` empty on all of them while its siblings linked fine.
        claim = inlined_schema(ResearchResult)["properties"]["claims"]["items"]
        assert "source_ids" in claim["required"]
        # And it must not simultaneously advertise a default to fall back on.
        assert "default" not in claim["properties"]["source_ids"]

    def test_id_stays_optional_for_the_decoder(self):
        # Deliberate: models routinely omit `id` and `_backfill_ids` assigns
        # positional ones. Requiring it would only invite a fabricated or
        # duplicated id, which poisons every downstream linkage at once.
        s = inlined_schema(ResearchResult)
        for obj in ("claims", "sources"):
            item = s["properties"][obj]["items"]
            assert "id" not in item["required"]
            assert item["properties"]["id"]["default"] == ""

    def test_required_stamp_does_not_make_parsing_stricter(self):
        # THE invariant. Generation is constrained; validation is not, and the
        # two must never be merged — a required field with no default once
        # discarded a whole worker's ledger (the 2/3-drop bug). A payload
        # omitting every field the decoder is now told to emit must still parse.
        raw = json.dumps({"claims": [{"text": "x"}], "sources": [{"title": "t"}]})
        r = parse_research_result(raw)
        assert r is not None
        assert len(r.claims) == 1 and len(r.sources) == 1
        assert r.claims[0].source_ids == []
        assert r.claims[0].risk == "medium"
        assert r.sources[0].source_type == "unknown"

    def test_required_stamp_survives_ref_inlining(self):
        # The stamp is applied where the $defs name is still known; a later
        # change to _inline_refs that drops the name would silently un-require
        # everything nested, leaving only the root stamped.
        s = inlined_schema(ResearchResult)
        txt = json.dumps(s)
        assert "$ref" not in txt and "$defs" not in txt
        assert s["required"] == ["claims", "sources"]
        assert s["properties"]["sources"]["items"]["required"] == [
            "title", "url", "source_type"]


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

    def test_src_and_source_number_refs_repaired(self):
        # The 2026-07-07 second-run shape (euclid/transformer/pythagoras):
        # a claim cites a positional source as "SRC-2" / "src_1" / "source3"
        # while the backfilled id is "s2" / "s1" / "s3". The punctuation-
        # stripped src{N}/source{N} alias resolves all these spellings.
        raw = json.dumps({
            "claims": [
                {"id": "c1", "text": "a", "risk": "low", "source_ids": ["SRC-2"]},
                {"id": "c2", "text": "b", "risk": "low", "source_ids": ["src_1"]},
                {"id": "c3", "text": "c", "risk": "low", "source_ids": ["source3"]},
            ],
            "sources": [
                {"id": "s1", "title": "T1", "url": "https://e.com/1",
                 "source_type": "reference"},
                {"id": "s2", "title": "T2", "url": "https://e.com/2",
                 "source_type": "reference"},
                {"id": "s3", "title": "T3", "url": "https://e.com/3",
                 "source_type": "reference"},
            ],
        })
        r = parse_research_result(raw)
        assert [c.source_ids for c in r.claims] == [["s2"], ["s1"], ["s3"]]

    def test_real_src_prefixed_id_not_shadowed_by_num_alias(self):
        # A genuine source whose id IS "src2" must win over the src{N} alias
        # for a positional "s2" source — exact-id match takes priority.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low",
                        "source_ids": ["src2"]}],
            "sources": [
                {"id": "s2", "title": "positional", "url": "https://e.com/a",
                 "source_type": "reference"},
                {"id": "src2", "title": "real", "url": "https://e.com/b",
                 "source_type": "reference"},
            ],
        })
        r = parse_research_result(raw)
        assert r.claims[0].source_ids == ["src2"]

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

    def test_unknown_arxiv_source_upgraded_to_primary_paper(self):
        # 2026-07-14 writer-A/B trace: researchers tagged arxiv.org/abs/1706.03762
        # ("Attention Is All You Need") as `unknown`, so hedge_policy hedged
        # settled facts. The domain upgrade repairs the type at parse time.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [{"id": "s1", "title": "Attention Is All You Need",
                         "url": "https://arxiv.org/abs/1706.03762",
                         "source_type": "unknown"}],
        })
        r = parse_research_result(raw)
        assert r.sources[0].source_type == "primary_paper"

    def test_unknown_gov_source_upgraded_to_official(self):
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [{"id": "s1", "title": "USGS", "source_type": "unknown",
                         "url": "https://pubs.usgs.gov/gip/dynamic/stripes.html"}],
        })
        r = parse_research_result(raw)
        assert r.sources[0].source_type == "official"

    def test_unknown_neurips_and_wikipedia_upgraded(self):
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [
                {"id": "s1", "title": "NeurIPS", "source_type": "unknown",
                 "url": "https://papers.neurips.cc/paper/7181-attention.pdf"},
                {"id": "s2", "title": "Wikipedia", "source_type": "unknown",
                 "url": "https://en.wikipedia.org/wiki/Attention_(machine_learning)"},
            ],
        })
        r = parse_research_result(raw)
        assert r.sources[0].source_type == "primary_paper"
        assert r.sources[1].source_type == "reference"

    def test_explicit_source_type_not_overridden_by_domain(self):
        # Only `unknown` is upgraded — a model that deliberately labels a .gov
        # page as `news` (e.g. a press release) keeps its choice.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [{"id": "s1", "title": "gov release", "source_type": "news",
                         "url": "https://example.gov/press/release"}],
        })
        r = parse_research_result(raw)
        assert r.sources[0].source_type == "news"

    def test_non_authoritative_domain_stays_unknown(self):
        # A blog that merely quotes a paper is not upgraded.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [{"id": "s1", "title": "blog", "source_type": "unknown",
                         "url": "https://medium.com/@x/attention-explained"}],
        })
        r = parse_research_result(raw)
        assert r.sources[0].source_type == "unknown"

    def test_lookalike_domain_not_upgraded(self):
        # `notarxiv.org` / `arxiv.org.evil.com` must not match the arxiv rule.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [
                {"id": "s1", "title": "a", "source_type": "unknown",
                 "url": "https://notarxiv.org/abs/1"},
                {"id": "s2", "title": "b", "source_type": "unknown",
                 "url": "https://arxiv.org.evil.com/abs/1"},
            ],
        })
        r = parse_research_result(raw)
        assert r.sources[0].source_type == "unknown"
        assert r.sources[1].source_type == "unknown"

    def test_domain_upgrade_reaches_hedge_policy(self):
        # End-to-end: a low-risk claim backed only by a `unknown`-tagged arxiv
        # source would hedge (rule 5); after the upgrade it states plainly. This
        # is the whole point of the fix — the writer-A/B over-hedge.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "softmax is applied row-wise",
                        "risk": "low", "source_ids": ["s1"]}],
            "sources": [{"id": "s1", "title": "Attention Is All You Need",
                         "url": "https://arxiv.org/abs/1706.03762",
                         "source_type": "unknown"}],
        })
        r = parse_research_result(raw)
        types = {s.source_type for s in r.sources if "c1" in s.supports}
        assert hedge_policy(r.claims[0], types) == "state_plainly"

    def test_urlless_authoritative_source_demoted(self):
        # 2026-08-13, three protocol runs: researchers emit a named authority as
        # if it were a fetched source. `reference` and `official` are in
        # `_AUTHORITATIVE_SOURCES`, so those claims were STATED PLAINLY on the
        # strength of a source nobody retrieved.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [
                {"id": "s1", "title": "Herodotus, Histories",
                 "source_type": "reference"},
                {"id": "s2", "title": "Meta Llama 4 Family Announcement",
                 "source_type": "official"},
                {"id": "s3", "title": "Plutarch (Ancient Writer)", "url": "   ",
                 "source_type": "scholarly"},
            ],
        })
        r = parse_research_result(raw)
        assert [s.source_type for s in r.sources] == ["unknown"] * 3

    def test_placeholder_url_string_does_not_confer_authority(self):
        # 2026-08-13 protocol run, `bio-pythagoras`: the demotion cleared every
        # url-less authoritative row in all ten cases EXCEPT one, which rendered
        # as `(reference) Aristoxenus and Dicaearchus Fragments — null`. The
        # model emitted the four-character STRING "null", not a JSON null, so
        # `_to_str_or_empty` never converted it and the first version's
        # `url.strip()` test saw a non-empty string. A bare title and a fragment
        # fail the same way.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [
                {"id": "s1", "title": "Aristoxenus fragments", "url": "null",
                 "source_type": "reference"},
                {"id": "s2", "title": "bare title", "url": "Herodotus, Histories",
                 "source_type": "official"},
                {"id": "s3", "title": "fragment", "url": "/entries/pythagoras/",
                 "source_type": "scholarly"},
                {"id": "s4", "title": "scheme only", "url": "https://",
                 "source_type": "primary_paper"},
            ],
        })
        r = parse_research_result(raw)
        assert [s.source_type for s in r.sources] == ["unknown"] * 4

    def test_urlless_non_authoritative_types_untouched(self):
        # Only membership of the authoritative set is at stake. `company_claim`
        # in particular must survive — it is not authoritative, and its job is to
        # force attribution, which is right with or without a URL.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [
                {"id": "s1", "title": "vendor benchmark",
                 "source_type": "company_claim"},
                {"id": "s2", "title": "a blog", "source_type": "blog"},
                {"id": "s3", "title": "a wire report", "source_type": "news"},
            ],
        })
        r = parse_research_result(raw)
        assert [s.source_type for s in r.sources] == [
            "company_claim", "blog", "news",
        ]

    def test_authoritative_source_with_url_survives_demotion(self):
        # The guard must not touch a real retrieved source — including one the
        # domain upgrade just promoted, since it runs after `_upgrade_source_types`.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [
                {"id": "s1", "title": "SEP", "source_type": "reference",
                 "url": "https://plato.stanford.edu/entries/pythagoras/"},
                {"id": "s2", "title": "AIAYN", "source_type": "unknown",
                 "url": "https://arxiv.org/abs/1706.03762"},
            ],
        })
        r = parse_research_result(raw)
        assert [s.source_type for s in r.sources] == ["reference", "primary_paper"]

    def test_urlless_demotion_reaches_hedge_policy(self):
        # End-to-end, and the whole point: a low-risk claim backed ONLY by
        # "Herodotus, Histories — no url" used to reach rule 4 and state plainly.
        # It must now fall through to rule 5 and hedge.
        raw = json.dumps({
            "claims": [{"id": "c1", "risk": "low", "source_ids": ["s1"],
                        "text": "Xenophanes wrote a satirical fragment"}],
            "sources": [{"id": "s1", "title": "Xenophanes, fragments",
                         "source_type": "reference"}],
        })
        r = parse_research_result(raw)
        types = {s.source_type for s in r.sources if "c1" in s.supports}
        assert hedge_policy(r.claims[0], types) == "hedge"

    def test_content_free_source_dropped(self):
        # The 2026-07-09 trace-run `w2_, untitled — no url` shape: the qwen
        # structuring pass emitted source rows with no title AND no url (a stray
        # token became the id). The fail-soft schema resurrects them; they back
        # nothing and render as broken. A source with neither title nor url is
        # dropped; the real one survives.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [
                {"id": ",", "source_type": "unknown"},          # content-free
                {"id": "s1", "title": "Euclid — Britannica",
                 "url": "https://britannica.com/euclid",
                 "source_type": "reference"},
            ],
        })
        r = parse_research_result(raw)
        assert [s.url for s in r.sources] == ["https://britannica.com/euclid"]

    def test_source_with_url_but_blank_title_kept(self):
        # The null-title case the schema deliberately tolerates: a real source
        # (has a url) with a blank title must NOT be dropped by the content-free
        # filter — only the entirely-empty rows go.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "x", "risk": "low"}],
            "sources": [{"id": "s1", "title": "", "url": "https://e.com/page",
                         "source_type": "reference"}],
        })
        r = parse_research_result(raw)
        assert len(r.sources) == 1
        assert r.sources[0].url == "https://e.com/page"

    def test_supports_backfilled_from_source_ids(self):
        # The 2026-07-09 `supports: none` shape: the model fills claim.source_ids
        # but leaves source.supports empty on every source. The source→claim
        # index is inverted from source_ids so the linkage is complete.
        raw = json.dumps({
            "claims": [
                {"id": "c1", "text": "a", "risk": "low", "source_ids": ["s1"]},
                {"id": "c2", "text": "b", "risk": "low", "source_ids": ["s1", "s2"]},
            ],
            "sources": [
                {"id": "s1", "title": "T1", "url": "https://e.com/1",
                 "source_type": "reference", "supports": []},
                {"id": "s2", "title": "T2", "url": "https://e.com/2",
                 "source_type": "reference", "supports": []},
            ],
        })
        r = parse_research_result(raw)
        by_id = {s.id: s for s in r.sources}
        assert by_id["s1"].supports == ["c1", "c2"]
        assert by_id["s2"].supports == ["c2"]

    def test_supports_backfill_unions_not_replaces(self):
        # If the model DID emit some supports, backfill only adds — never drops
        # or duplicates. s1 already claims c1; source_ids also links c1 → no dup.
        raw = json.dumps({
            "claims": [{"id": "c1", "text": "a", "risk": "low", "source_ids": ["s1"]}],
            "sources": [{"id": "s1", "title": "T", "url": "https://e.com",
                         "source_type": "reference", "supports": ["c1"]}],
        })
        r = parse_research_result(raw)
        assert r.sources[0].supports == ["c1"]

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
        # Needs a REAL url: this asserts case-normalization only, and a stub like
        # "u" would now be demoted to `unknown` by `_demote_urlless_authority`,
        # failing the assertion for an unrelated reason.
        raw = ('{"sources": [{"id": "s1", "title": "t",'
               ' "url": "https://plato.stanford.edu/", "source_type": "Reference"}]}')
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

    def test_fatal_errors_object_entries_coerced_not_dropped(self):
        # 2026-07-08 eval (bio-euclid, hist-library-alexandria): the model put
        # correction/conflict OBJECTS into fatal_errors (a list[str] field), so
        # Pydantic rejected the whole FactCheckResult and the writer silently got
        # NO CORRECTIONS while a fact-check with real checks was discarded. The
        # coercer flattens the dicts; crucially, `checks` must survive.
        euclid_shape = json.dumps({
            "checks": [{"claim_id": "w0_c3", "verdict": "unsupported"}],
            "fatal_errors": [
                {"claim_ids": ["w0_c3", "w0_c4"],
                 "message": "birthplace is unknown."},
            ],
        })
        r = parse_factcheck_result(euclid_shape)
        assert isinstance(r, FactCheckResult)
        assert len(r.checks) == 1  # the valid check is NOT lost
        assert r.checks[0].claim_id == "w0_c3"
        assert r.fatal_errors == ["birthplace is unknown."]

        # library shape: no message-like key → compact key=value flatten.
        library_shape = json.dumps({
            "checks": [{"claim_id": "w2_claim-1", "verdict": "supported"}],
            "fatal_errors": [
                {"claim_id": "w2_claim-10", "conflicting_claim_id": "w0_c6"},
            ],
        })
        r = parse_factcheck_result(library_shape)
        assert isinstance(r, FactCheckResult)
        assert len(r.checks) == 1
        assert "w2_claim-10" in r.fatal_errors[0]
        assert "w0_c6" in r.fatal_errors[0]

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


class TestHedgePolicyRespectsTheFactCheckVerdict:
    """Rule 3's exemption. The disposition block and the corrections block are
    built by separate functions from the same fact-check result; before this,
    only the corrections block read the verdicts, so a claim could arrive at the
    writer as CONFIRMED and HEDGE at once (run `113119`, 15 of 28 checked
    claims). An unchecked claim must behave exactly as it always did."""

    def test_a_supported_verdict_lets_an_authoritative_high_risk_claim_be_plain(self):
        # The Tokio-release case: risk high, backed by the official releases
        # page, and the checker returned `supported`. "Unless a strong source
        # backs it" has been answered — stop asking.
        c = Claim(id="c1", text="Tokio's latest release is v1.53.1", risk="high")
        assert hedge_policy(c, {"official"}, "supported") == "state_plainly"

    def test_a_supported_verdict_without_authority_still_hedges(self):
        # Verified against a blog is not plain-statement material — rule 5.
        c = Claim(id="c1", text="x", risk="high")
        assert hedge_policy(c, {"news", "blog"}, "supported") == "hedge"

    def test_a_supported_verdict_with_no_sources_at_all_still_hedges(self):
        # `w2_c11` in run `113119`: the checker returned `supported` for a claim
        # carrying zero sources. A verdict must not manufacture confidence the
        # ledger has no grounding for.
        c = Claim(id="c1", text="x", risk="high")
        assert hedge_policy(c, set(), "supported") == "hedge"

    def test_a_supported_verdict_does_not_override_company_attribution(self):
        # Verification changes whether a claim is true, not who asserted it.
        c = Claim(id="c1", text="our model is SOTA", risk="high")
        assert hedge_policy(c, {"company_claim", "official"}, "supported") == (
            "attribute_to_company"
        )

    def test_a_supported_verdict_does_not_override_needs_hedge(self):
        c = Claim(id="c1", text="x", risk="high", needs_hedge=True)
        assert hedge_policy(c, {"official"}, "supported") == "hedge"

    def test_a_non_supported_verdict_leaves_rule_3_alone(self):
        for v in ("needs_hedge", "conflicting", "irrelevant", None):
            c = Claim(id="c1", text="x", risk="high")
            assert hedge_policy(c, {"official"}, v) == "hedge_or_cite_strongly"

    def test_the_verdict_argument_is_optional(self):
        # Every pre-existing caller passes two arguments; the default must
        # reproduce the old behaviour exactly.
        c = Claim(id="c1", text="x", risk="high")
        assert hedge_policy(c, {"official"}) == "hedge_or_cite_strongly"
